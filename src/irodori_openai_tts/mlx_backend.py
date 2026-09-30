"""Hybrid Irodori backend: mlx-audio DiT blocks, native MLX sampling and DACVAE.

Text/caption encoders, duration prediction and audio preprocessing stay in PyTorch.
Imports are lazy so the standard server remains usable without MLX/Metal.
"""

from __future__ import annotations

import inspect
import math
from types import FunctionType, MethodType

import numpy as np
import torch


def to_mlx(tensor):
    import mlx.core as mx

    if tensor is None:
        return None
    dtype = {torch.bfloat16: mx.bfloat16, torch.float16: mx.float16}.get(tensor.dtype)
    cpu = tensor.detach().cpu()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.float()
    array = mx.array(cpu.contiguous().numpy())
    return array.astype(dtype) if dtype is not None else array


def to_torch(array, *, device, dtype):
    import mlx.core as mx

    mx.eval(array)
    # NumPy cannot represent bfloat16. Convert at the framework boundary only.
    if array.dtype == mx.bfloat16:
        array = array.astype(mx.float32)
    return torch.from_numpy(np.array(array)).to(device=device, dtype=dtype)


def set_dit_precision(dit, precision):
    """Configure before compilation; FP16 GEMMs return to FP32 at each boundary.

    Changing FP16 back to FP32 cannot restore rounded weights. Reload the
    checkpoint for a full-precision baseline, and recompile after any change.
    """
    import mlx.core as mx
    import mlx.nn as nn

    if precision not in {"fp32", "fp16"}:
        raise ValueError("MLX DiT precision must be fp32 or fp16.")

    class MatmulLinear(nn.Linear):
        def __init__(self, source):
            nn.Module.__init__(self)
            dtype = mx.float16 if precision == "fp16" else mx.float32
            self.weight = source.weight.astype(dtype)
            if "bias" in source:
                self.bias = source.bias.astype(dtype)

        def __call__(self, x):
            return super().__call__(x.astype(self.weight.dtype)).astype(mx.float32)

    # Keep AdaLN, timestep conditioning, norms and small input/output projectors
    # in FP32. Only the large attention/MLP matrices use reduced precision.
    dit.apply(lambda value: value.astype(mx.float32))
    for block in dit.blocks:
        for module in (block.attention, block.mlp):
            replacements = {
                name: MatmulLinear(child)
                for name, child in module.children().items()
                if isinstance(child, nn.Linear)
            }
            module.update_modules(replacements)
    mx.eval(dit.parameters())
    dit.eval()


def make_dit(model, *, precision="fp32"):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_audio.tts.models.irodori_tts.model import (
        DiffusionBlock,
        RMSNorm,
        get_timestep_embedding,
        precompute_freqs_cis,
    )

    cfg = model.cfg

    class DiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(cfg.patched_latent_dim, cfg.model_dim)
            self.out_proj = nn.Linear(cfg.model_dim, cfg.patched_latent_dim)
            self.out_norm = RMSNorm(cfg.model_dim, cfg.norm_eps)
            self.cond_module = self.condition_layers()
            self.delta_cond_module = (
                self.condition_layers() if cfg.flow_parameterization == "meanflow" else None
            )
            self.blocks = [
                DiffusionBlock(
                    cfg.model_dim,
                    cfg.num_heads,
                    int(cfg.model_dim * cfg.mlp_ratio),
                    cfg.text_dim,
                    cfg.speaker_dim if cfg.use_speaker_condition_resolved else None,
                    cfg.adaln_rank,
                    cfg.norm_eps,
                    caption_ctx_dim=cfg.caption_dim_resolved if cfg.use_caption_condition else None,
                )
                for _ in range(cfg.num_layers)
            ]

        @staticmethod
        def condition_layers():
            return [
                nn.Linear(cfg.timestep_embed_dim, cfg.model_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cfg.model_dim, cfg.model_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cfg.model_dim, cfg.model_dim * 3, bias=False),
            ]

        def conditions(self, bundle):
            text, text_mask, speaker, speaker_mask, caption, caption_mask = bundle
            caches = []
            for block in self.blocks:
                a = block.attention
                caches.append(
                    (
                        a.get_kv_cache_text(text),
                        a.get_kv_cache_speaker(speaker) if speaker is not None else None,
                        a.get_kv_cache_caption(caption) if caption is not None else None,
                    )
                )
            mx.eval(caches)
            return (text_mask, speaker_mask, caption_mask, caches)

        def __call__(self, x, t, prepared, delta=None, speaker_scale=1.0, speaker_layers=None):
            x = x.astype(mx.float32)
            cond = get_timestep_embedding(t, cfg.timestep_embed_dim).astype(x.dtype)
            for layer in self.cond_module:
                cond = layer(cond)
            if self.delta_cond_module is not None:
                if delta is None:
                    raise ValueError("MeanFlow requires delta_t.")
                extra = get_timestep_embedding(delta, cfg.timestep_embed_dim).astype(x.dtype)
                for layer in self.delta_cond_module:
                    extra = layer(extra)
                cond = cond + extra
            cond = cond[:, None, :]
            h = self.in_proj(x)
            freqs = precompute_freqs_cis(cfg.model_dim // cfg.num_heads, h.shape[1])
            text_mask, speaker_mask, caption_mask, caches = prepared
            for i, (block, (text_kv, speaker_kv, caption_kv)) in enumerate(
                zip(self.blocks, caches, strict=True)
            ):
                if (
                    speaker_kv is not None
                    and speaker_scale != 1.0
                    and (speaker_layers is None or i < speaker_layers)
                ):
                    speaker_kv = tuple(v * speaker_scale for v in speaker_kv)
                h = block(
                    h,
                    cond,
                    text_mask,
                    freqs,
                    text_kv,
                    speaker_kv,
                    speaker_mask,
                    caption_kv,
                    caption_mask,
                )
            return self.out_proj(self.out_norm(h)).astype(x.dtype)

    dit = DiT()
    names = ("in_proj.", "out_proj.", "out_norm.", "cond_module.", "delta_cond_module.", "blocks.")
    weights = [
        (name, to_mlx(value))
        for name, value in model.state_dict().items()
        if name.startswith(names)
    ]
    dit.load_weights(weights, strict=True)
    set_dit_precision(dit, precision)
    return dit


def convert_codec_module(module):
    """Convert the actual active PyTorch codec graph, including its padding rules."""
    import mlx.core as mx
    from mlx_audio.codec.models.dacvae.codec import snake

    name = type(module).__name__
    if isinstance(module, torch.nn.Sequential):
        layers = [convert_codec_module(layer) for layer in module]

        def sequential(x):
            for layer in layers:
                x = layer(x)
            return x

        return sequential
    if name in {"Encoder", "EncoderBlock"}:
        return convert_codec_module(module.block)
    if name == "ResidualUnit":
        block = convert_codec_module(module.block)
        true_skip = module.true_skip

        def residual(x):
            y = block(x)
            pad = (x.shape[1] - y.shape[1]) // 2
            return y + (x[:, pad:-pad] if pad > 0 and not true_skip else x)

        return residual
    if name == "DecoderBlock":
        size = module._chunk_size
        # The upstream decoder only executes the non-watermark groups.
        active = [layer for i, layer in enumerate(module.block) if (i // size) % size == 0]
        return convert_codec_module(torch.nn.Sequential(*active))
    if name == "Snake1d":
        alpha = to_mlx(module.alpha).transpose(0, 2, 1)
        mx.eval(alpha)
        return lambda x: snake(x, alpha)
    if isinstance(module, torch.nn.Identity):
        return lambda x: x
    if isinstance(module, torch.nn.ELU):
        alpha = module.alpha
        return lambda x: mx.where(x > 0, x, alpha * mx.expm1(x))
    if isinstance(module, torch.nn.Tanh):
        return mx.tanh
    if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        transpose = isinstance(module, torch.nn.ConvTranspose1d)
        weight = module.weight.detach()
        if hasattr(module, "weight_g"):
            # weight_norm's cached .weight can be stale after a load_state_dict.
            weight = torch._weight_norm(module.weight_v.detach(), module.weight_g.detach(), 0)
        weight = (
            to_mlx(weight).transpose(1, 2, 0) if transpose else to_mlx(weight).transpose(0, 2, 1)
        )
        bias = to_mlx(module.bias)
        mx.eval(weight, bias) if bias is not None else mx.eval(weight)
        stride, dilation, padding = module.stride[0], module.dilation[0], module.padding[0]
        kernel, groups = module.kernel_size[0], module.groups
        output_padding = module.output_padding[0] if transpose else 0
        auto = getattr(module, "pad_mode", "none") != "none"
        causal = getattr(module, "causal", False)

        def convolution(x):
            if auto and not transpose:
                total = (kernel - 1) * dilation + 1 - stride
                frames = (x.shape[1] - ((kernel - 1) * dilation + 1) + total) / stride + 1
                ideal = (math.ceil(frames) - 1) * stride + kernel - total
                extra = ideal - x.shape[1]
                right = 0 if causal else extra // 2
                left = total if causal else total - right
                # PyTorch F.pad allows negative padding (crop) as well.
                x = x[:, max(0, -left) : x.shape[1] - max(0, -(right + extra))]
                x = mx.pad(x, [(0, 0), (max(0, left), max(0, right + extra)), (0, 0)])
            if transpose:
                y = mx.conv_transpose1d(
                    x,
                    weight,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    output_padding=output_padding,
                    groups=groups,
                )
            else:
                y = mx.conv1d(
                    x, weight, stride=stride, padding=padding, dilation=dilation, groups=groups
                )
            if bias is not None:
                y = y + bias
            if auto and transpose:
                total = kernel - stride
                right = total if causal else total // 2
                left = 0 if causal else total - right
                y = y[:, left : y.shape[1] - right]
            return y

        return convolution
    raise ValueError(f"Unsupported MLX codec module: {name}")


class MLXCodec:
    def __init__(self, codec):
        self.encoder = convert_codec_module(codec.model.encoder)
        self.in_proj = convert_codec_module(codec.model.quantizer.in_proj)
        self.out_proj = convert_codec_module(codec.model.quantizer.out_proj)
        decoder = codec.model.decoder
        self.decoder = convert_codec_module(torch.nn.Sequential(*decoder.model))
        # Irodori disables watermark injection but retains the mono output path.
        self.output = convert_codec_module(
            torch.nn.Sequential(*list(decoder.wm_model.encoder_block.pre)[:-1])
        )
        self.hop_length = int(codec.model.hop_length)

    def encode(self, waveform):
        import mlx.core as mx

        x = to_mlx(waveform).transpose(0, 2, 1)
        pad = (-x.shape[1]) % self.hop_length
        if pad:
            # Match DACVAE._pad's reflect padding (MLX codec defaults to zeros).
            if pad >= x.shape[1]:
                raise ValueError("Reference audio is too short for DACVAE reflect padding.")
            x = mx.concatenate([x, x[:, -pad - 1 : -1][:, ::-1]], axis=1)
        mean, _ = mx.split(self.in_proj(self.encoder(x)), 2, axis=-1)
        return mean.transpose(0, 2, 1)

    def decode(self, latent):
        return self.output(self.decoder(self.out_proj(to_mlx(latent)))).transpose(0, 2, 1)


class MLXSampler:
    def __init__(self, model, dit, *, compile_forward=False):
        self.model = model
        self.dit = dit
        self.forward = dit
        if compile_forward:
            self.recompile()

    def recompile(self):
        """Use a new function identity so a graph cannot retain old weights."""
        import mlx.core as mx

        def forward(*args, **kw):
            return self.dit(*args, **kw)

        self.forward = mx.compile(forward)

    def sample(self, *, meanflow=False, **kw):
        import mlx.core as mx

        from irodori_tts.rf import _make_rng

        model = self.model
        b = kw["text_input_ids"].shape[0]
        rng, rng_device = _make_rng(seed=kw.get("seed", 0), device=model.device)
        noise = torch.randn(
            (b, kw["sequence_length"], model.cfg.patched_latent_dim),
            device=rng_device,
            dtype=model.dtype,
            generator=rng,
        )
        # CFG, temporal rescaling and Euler accumulation stay FP32 even when
        # the DiT's large matrix multiplications use FP16.
        x = to_mlx(noise).astype(mx.float32)
        if kw.get("truncation_factor") is not None:
            x = x * kw["truncation_factor"]
        encode_keys = (
            "text_input_ids",
            "text_mask",
            "ref_latent",
            "ref_mask",
            "caption_input_ids",
            "caption_mask",
            "speaker_state_override",
            "speaker_mask_override",
            "speaker_uncond_mode",
        )
        encoded = model.encode_conditions(**{k: kw[k] for k in encode_keys if k in kw})
        cond = tuple(to_mlx(t) for t in encoded)
        uncond = tuple(mx.zeros_like(t) if t is not None else None for t in cond)
        if kw.get("speaker_uncond_mode", "mask") == "noise" and encoded[2] is not None:
            speaker_noise = torch.randn(
                encoded[2].shape, device=rng_device, dtype=model.dtype, generator=rng
            )
            speaker_noise = speaker_noise * encoded[2].std().to(rng_device).clamp_min(1e-6)
            uncond = (*uncond[:2], to_mlx(speaker_noise), mx.ones_like(cond[3]), *uncond[4:])
        steps = kw.get("num_steps", 4 if meanflow else 40)
        u = np.linspace(0, 1, steps + 1, dtype=np.float32)
        if not meanflow and kw.get("t_schedule_mode", "linear") == "sway":
            u = np.clip(u + kw.get("sway_coeff", -1) * (np.cos(0.5 * math.pi * u) + u - 1), 0, 1)
        schedule = (1 - u) * (1.0 if meanflow else 0.999)
        if not np.all(schedule[:-1] > schedule[1:]):
            raise ValueError("t_schedule must be strictly decreasing.")
        scales = {
            name: kw.get(f"cfg_scale_{name}", default)
            for name, default in (("text", 3.0), ("speaker", 5.0), ("caption", 3.0))
        }
        active = [
            n
            for n in scales
            if scales[n] > 0
            and (
                n == "text"
                or (n == "speaker" and model.cfg.use_speaker_condition_resolved)
                or (n == "caption" and cond[5] is not None and bool(mx.any(cond[5])))
            )
        ]
        mode = kw.get("cfg_guidance_mode", "independent")
        if (
            mode == "joint"
            and active
            and max(scales[n] for n in active) - min(scales[n] for n in active) > 1e-6
        ):
            raise ValueError("Joint CFG requires equal enabled guidance scales.")
        bundles = {"cond": cond, "joint": uncond}
        for name, offset in (("text", 0), ("speaker", 2), ("caption", 4)):
            bundles[name] = tuple(
                uncond[j] if j in (offset, offset + 1) else cond[j] for j in range(6)
            )
        prepared = {"cond": self.dit.conditions(cond)}
        if not meanflow and active:
            if mode == "independent":
                names = ["cond", *active]
                combined = tuple(
                    mx.concatenate([bundles[n][j] for n in names], axis=0)
                    if cond[j] is not None
                    else None
                    for j in range(6)
                )
                prepared["independent"] = self.dit.conditions(combined)
            else:
                for name in ["joint"] if mode == "joint" else active:
                    prepared[name] = self.dit.conditions(bundles[name])
        mx.eval(x)
        for i in range(steps):
            t, following = float(schedule[i]), float(schedule[i + 1])
            speaker_scale = kw.get("speaker_kv_scale")
            speaker_scale = (
                speaker_scale
                if speaker_scale is not None and t >= kw.get("speaker_kv_min_t", 0.9)
                else 1.0
            )

            def forward(value, name, *, t=t, following=following, speaker_scale=speaker_scale):
                time_dtype = mx.float32 if meanflow else value.dtype
                return self.forward(
                    value,
                    mx.full((value.shape[0],), t, dtype=time_dtype),
                    prepared[name],
                    delta=mx.full((value.shape[0],), t - following, dtype=time_dtype)
                    if meanflow
                    else None,
                    speaker_scale=speaker_scale,
                    speaker_layers=kw.get("speaker_kv_max_layers"),
                )

            use_cfg = (
                not meanflow
                and active
                and kw.get("cfg_min_t", 0.5) <= t <= kw.get("cfg_max_t", 1.0)
            )
            if use_cfg and mode == "independent":
                values = mx.split(
                    forward(mx.concatenate([x] * (len(active) + 1), axis=0), "independent"),
                    len(active) + 1,
                    axis=0,
                )
                velocity = values[0]
                for name, dropped in zip(active, values[1:], strict=True):
                    velocity = velocity + scales[name] * (values[0] - dropped)
            else:
                velocity = forward(x, "cond")
                if use_cfg:
                    name = "joint" if mode == "joint" else active[i % len(active)]
                    scale = scales[active[0]] if mode == "joint" else scales[name]
                    velocity = velocity + scale * (velocity - forward(x, name))
            if kw.get("rescale_k") is not None and kw.get("rescale_sigma") is not None and t < 1:
                snr = (1 - t) ** 2 / t**2
                sigma_sq = kw["rescale_sigma"] ** 2
                ratio = (snr * sigma_sq + 1) / (snr * sigma_sq / kw["rescale_k"] + 1)
                velocity = (ratio * ((1 - t) * velocity + x) - x) / (1 - t)
            x = x + velocity * (following - t)
            # Bound the lazy graph; all diffusion steps remain on Metal.
            mx.eval(x)
        return to_torch(x, device=model.device, dtype=model.dtype)


def install_mlx_backend(runtime, *, dit_precision="fp32"):
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "MLX requires Apple Silicon with Metal access. Install with uv sync --extra mlx."
        ) from exc
    if runtime.key.compile_model:
        raise ValueError("IRODORI_COMPILE_MODEL must be false with the MLX backend.")
    if not runtime.codec.deterministic_encode or not runtime.codec.deterministic_decode:
        raise ValueError("The MLX codec requires deterministic encode/decode.")
    if any(type(p).__module__.startswith("torchao") for p in runtime.model.parameters()):
        raise ValueError("The MLX backend requires an unquantized checkpoint.")
    dit = make_dit(runtime.model, precision=dit_precision)
    codec = MLXCodec(runtime.codec)
    sampler = MLXSampler(runtime.model, dit, compile_forward=True)
    # Bind a private copy of upstream synthesize's globals. Other runtimes keep
    # their original sampler; no process-wide monkeypatch or site-packages edits.
    original = inspect.unwrap(runtime.synthesize.__func__)
    namespace = dict(original.__globals__)
    namespace["sample_euler_rf_cfg"] = sampler.sample
    namespace["sample_euler_meanflow"] = lambda **kw: sampler.sample(meanflow=True, **kw)
    function = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    function.__kwdefaults__ = original.__kwdefaults__
    runtime.synthesize = MethodType(function, runtime)

    def reject_lora(_self, adapter_path, **kw):
        if _self._resolve_lora_adapter_path(adapter_path) is not None:
            raise ValueError(
                "Dynamic LoRA is not supported by the MLX backend; use a merged checkpoint."
            )
        from contextlib import nullcontext

        return nullcontext()

    runtime._prepare_lora_for_request = MethodType(reject_lora, runtime)

    # encode_waveform's deterministic path calls encoder then quantizer.in_proj.
    # Keep its preprocessing, but substitute the complete MLX codec encode path
    # at the model encoder boundary using a paired projection below.
    # A facade exposes the upstream normalization API without running PyTorch convolutions.
    class CodecModel:
        hop_length = codec.hop_length

        def _pad(self, wav):
            return wav

        class Quantizer:
            @staticmethod
            def in_proj(mean):
                return torch.cat((mean, torch.zeros_like(mean)), dim=1)

        quantizer = Quantizer()

        @staticmethod
        def encoder(wav):
            return to_torch(
                codec.encode(wav), device=runtime.codec.device, dtype=runtime.codec.dtype
            )

    runtime.codec.model = CodecModel()
    runtime.codec.decode_latent = lambda latent: to_torch(
        codec.decode(latent), device=runtime.codec.device, dtype=runtime.codec.dtype
    )
    runtime.mlx_dit = dit
    runtime.mlx_sampler = sampler
    runtime.mlx_dit_precision = dit_precision
    runtime.mlx_codec = codec
    # Only condition encoding/duration use the Torch model now. Release duplicate DiT weights.
    for name in ("blocks", "cond_module", "delta_cond_module", "in_proj", "out_norm", "out_proj"):
        module = getattr(runtime.model, name, None)
        if module is not None:
            module.to("meta")
    if runtime.model_device.type == "mps":
        torch.mps.empty_cache()
    mx.clear_cache()
