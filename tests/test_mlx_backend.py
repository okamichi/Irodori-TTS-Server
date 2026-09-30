"""Metal parity checks. Run with IRODORI_TEST_MLX=1 on Apple Silicon."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("IRODORI_TEST_MLX") != "1", reason="opt-in Metal tests"
)


def tiny_model(flow="rf_velocity"):
    from irodori_tts.config import ModelConfig
    from irodori_tts.model import TextToLatentRFDiT

    torch.manual_seed(71)
    cfg = ModelConfig(
        model_dim=32,
        num_layers=2,
        num_heads=4,
        text_dim=16,
        text_layers=1,
        text_heads=4,
        text_vocab_size=32,
        speaker_dim=16,
        speaker_layers=1,
        speaker_heads=4,
        caption_dim=16,
        caption_layers=1,
        caption_heads=4,
        caption_vocab_size=32,
        use_caption_condition=True,
        use_speaker_condition=True,
        timestep_embed_dim=16,
        adaln_rank=8,
        latent_dim=4,
        flow_parameterization=flow,
    )
    model = TextToLatentRFDiT(cfg).eval()
    torch.nn.init.normal_(model.out_proj.weight, std=0.1)
    if flow == "meanflow":
        torch.nn.init.normal_(model.delta_cond_module[-1].weight, std=0.05)
    return model


def sample_args():
    return {
        "text_input_ids": torch.tensor([[2, 4, 6]]),
        "text_mask": torch.tensor([[True, True, False]]),
        "ref_latent": torch.randn(1, 5, 4),
        "ref_mask": torch.ones(1, 5, dtype=torch.bool),
        "caption_input_ids": torch.tensor([[2, 3]]),
        "caption_mask": torch.tensor([[True, False]]),
        "sequence_length": 7,
        "num_steps": 4,
        "seed": 19,
    }


def test_dit_matches_torch_with_masked_contexts():
    import mlx.core as mx

    from irodori_openai_tts.mlx_backend import make_dit, to_mlx

    model = tiny_model()
    dit = make_dit(model)
    args = sample_args()
    with torch.inference_mode():
        bundle = model.encode_conditions(
            **{k: v for k, v in args.items() if k not in {"sequence_length", "num_steps", "seed"}}
        )
        x = torch.randn(1, 7, 4)
        t = torch.tensor([0.6])
        expected = model.forward_with_encoded_conditions(x, t, *bundle)
        result = dit(to_mlx(x), to_mlx(t), dit.conditions(tuple(to_mlx(v) for v in bundle)))
        mx.eval(result)
    np.testing.assert_allclose(np.array(result), expected.numpy(), rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("mode", ["independent", "joint", "alternating"])
@pytest.mark.parametrize("sway", [False, True])
def test_rf_sampler_matches_torch(mode, sway):
    from irodori_openai_tts.mlx_backend import MLXSampler, make_dit
    from irodori_tts.rf import sample_euler_rf_cfg

    model = tiny_model()
    args = sample_args()
    args.update(
        cfg_guidance_mode=mode,
        cfg_scale_text=2.0,
        cfg_scale_speaker=2.0,
        cfg_scale_caption=2.0,
        t_schedule_mode="sway" if sway else "linear",
        speaker_kv_scale=1.1,
        speaker_kv_min_t=0.9,
        rescale_k=1.1,
        rescale_sigma=0.8,
        truncation_factor=0.9,
    )
    with torch.inference_mode():
        expected = sample_euler_rf_cfg(model=model, **args)
        result = MLXSampler(model, make_dit(model)).sample(model=model, **args)
    torch.testing.assert_close(result, expected, rtol=8e-4, atol=8e-5)


def test_meanflow_sampler_matches_torch():
    from irodori_openai_tts.mlx_backend import MLXSampler, make_dit
    from irodori_tts.meanflow import sample_euler_meanflow

    model = tiny_model("meanflow")
    args = sample_args()
    with torch.inference_mode():
        expected = sample_euler_meanflow(model=model, **args)
        result = MLXSampler(model, make_dit(model)).sample(meanflow=True, model=model, **args)
    torch.testing.assert_close(result, expected, rtol=8e-4, atol=8e-5)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("auto,causal", [(False, False), (True, False), (True, True)])
def test_codec_convolutions_match_torch(transpose, auto, causal):
    import mlx.core as mx
    from dacvae.nn.layers import NormConv1d, NormConvTranspose1d

    from irodori_openai_tts.mlx_backend import convert_codec_module, to_mlx

    conv = NormConvTranspose1d if transpose else NormConv1d
    torch.manual_seed(5)
    module = conv(
        4, 6, kernel_size=6, stride=3, pad_mode="auto" if auto else "none", causal=causal
    ).eval()
    x = torch.randn(2, 4, 13)
    with torch.inference_mode():
        expected = module(x).transpose(1, 2).numpy()
        result = convert_codec_module(module)(to_mlx(x).transpose(0, 2, 1))
        mx.eval(result)
    np.testing.assert_allclose(np.array(result), expected, rtol=3e-4, atol=2e-5)


def test_codec_encode_decode_matches_torch():
    import mlx.core as mx
    from dacvae import DACVAE

    from irodori_openai_tts.mlx_backend import MLXCodec
    from irodori_tts.codec import DACVAECodec

    torch.manual_seed(8)
    model = DACVAE(
        encoder_dim=4,
        encoder_rates=[2, 4],
        latent_dim=16,
        decoder_dim=48,
        decoder_rates=[4, 2],
        codebook_dim=4,
        sample_rate=48000,
    ).eval()
    model.decoder.alpha = 0.0
    model.decoder.watermark = lambda x, message=None: (
        model.decoder.wm_model.encoder_block.forward_no_conv(x)
    )
    codec = DACVAECodec(
        model=model,
        sample_rate=48000,
        latent_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        deterministic_encode=True,
        deterministic_decode=True,
        normalize_db=None,
    )
    mlx_codec = MLXCodec(codec)
    wav = torch.randn(1, 1, 123)
    with torch.inference_mode():
        expected_latent = codec.encode_waveform(wav, 48000, ensure_max=False)
        latent = mlx_codec.encode(wav)
        mx.eval(latent)
        np.testing.assert_allclose(
            np.array(latent).transpose(0, 2, 1), expected_latent.numpy(), rtol=5e-4, atol=4e-5
        )
        expected_audio = codec.decode_latent(expected_latent)
        audio = mlx_codec.decode(expected_latent)
        mx.eval(audio)
        np.testing.assert_allclose(np.array(audio), expected_audio.numpy(), rtol=5e-4, atol=4e-5)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_compiled_sampler_matches_eager(precision):
    from irodori_openai_tts.mlx_backend import MLXSampler, make_dit

    model = tiny_model()
    dit = make_dit(model, precision=precision)
    args = sample_args()
    with torch.inference_mode():
        expected = MLXSampler(model, dit).sample(**args)
        actual = MLXSampler(model, dit, compile_forward=True).sample(**args)
    torch.testing.assert_close(actual, expected, rtol=8e-4, atol=8e-5)


def test_recompile_uses_updated_precision():
    from irodori_openai_tts.mlx_backend import MLXSampler, make_dit, set_dit_precision

    model = tiny_model()
    args = sample_args()
    sampler = MLXSampler(model, make_dit(model), compile_forward=True)
    with torch.inference_mode():
        baseline = sampler.sample(**args)
        set_dit_precision(sampler.dit, "fp16")
        sampler.recompile()
        actual = sampler.sample(**args)
        expected = MLXSampler(model, make_dit(model, precision="fp16")).sample(**args)
    torch.testing.assert_close(actual, expected, rtol=8e-4, atol=8e-5)
    assert not torch.equal(actual, baseline)


@pytest.mark.parametrize("flow", ["rf_velocity", "meanflow"])
@pytest.mark.parametrize("mode", ["independent", "joint", "alternating"])
def test_fp16_sampler_preserves_fp32_updates_and_stays_close(flow, mode):
    import mlx.core as mx

    from irodori_openai_tts.mlx_backend import MLXSampler, make_dit

    model = tiny_model(flow)
    dit = make_dit(model, precision="fp16")
    block = dit.blocks[0]
    assert block.attention.wq.weight.dtype == mx.float16
    assert block.mlp.w1.weight.dtype == mx.float16
    assert block.attention.q_norm.weight.dtype == mx.float32
    assert block.attention_adaln.shift_down.weight.dtype == mx.float32
    assert dit.in_proj.weight.dtype == mx.float32
    assert dit.cond_module[0].weight.dtype == mx.float32
    args = sample_args()
    args.update(
        num_steps=40 if flow == "rf_velocity" else 4,
        cfg_guidance_mode=mode,
        cfg_scale_text=2.0,
        cfg_scale_speaker=2.0,
        cfg_scale_caption=2.0,
        t_schedule_mode="sway",
        speaker_kv_scale=1.1,
        rescale_k=1.1,
        rescale_sigma=0.8,
    )
    sampler = MLXSampler(model, dit, compile_forward=True)
    compiled = sampler.forward
    observed_dtypes = []

    def observe(x, *args, **kw):
        observed_dtypes.append(x.dtype)
        result = compiled(x, *args, **kw)
        assert result.dtype == mx.float32
        return result

    sampler.forward = observe
    with torch.inference_mode():
        expected = MLXSampler(model, make_dit(model)).sample(meanflow=flow == "meanflow", **args)
        actual = sampler.sample(meanflow=flow == "meanflow", **args)
    assert observed_dtypes and all(dtype == mx.float32 for dtype in observed_dtypes)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=4e-3, atol=2e-3)


def test_fp16_gemm_upcasts_before_residual_and_norm():
    import mlx.core as mx

    from irodori_openai_tts.mlx_backend import make_dit

    dit = make_dit(tiny_model(), precision="fp16")
    block = dit.blocks[0]
    x = mx.full((1, 2, 32), 1000.0, dtype=mx.float32)
    projected = block.attention.wq(x)
    assert projected.dtype == mx.float32
    normalized = block.attention.q_norm(projected.reshape(1, 2, 4, 8))
    assert normalized.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(normalized)))
