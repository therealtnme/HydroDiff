"""
Tests for HydroDiff. Run with:

    cd /home/user/HydroDiff && PYTHONPATH=. python3 -m pytest -q test/test_hydrodiff.py

Everything uses tiny configs (16x16 images, 4x4 patches = 16 tokens, 2 layers of width 64) and a single CPU thread so
the file stays fast and does not compete with training jobs on the machine.
"""

import pytest
import torch


torch.set_num_threads(1)

from hydrodiff import HydroDiffConfig, HydroDiffForImageGeneration  # noqa: E402
from hydrodiff.modeling_hydrodiff import (  # noqa: E402
    X_PRED_MIN_T,
    HydroDiffTokenCache,
    gather_tokens,
    patchify,
)
from hydrodiff.optimization_hydrodiff import Muon, build_optimizers  # noqa: E402


TINY = dict(hidden_size=64, intermediate_size=128, num_attention_heads=2, num_hidden_layers=2, image_size=16, patch_size=4)
TINY_MOE = dict(num_experts=4, moe_intermediate_size=64)

# name -> (preset, overrides on top of TINY)
VARIANTS = {
    "dense": ("hydrodiff-s", {}),
    "moe-token": ("hydrodiff-s-moe", {**TINY_MOE, "moe_routing": "token"}),
    "moe-image": ("hydrodiff-s-moe", {**TINY_MOE, "moe_routing": "image"}),
    "moe-time": ("hydrodiff-s-moe", {**TINY_MOE, "moe_routing": "time"}),
}

STEPS = 8
LABELS = torch.tensor([0, 5, 9])


@pytest.fixture
def make_model():
    """
    Builds a tiny model and perturbs every parameter with small noise: with the adaLN-Zero init the block gates and the
    output projection would start at zero and most checks below would hold trivially.
    """

    def _make(variant: str = "dense", seed: int = 0, **overrides) -> HydroDiffForImageGeneration:
        preset, variant_overrides = VARIANTS[variant]
        config = HydroDiffConfig.from_preset(preset, **{**TINY, **variant_overrides, **overrides})
        torch.manual_seed(seed)
        model = HydroDiffForImageGeneration(config)
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.shape, generator=generator) * 0.05)
        return model.eval()

    return _make


def sample(model, seed: int = 0, labels: torch.Tensor = LABELS, **kwargs):
    kwargs.setdefault("steps", STEPS)
    kwargs.setdefault("min_token_steps", 0)  # 8 steps are too few for the guard to allow any speed-up
    return model.generate(labels=labels, generator=torch.Generator().manual_seed(seed), **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# 1. Active-token path
# ----------------------------------------------------------------------------------------------------------------------


def run_full_then_active(model, num_active: int = 6, batch: int = 3, seed: int = 0, cache_for_active=True):
    """Runs all tokens with a fresh cache, then a random subset of tokens per image through `index`."""
    cfg = model.config
    n = cfg.num_patches
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, n, cfg.patch_dim, generator=g)
    t = torch.rand(batch, n, generator=g)
    labels = torch.tensor([0, 7, cfg.num_classes])[:batch]  # includes the null (unconditional) class
    # a different, unsorted subset in every image
    index = torch.stack([torch.randperm(n, generator=g)[:num_active] for _ in range(batch)])
    cache = HydroDiffTokenCache()
    with torch.no_grad():
        full = model.model(x, t, labels, None, cache)
        active = model.model(
            gather_tokens(x, index), gather_tokens(t, index), labels, index, cache if cache_for_active else None
        )
    return full, active, index


@pytest.mark.parametrize("variant", ["dense", "moe-token", "moe-image", "moe-time"])
def test_active_tokens_match_full_run(make_model, variant):
    model = make_model(variant)
    full, active, index = run_full_then_active(model)
    torch.testing.assert_close(active.prediction, gather_tokens(full.prediction, index), atol=1e-5, rtol=0)
    torch.testing.assert_close(active.log_error, gather_tokens(full.log_error, index), atol=1e-5, rtol=0)

    # Negative control: without the cache the active tokens lose the other tokens as context, so the match above is
    # not vacuous (the outputs really depend on the cached keys and values).
    _, no_context, index = run_full_then_active(model, cache_for_active=False)
    assert (no_context.prediction - gather_tokens(full.prediction, index)).abs().max() > 1e-3


# ----------------------------------------------------------------------------------------------------------------------
# 2. Training forward
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["dense", "moe-token"])
def test_training_forward_backward_is_finite(make_model, variant):
    model = make_model(variant).train()
    g = torch.Generator().manual_seed(3)
    pixels = torch.rand(4, 3, 16, 16, generator=g) * 2 - 1
    labels = torch.tensor([0, 1, 2, 3])
    torch.manual_seed(0)  # sampled noise levels, noise and class dropout
    out = model(pixels, labels=labels)

    for value in (out.loss, out.denoise_loss, out.uncertainty_loss):
        assert value.ndim == 0 and torch.isfinite(value)
    torch.testing.assert_close(out.loss, out.denoise_loss + model.config.uncertainty_loss_weight * out.uncertainty_loss)
    assert out.prediction.shape == (4, model.config.num_patches, model.config.patch_dim)
    assert out.timesteps.shape == (4, model.config.num_patches)
    assert ((out.timesteps >= 0) & (out.timesteps <= 1)).all()

    out.loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), name
    assert sum(p.grad.abs().sum() for p in model.parameters()) > 0


@pytest.mark.parametrize("prediction_type", ["x", "v"])
def test_denoise_loss_is_masked_mean_over_noisy_tokens(make_model, prediction_type):
    model = make_model(prediction_type=prediction_type)  # eval mode: no class dropout
    cfg = model.config
    b, n = 3, cfg.num_patches
    g = torch.Generator().manual_seed(4)
    pixels = torch.rand(b, 3, 16, 16, generator=g) * 2 - 1
    labels = torch.tensor([1, 2, 3])
    timesteps = torch.rand(b, n, generator=g) * 0.9 + 0.05
    timesteps[:, 1] = 0.01  # below X_PRED_MIN_T: exercises the clip of the x-prediction velocity
    clean = torch.rand(b, n, generator=g) < 0.4
    clean[:, 0], clean[:, 1] = True, False  # every image has both clean and noisy tokens
    timesteps[clean] = 0.0
    noise = torch.randn(b, n, cfg.patch_dim, generator=g)

    out = model(pixels, labels=labels, timesteps=timesteps, noise=noise)

    x0 = patchify(pixels, cfg.patch_size)
    t = timesteps[..., None]
    x_t = (1 - t) * x0 + t * noise
    if prediction_type == "v":
        v_pred, v_target = out.prediction, noise - x0
    else:  # prediction and target share the clipped 1 / t, so a perfect prediction has zero loss at every t
        v_pred = (x_t - out.prediction) / t.clamp_min(X_PRED_MIN_T)
        v_target = (x_t - x0) / t.clamp_min(X_PRED_MIN_T)
    per_token = (v_pred - v_target).pow(2).mean(-1)
    expected = per_token[timesteps > 0].mean()
    torch.testing.assert_close(out.denoise_loss, expected)
    # the clean tokens' errors are not negligible: had they been counted, the loss would be different
    assert not torch.isclose(per_token.mean(), expected)

    # Re-drawing the noise of clean tokens changes their target (noise - x0) but not their input (x_t = x0 at t = 0);
    # since they contribute nothing, the predictions and the denoise loss stay the same.
    noise2 = noise.clone()
    noise2[clean] = 3 * torch.randn(int(clean.sum()), cfg.patch_dim, generator=g)
    out2 = model(pixels, labels=labels, timesteps=timesteps, noise=noise2)
    assert torch.equal(out2.prediction, out.prediction)
    torch.testing.assert_close(out2.denoise_loss, out.denoise_loss)


# ----------------------------------------------------------------------------------------------------------------------
# 3. Sampler identities
# ----------------------------------------------------------------------------------------------------------------------


def test_generate_non_adaptive_computes_every_token_and_ends_clean(make_model):
    model = make_model()
    out = sample(model, adaptive=False, solver="euler", churn=0.0, return_masks=True)
    assert out.compute == 1.0
    assert out.images.shape == (3, 3, 16, 16)
    assert out.noise_masks.shape == (STEPS + 1, 3, model.config.num_patches)
    assert out.update_masks.shape == (STEPS, 3, model.config.num_patches)
    assert torch.equal(out.noise_masks[-1], torch.zeros_like(out.noise_masks[-1]))
    assert out.update_masks.all()
    # one shared clock: all tokens of all images have the same noise level at every step
    assert (out.noise_masks == out.noise_masks[:, :1, :1]).all()


def test_generate_adaptive_saves_compute_and_shrinks_update_mask(make_model):
    model = make_model()
    out = sample(model, adaptive=True, return_masks=True)
    assert 0 < out.compute < 1
    active = out.update_masks.sum(dim=(1, 2))
    assert (active[1:] <= active[:-1]).all(), active.tolist()
    assert active[-1] < active[0], active.tolist()
    # every evaluated-and-moving token is counted in compute (padding with finished tokens can only add to it)
    assert out.compute >= out.update_masks.float().mean().item()
    # without reuse, a token is computed in a step exactly when it still carries noise
    assert torch.equal(out.update_masks, out.noise_masks[:-1] > 0)


def test_generate_skip_finished_false_computes_everything(make_model):
    model = make_model()
    out = sample(model, adaptive=True, skip_finished=False)
    assert out.compute == 1.0
    assert sample(model, adaptive=True).compute < 1  # the same sampler with skipping does save compute


def test_generate_reuse_lowers_compute(make_model):
    model = make_model()
    base = sample(model, adaptive=True, reuse=0.0)
    reused = sample(model, adaptive=True, reuse=0.3)
    assert reused.compute < base.compute
    assert torch.isfinite(reused.images).all()


def test_generate_is_deterministic_and_max_rate_one_is_uniform(make_model):
    model = make_model(max_rate=1)
    first = sample(model, adaptive=False)
    second = sample(model, adaptive=False)
    assert torch.equal(first.images, second.images)
    assert first.compute == second.compute == 1.0
    # the seed matters
    assert not torch.equal(first.images, sample(model, seed=1, adaptive=False).images)
    # with max_rate = 1 every token runs at the schedule's speed, so the adaptive sampler is the uniform one
    adaptive = sample(model, adaptive=True)
    assert adaptive.compute == 1.0
    torch.testing.assert_close(adaptive.images, first.images, atol=1e-6, rtol=0)


def test_generate_with_classifier_free_guidance(make_model):
    model = make_model()
    guided = sample(model, guidance_scale=2.0)
    assert guided.images.shape == (3, 3, 16, 16)
    assert torch.isfinite(guided.images).all()
    assert guided.images.min() >= -1 and guided.images.max() <= 1
    assert not torch.allclose(guided.images, sample(model, guidance_scale=1.0).images)


# ----------------------------------------------------------------------------------------------------------------------
# 4. Per-token clocks
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["euler", "dpm2m"])
def test_per_token_clocks(make_model, solver):
    model = make_model()
    out = sample(model, adaptive=True, solver=solver, return_masks=True)
    levels = out.noise_masks
    assert levels.shape == (STEPS + 1, 3, model.config.num_patches)
    assert torch.equal(levels[0], torch.ones_like(levels[0]))
    assert ((levels >= 0) & (levels <= 1)).all()
    assert (levels[1:] <= levels[:-1]).all(), "a token's noise level went up"
    assert torch.equal(levels[-1], torch.zeros_like(levels[-1]))
    # the clocks really are per token: at some step tokens of the same image sit at different noise levels
    assert (levels.amax(-1) - levels.amin(-1)).max() > 0.05
    assert torch.isfinite(out.images).all()


# ----------------------------------------------------------------------------------------------------------------------
# 5. save_pretrained / from_pretrained
# ----------------------------------------------------------------------------------------------------------------------


def test_save_and_load_round_trip(make_model, tmp_path):
    model = make_model("moe-token")
    with torch.no_grad():  # make the (non-parameter) balancing bias non-trivial so its persistence is checked too
        for module in model.modules():
            if hasattr(module, "expert_bias"):
                module.expert_bias.copy_(torch.linspace(-0.1, 0.1, module.expert_bias.numel()))
    model.save_pretrained(tmp_path)
    loaded = HydroDiffForImageGeneration.from_pretrained(tmp_path).eval()

    config_fields = lambda config: {k: v for k, v in config.to_dict().items() if k != "_name_or_path"}  # noqa: E731
    assert config_fields(loaded.config) == config_fields(model.config)
    original, restored = model.state_dict(), loaded.state_dict()
    assert original.keys() == restored.keys()
    for key in original:
        assert torch.equal(original[key], restored[key]), key

    a, b = sample(model, seed=7), sample(loaded, seed=7)
    assert torch.equal(a.images, b.images)
    assert a.compute == b.compute


# ----------------------------------------------------------------------------------------------------------------------
# 6. Config validation
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(prediction_type="eps"), "prediction_type"),
        (dict(moe_routing="expert"), "moe_routing"),
        (dict(max_rate=0.5), "max_rate"),
        (dict(image_size=18), "divisible"),
    ],
)
def test_invalid_config_raises(overrides, message):
    with pytest.raises(ValueError, match=message):
        HydroDiffConfig.from_preset("hydrodiff-s", **{**TINY, **overrides})


def test_valid_tiny_config():
    config = HydroDiffConfig.from_preset("hydrodiff-s-moe", **TINY, **TINY_MOE, max_rate=1)
    assert config.num_patches == 16 and config.patch_dim == 48 and config.head_dim == 32
    assert not config.is_moe_layer(0) and config.is_moe_layer(1)


# ----------------------------------------------------------------------------------------------------------------------
# 7. Optimizers
# ----------------------------------------------------------------------------------------------------------------------


def test_build_optimizers_muon(make_model):
    model = make_model("moe-token").train()
    frozen = model.model.patch_embed.bias
    frozen.requires_grad_(False)

    optimizers = build_optimizers(model, optimizer="muon")
    assert len(optimizers) == 2
    assert isinstance(optimizers[0], torch.optim.AdamW) and isinstance(optimizers[1], Muon)

    seen: dict[int, int] = {}
    for opt in optimizers:
        for group in opt.param_groups:
            assert group["base_lr"] == group["lr"]
            for p in group["params"]:
                seen[id(p)] = seen.get(id(p), 0) + 1
    names = {id(p): name for name, p in model.named_parameters()}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert set(seen) == trainable, f"missing: {[names[i] for i in trainable - set(seen)]}"
    assert all(count == 1 for count in seen.values()), [names[i] for i, c in seen.items() if c > 1]
    assert id(frozen) not in seen

    muon = {names[id(p)] for group in optimizers[1].param_groups for p in group["params"]}
    assert "model.layers.0.self_attn.qkv_proj.weight" in muon
    assert "model.layers.1.mlp.experts_gate_up" in muon
    assert not any("router" in name or "modulation" in name for name in muon)

    g = torch.Generator().manual_seed(5)
    pixels = torch.rand(2, 3, 16, 16, generator=g) * 2 - 1
    torch.manual_seed(0)
    model(pixels, labels=torch.tensor([1, 2])).loss.backward()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for opt in optimizers:
        opt.step()
    unchanged = [name for name, p in model.named_parameters() if p.requires_grad and torch.equal(p, before[name])]
    assert not unchanged, f"parameters not updated: {unchanged}"
    assert torch.equal(frozen, before["model.patch_embed.bias"])


# ----------------------------------------------------------------------------------------------------------------------
# Initialization (library bug, see reason)
# ----------------------------------------------------------------------------------------------------------------------


def test_fresh_model_is_zero_initialized():
    model = HydroDiffForImageGeneration(HydroDiffConfig.from_preset("hydrodiff-s", **TINY))
    state = model.state_dict()
    for key in (
        "model.proj_out.weight",
        "model.uncertainty_head.weight",
        "model.conditioning.modulation.weight",
        "model.conditioning.final_modulation.weight",
    ):
        assert torch.count_nonzero(state[key]) == 0, key


# ----------------------------------------------------------------------------------------------------------------------
# 8. Regressions from the code review
# ----------------------------------------------------------------------------------------------------------------------


def test_long_skips_connect_mirrored_layers(make_model):
    """With 4 layers, layer 2 reads the output of layer 1 and layer 3 the output of layer 0 (not their own input)."""
    model = make_model(num_hidden_layers=4, long_skips=True)
    seen, outputs = {}, {}
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(lambda mod, args, out, i=i: outputs.__setitem__(i, out))
        layer.register_forward_pre_hook(lambda mod, args, i=i: seen.__setitem__(i, args[7] if len(args) > 7 else None))
    x = torch.randn(2, model.config.num_patches, model.config.patch_dim)
    model.model(x, torch.rand(2, model.config.num_patches), torch.tensor([1, 2]))
    assert seen[0] is None and seen[1] is None
    assert seen[2] is outputs[1] and seen[3] is outputs[0]


def test_top1_router_gets_a_gradient(make_model):
    model = make_model("moe-token", num_experts_per_tok=1).train()
    model(torch.rand(4, 3, 16, 16) * 2 - 1, torch.tensor([0, 1, 2, 3])).loss.backward()
    router = model.model.layers[1].mlp.router.weight
    assert router.grad is not None and router.grad.abs().max() > 1e-6


def test_perfect_x_prediction_has_zero_loss_at_every_t(make_model):
    model = make_model(prediction_type="x")
    cfg = model.config
    pixels = torch.rand(2, 3, 16, 16) * 2 - 1
    timesteps = torch.linspace(0.001, 1, cfg.num_patches).expand(2, -1).contiguous()
    perfect = patchify(pixels, cfg.patch_size)
    model.model.forward = lambda x, t, labels=None, index=None, cache=None, **kwargs: type(
        "Out", (), {"prediction": perfect, "log_error": torch.zeros(t.shape)}
    )()
    out = model(pixels, torch.tensor([0, 1]), timesteps=timesteps)
    assert out.denoise_loss.abs() < 1e-10


def test_min_token_steps_caps_the_speed_up(make_model):
    """No token may take fewer steps than `min_token_steps`; with too few steps the noise mask is the dense sampler."""
    model = make_model()
    guarded = sample(model, steps=12, min_token_steps=12, return_masks=True)
    dense = sample(model, steps=12, adaptive=False)
    assert guarded.compute == 1.0
    torch.testing.assert_close(guarded.images, dense.images)
    out = sample(model, steps=24, min_token_steps=16, return_masks=True)
    taken = (out.noise_masks[:-1] > 0).sum(0)  # steps each token spent with noise left
    assert taken.min() >= 16 and out.compute < 1.0


def test_reuse_keeps_min_token_steps_evaluations(make_model):
    """`reuse` never leaves a token with fewer than `min_token_steps` evaluations, even on top of the noise mask."""
    model = make_model()
    out = sample(model, steps=24, min_token_steps=16, reuse=0.5, return_masks=True)
    evaluations = out.update_masks.sum(0)  # steps in which each token was run while it still had noise
    assert evaluations.min() >= 16
    unguarded = sample(model, steps=24, min_token_steps=0, max_rate=2.0, reuse=0.5, return_masks=True)
    assert unguarded.update_masks.sum(0).min() < 16 and out.compute > unguarded.compute


def test_guidance_paths_agree(make_model):
    """Batched classifier-free guidance equals the separate-pass path (a skipped block that does not exist)."""
    model = make_model()
    batched = sample(model, guidance_scale=2.0)
    separate = sample(model, guidance_scale=2.0, uncond_skip_layers=(99,))
    torch.testing.assert_close(batched.images, separate.images)
    weaker = sample(model, guidance_scale=2.0, uncond_skip_layers=(1,))
    assert not torch.allclose(weaker.images, batched.images)
    outside = sample(model, guidance_scale=2.0, guidance_interval=(2.0, 3.0))  # no token ever inside
    torch.testing.assert_close(outside.images, sample(model).images)


def test_quality_training_options_are_finite_and_train_their_parts(make_model):
    """Self-conditioning, contrastive flow matching and the dispersive loss give a finite loss and train the new parts."""
    model = make_model(self_condition=True, contrastive_weight=0.05, dispersive_weight=0.25).train()
    pixels, labels = torch.rand(4, 3, 16, 16) * 2 - 1, torch.tensor([0, 1, 2, 3])
    torch.manual_seed(0)
    for _ in range(4):  # the self-conditioning pass runs on half of the steps
        loss = model(pixels, labels).loss
        assert torch.isfinite(loss)
        loss.backward()
    assert model.model.self_cond_embed.weight.grad.abs().sum() > 0


def test_self_conditioned_active_tokens_match_full_run(make_model):
    """The previous estimate is gathered with the active tokens, so the active path still equals the full run."""
    model = make_model(self_condition=True)
    cfg = model.config
    g = torch.Generator().manual_seed(0)
    x, t = torch.randn(2, cfg.num_patches, cfg.patch_dim, generator=g), torch.rand(2, cfg.num_patches, generator=g)
    estimate = torch.randn(x.shape, generator=g)
    index, labels, cache = torch.tensor([[0, 5, 9], [3, 4, 15]]), torch.tensor([1, 2]), HydroDiffTokenCache()
    with torch.no_grad():
        full = model.model(x, t, labels, None, cache, self_cond=estimate)
        active = model.model(
            gather_tokens(x, index), gather_tokens(t, index), labels, index, cache, self_cond=gather_tokens(estimate, index)
        )
        plain = model.model(x, t, labels)
    torch.testing.assert_close(active.prediction, gather_tokens(full.prediction, index), atol=1e-5, rtol=0)
    assert (plain.prediction - full.prediction).abs().max() > 1e-3
    assert torch.isfinite(sample(model).images).all()


def test_guidance_schedule_and_cfg_zero(make_model):
    model = make_model()
    constant = sample(model, guidance_scale=2.0)
    linear = sample(model, guidance_scale=2.0, guidance_schedule="linear")
    zero = sample(model, guidance_scale=2.0, cfg_zero=True)
    for out in (linear, zero):
        assert torch.isfinite(out.images).all() and not torch.allclose(out.images, constant.images)
    torch.testing.assert_close(sample(model, cfg_zero=True).images, sample(model).images)  # no guidance: no change
