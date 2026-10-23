import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_hydrodiff import HydroDiffConfig


X_PRED_MIN_T = 0.05  # x-prediction divides by t to get a velocity; training clips t here (as in JiT)
DISPERSIVE_TEMPERATURE = 0.5
TIME_FREQUENCIES = 256


# --------------------------------------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------------------------------------


class HydroDiffRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) if affine else None
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        hidden_states = hidden_states.to(input_dtype)
        return hidden_states if self.weight is None else self.weight * hidden_states


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(batch, channels, height, width) -> (batch, tokens, channels * patch_size**2), tokens in row-major order."""
    b, c, h, w = images.shape
    p = patch_size
    x = images.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // p) * (w // p), c * p * p)


def unpatchify(tokens: torch.Tensor, patch_size: int, channels: int) -> torch.Tensor:
    b, n, _ = tokens.shape
    side, p = math.isqrt(n), patch_size
    x = tokens.reshape(b, side, side, channels, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, channels, side * p, side * p)


def gather_tokens(x: torch.Tensor, index: torch.Tensor | None) -> torch.Tensor:
    """Picks `index` (batch, active) out of the token axis of `x` (batch, tokens, ...)."""
    if index is None:
        return x
    shape = index.shape + x.shape[2:]
    return x.gather(1, index.view(*index.shape, *([1] * (x.ndim - 2))).expand(shape))


def scatter_tokens(x: torch.Tensor, index: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
    if index is None:
        return values
    return x.scatter(1, index.view(*index.shape, *([1] * (x.ndim - 2))).expand(values.shape), values)


def shift_time(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Resolution shift of a noise level (SD3): spends more of the schedule at high noise when `shift > 1`."""
    return t if shift == 1 else shift * t / (1 + (shift - 1) * t)


def log_snr(t: torch.Tensor) -> torch.Tensor:
    """log(alpha / sigma) of the rectified-flow path, alpha = 1 - t and sigma = t."""
    return torch.log((1 - t).clamp_min(1e-12)) - torch.log(t.clamp_min(1e-12))


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rotary_2d(config: HydroDiffConfig, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Axial 2D rotary angles, (tokens, head_dim): half of each head rotates with the row, half with the column."""
    side, quarter = config.patches_per_side, config.head_dim // 4
    inv_freq = 1.0 / config.rope_theta ** (torch.arange(quarter, device=device, dtype=torch.float32) / quarter)
    pos = torch.arange(side, device=device, dtype=torch.float32)
    rows, cols = pos.repeat_interleave(side), pos.repeat(side)
    angles = torch.cat([rows[:, None] * inv_freq, cols[:, None] * inv_freq], dim=-1)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def timestep_embedding(t: torch.Tensor, dim: int = TIME_FREQUENCIES, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000)[..., None] * freqs
    return torch.cat([args.cos(), args.sin()], dim=-1)


class HydroDiffTokenCache:
    """
    Keys and values of every token, per layer, from the last step that computed the token. A step that runs only some
    tokens writes their fresh keys and values over the cached ones and attends over the whole image, so tokens that are
    not computed still serve as context.
    """

    def __init__(self):
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}
        self.router_inputs: dict[int, torch.Tensor] = {}

    def update(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor, index: torch.Tensor | None):
        if index is None or layer_idx not in self.keys:
            self.keys[layer_idx], self.values[layer_idx] = keys.clone(), values.clone()
            return keys, values
        slots = index[:, None, :, None].expand_as(keys)
        self.keys[layer_idx].scatter_(2, slots, keys)
        self.values[layer_idx].scatter_(2, slots, values)
        return self.keys[layer_idx], self.values[layer_idx]

    def update_router_inputs(self, layer_idx: int, hidden_states: torch.Tensor, index: torch.Tensor | None):
        """Every token's latest input to an image-routed MoE layer, so the router can average the whole image."""
        if index is None or layer_idx not in self.router_inputs:
            self.router_inputs[layer_idx] = hidden_states.clone()
        else:
            self.router_inputs[layer_idx].scatter_(1, index[..., None].expand_as(hidden_states), hidden_states)
        return self.router_inputs[layer_idx]


# --------------------------------------------------------------------------------------------------------------------
# Token mixer and channel mixers
# --------------------------------------------------------------------------------------------------------------------


class HydroDiffAttention(nn.Module):
    """Bidirectional attention with QK-norm and axial 2D rotary positions; queries come only from the active tokens."""

    def __init__(self, config: HydroDiffConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads, self.head_dim = config.num_attention_heads, config.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = HydroDiffRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = HydroDiffRMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, index=None, cache=None):
        b, a, d = hidden_states.shape
        q, k, v = self.qkv_proj(hidden_states).view(b, a, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v, index)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).reshape(b, a, d))


class HydroDiffMLP(nn.Module):
    def __init__(self, config: HydroDiffConfig, intermediate_size: int | None = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, timesteps=None, index=None, cache=None):
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(self.activation_fn(gate) * up)


class HydroDiffMoE(HydroDiffMLP):
    """
    Fine-grained mixture of experts: one always-on shared expert (the inherited MLP) plus `num_experts` routed experts,
    top-k by sigmoid affinity. Load balance comes from a per-expert bias that only affects *which* experts are chosen,
    nudged after each training step towards equal load (auxiliary-loss-free). With `moe_routing="token"` every token
    picks its experts; with `"image"` the router reads the mean of the image's tokens and the whole image uses the chosen
    experts for this layer and this step; with `"time"` there is no router: each token goes to the expert of its own
    noise level (`num_experts` equal bins of `t`).
    """

    def __init__(self, config: HydroDiffConfig, layer_idx: int):
        super().__init__(config, intermediate_size=config.moe_intermediate_size)
        self.layer_idx = layer_idx
        E, I, d = config.num_experts, config.moe_intermediate_size, config.hidden_size
        self.num_experts, self.top_k, self.routing = E, config.num_experts_per_tok, config.moe_routing
        self.bias_rate = config.router_bias_update_rate
        self.router = nn.Linear(d, E, bias=False)
        self.expert_bias = nn.Buffer(torch.zeros(E))
        self.experts_gate_up = nn.Parameter(torch.empty(E, 2 * I, d))
        self.experts_down = nn.Parameter(torch.empty(E, d, I))

    def expert(self, e: int, x: torch.Tensor) -> torch.Tensor:
        gate, up = F.linear(x, self.experts_gate_up[e]).chunk(2, dim=-1)
        return F.linear(self.activation_fn(gate) * up, self.experts_down[e])

    def forward(self, hidden_states, timesteps=None, index=None, cache=None):
        shared = super().forward(hidden_states)
        x = hidden_states if self.routing == "image" else hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.routing == "time":
            chosen = (timesteps.reshape(-1, 1) * self.num_experts).long().clamp(0, self.num_experts - 1)
            weights = torch.ones(chosen.shape, dtype=x.dtype, device=x.device)
        else:
            if self.routing == "image":  # the mean of the whole image, with cached inputs for tokens not run this step
                whole = x if cache is None else cache.update_router_inputs(self.layer_idx, x, index)
                scores = torch.sigmoid(self.router(whole.mean(1)).float())
            else:
                scores = torch.sigmoid(self.router(x).float())
            chosen = (scores + self.expert_bias).topk(self.top_k, dim=-1).indices
            weights = scores.gather(-1, chosen)
            if self.top_k > 1:  # with one expert the normalised weight would be exactly 1, and the router untrainable
                weights = weights / weights.sum(-1, keepdim=True)
            weights = weights.to(x.dtype)
        experts = chosen.unique().tolist()
        if len(experts) == 1 and self.top_k == 1:  # one expert for everything (e.g. every token at the same noise level)
            out = self.expert(experts[0], x) * weights.view(-1, *([1] * (x.ndim - 1)))
        else:
            out = torch.zeros_like(x)
            for e in experts:
                row, slot = (chosen == e).nonzero(as_tuple=True)
                w = weights[row, slot].view(-1, *([1] * (x.ndim - 1)))
                out.index_add_(0, row, self.expert(e, x[row]) * w)
        if self.training and self.routing != "time":
            with torch.no_grad():
                load = torch.bincount(chosen.flatten(), minlength=self.num_experts).float()
                self.expert_bias.add_(self.bias_rate * torch.sign(load.mean() - load))
        return shared + out.view_as(hidden_states)


# --------------------------------------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------------------------------------


class HydroDiffConditioning(nn.Module):
    """
    Per-token conditioning: each token's own noise level plus the class label. One shared projection turns it into the
    six modulation vectors of every block (adaLN-single); blocks add their own learned offsets. Running the projection
    once instead of once per layer is what makes per-token (rather than per-image) modulation affordable.
    """

    def __init__(self, config: HydroDiffConfig):
        super().__init__()
        d = config.hidden_size
        self.time_mlp = nn.Sequential(nn.Linear(TIME_FREQUENCIES, d), nn.SiLU(), nn.Linear(d, d))
        self.class_embed = nn.Embedding(config.num_classes + 1, d) if config.num_classes else None
        self.modulation = nn.Linear(d, 6 * d)
        self.final_modulation = nn.Linear(d, 2 * d)
        self.modulation.zero_init = self.final_modulation.zero_init = True

    def forward(self, timesteps: torch.Tensor, labels: torch.Tensor | None):
        cond = self.time_mlp(timestep_embedding(timesteps))
        if self.class_embed is not None:
            cond = cond + self.class_embed(labels)[:, None]
        cond = F.silu(cond)
        return self.modulation(cond).unflatten(-1, (6, -1)), self.final_modulation(cond).chunk(2, dim=-1)


class HydroDiffBlock(nn.Module):
    def __init__(self, config: HydroDiffConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = HydroDiffRMSNorm(config.hidden_size, config.rms_norm_eps, affine=False)
        self.self_attn = HydroDiffAttention(config, layer_idx)
        self.mlp_norm = HydroDiffRMSNorm(config.hidden_size, config.rms_norm_eps, affine=False)
        self.mlp = HydroDiffMoE(config, layer_idx) if config.is_moe_layer(layer_idx) else HydroDiffMLP(config)
        self.modulation = nn.Parameter(torch.zeros(6, config.hidden_size))
        # U-ViT long skip: the second half of the blocks also reads the output of its mirror block in the first half
        has_skip = config.long_skips and layer_idx >= config.num_hidden_layers - config.num_hidden_layers // 2
        self.skip_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False) if has_skip else None

    def forward(self, hidden_states, modulation, cos, sin, timesteps, index=None, cache=None, skip=None):
        if self.skip_proj is not None:
            hidden_states = self.skip_proj(torch.cat([hidden_states, skip], dim=-1))
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (modulation + self.modulation).unbind(-2)
        h = self.attn_norm(hidden_states) * (1 + scale_a) + shift_a
        hidden_states = hidden_states + gate_a * self.self_attn(h, cos, sin, index, cache)
        h = self.mlp_norm(hidden_states) * (1 + scale_m) + shift_m
        return hidden_states + gate_m * self.mlp(h, timesteps, index, cache)


# --------------------------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class HydroDiffModelOutput(ModelOutput):
    """
    prediction: the clean patches (`prediction_type="x"`) or the velocity (`"v"`) for the tokens that were run.
    log_error: predicted log mean squared error of each token's clean-patch estimate - the source of the noise mask.
    """

    prediction: torch.Tensor | None = None
    log_error: torch.Tensor | None = None
    features: torch.Tensor | None = None


@dataclass
class HydroDiffOutput(ModelOutput):
    loss: torch.Tensor | None = None
    denoise_loss: torch.Tensor | None = None
    uncertainty_loss: torch.Tensor | None = None
    prediction: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None


@dataclass
class HydroDiffGenerationOutput(ModelOutput):
    """
    images: generated images in [-1, 1].
    compute: token evaluations divided by `steps * tokens` (what a dense sampler with the same steps would spend);
        guidance adds one reference evaluation per token evaluation on top.
    noise_masks: optional per-step noise levels of every token, (steps + 1, batch, tokens).
    update_masks: optional per-step binary masks of computed tokens, (steps, batch, tokens).
    """

    images: torch.Tensor | None = None
    compute: float | None = None
    noise_masks: torch.Tensor | None = None
    update_masks: torch.Tensor | None = None


class HydroDiffPreTrainedModel(PreTrainedModel):
    config: HydroDiffConfig
    config_class = HydroDiffConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = ["HydroDiffBlock"]
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        # adaLN-Zero (every block starts as the identity, the output at zero) is set on the linear layers themselves:
        # transformers skips custom-code modules that hold no parameters of their own, such as HydroDiffModel
        if isinstance(module, nn.Linear) and getattr(module, "zero_init", False):
            init.zeros_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, HydroDiffBlock):
            init.zeros_(module.modulation)
            if module.skip_proj is not None:  # starts as "ignore the skip"
                d = module.skip_proj.weight.shape[0]
                init.copy_(module.skip_proj.weight, torch.cat([torch.eye(d), torch.zeros(d, d)], dim=1))
        elif isinstance(module, HydroDiffMoE):
            init.normal_(module.experts_gate_up, std=std)
            init.normal_(module.experts_down, std=std)
            init.zeros_(module.expert_bias)


class HydroDiffModel(HydroDiffPreTrainedModel):
    """The denoiser: noisy patch tokens with per-token noise levels in, clean-patch (or velocity) predictions out."""

    def __init__(self, config: HydroDiffConfig):
        super().__init__(config)
        self.patch_embed = nn.Linear(config.patch_dim, config.hidden_size)
        # Self-conditioning: the previous clean-patch estimate enters through its own (zero-initialised) projection
        self.self_cond_embed = nn.Linear(config.patch_dim, config.hidden_size, bias=False) if config.self_condition else None
        if self.self_cond_embed is not None:
            self.self_cond_embed.zero_init = True
        self.conditioning = HydroDiffConditioning(config)
        self.layers = nn.ModuleList([HydroDiffBlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm = HydroDiffRMSNorm(config.hidden_size, config.rms_norm_eps, affine=False)
        self.proj_out = nn.Linear(config.hidden_size, config.patch_dim)
        self.uncertainty_head = nn.Linear(config.hidden_size, 1)
        self.proj_out.zero_init = self.uncertainty_head.zero_init = True
        self.post_init()

    def forward(
        self,
        noisy_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor | None = None,
        index: torch.Tensor | None = None,
        cache: HydroDiffTokenCache | None = None,
        skip_layers: tuple[int, ...] = (),
        self_cond: torch.Tensor | None = None,
    ) -> HydroDiffModelOutput:
        """
        noisy_tokens: (batch, active, patch_dim) - all tokens, or only the active ones when `index` is given.
        timesteps: (batch, active) per-token noise levels in [0, 1].
        index: (batch, active) token positions of the active tokens. Tokens not listed are not computed; with a
            `cache` they still serve as keys and values from the last step that computed them.
        skip_layers: blocks to leave out (a deliberately weakened model, used as the reference of guidance).
        self_cond: the previous clean-patch estimate of the same tokens (`self_condition=True`; None = no estimate).
        """
        cos, sin = rotary_2d(self.config, noisy_tokens.device)
        if index is not None:
            cos, sin = cos[index], sin[index]
        cos, sin = cos.unsqueeze(-3).to(noisy_tokens.dtype), sin.unsqueeze(-3).to(noisy_tokens.dtype)
        modulation, (final_shift, final_scale) = self.conditioning(timesteps, labels)
        hidden_states = self.patch_embed(noisy_tokens)
        if self.self_cond_embed is not None and self_cond is not None:
            hidden_states = hidden_states + self.self_cond_embed(self_cond)
        skips, features = [], None
        for i, layer in enumerate(self.layers):
            skip = skips.pop() if layer.skip_proj is not None else None
            if i not in skip_layers:
                hidden_states = layer(hidden_states, modulation, cos, sin, timesteps, index, cache, skip)
            if self.config.long_skips and i < len(self.layers) // 2:
                skips.append(hidden_states)
            if i == len(self.layers) // 2 - 1:
                features = hidden_states.mean(1)  # image-level representation for the dispersive loss
        hidden_states = self.norm(hidden_states)
        log_error = self.uncertainty_head(hidden_states.detach()).squeeze(-1)
        prediction = self.proj_out(hidden_states * (1 + final_scale) + final_shift)
        return HydroDiffModelOutput(prediction=prediction, log_error=log_error, features=features)


class HydroDiffForImageGeneration(HydroDiffPreTrainedModel):
    """
    Rectified flow: `x_t = (1 - t) x_0 + t * noise` per token, trained on the velocity `noise - x_0`. `forward` returns
    the training loss; [`generate`] samples with the noise mask and the update mask.
    """

    def __init__(self, config: HydroDiffConfig):
        super().__init__(config)
        self.model = HydroDiffModel(config)
        self.post_init()

    # ---------------------------------------------------------------------------------------------------- training

    def sample_timesteps(self, batch: int, device, generator=None) -> torch.Tensor:
        """
        Per-token noise levels for training. Each image gets a logit-normal progress `q` (how far denoising has got;
        `timestep_mean` > 0 moves it towards high noise).
        A `hetero_prob` share of images also gets a per-token speed `r` in [1, max_rate] from progress `hetero_start`
        on, so a token's progress is `q + (r - 1) * (q - hetero_start)`: the states the adaptive sampler produces, where
        confident tokens are ahead of (cleaner than) the rest once its warmup is over.
        """
        cfg, n = self.config, self.config.num_patches
        q = 1 - torch.sigmoid(torch.randn(batch, 1, device=device, generator=generator) + cfg.timestep_mean)
        rate = 1 + (cfg.max_rate - 1) * torch.rand(batch, n, device=device, generator=generator)
        hetero = torch.rand(batch, 1, device=device, generator=generator) < cfg.hetero_prob
        rate = torch.where(hetero, rate, torch.ones_like(rate))
        progress = q + (rate - 1) * (q - cfg.hetero_start).clamp_min(0)
        return shift_time((1 - progress).clamp(0, 1), cfg.time_shift)

    def velocity(self, prediction: torch.Tensor, noisy_tokens: torch.Tensor, timesteps: torch.Tensor, min_t: float = 0.0):
        if self.config.prediction_type == "v":
            return prediction
        return (noisy_tokens - prediction) / timesteps.clamp_min(max(min_t, 1e-6))[..., None]

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None, timesteps=None, noise=None):
        """pixel_values: (batch, channels, height, width) in [-1, 1]."""
        cfg = self.config
        x0 = patchify(pixel_values, cfg.patch_size)
        b = x0.shape[0]
        if timesteps is None:
            timesteps = self.sample_timesteps(b, x0.device)
        noise = torch.randn_like(x0) if noise is None else noise
        if cfg.num_classes:
            labels = torch.full((b,), cfg.num_classes, device=x0.device) if labels is None else labels
            if self.training and cfg.class_dropout:
                drop = torch.rand(b, device=x0.device) < cfg.class_dropout
                labels = torch.where(drop, torch.full_like(labels, cfg.num_classes), labels)
        t = timesteps[..., None]
        x_t = (1 - t) * x0 + t * noise
        self_cond = None
        if cfg.self_condition and self.training and torch.rand(()) < 0.5:
            with torch.no_grad():  # half the time, condition on the model's own first estimate (Analog Bits)
                first = self.model(x_t, timesteps, labels).prediction
                self_cond = first if cfg.prediction_type == "x" else x_t - t * first
        out = self.model(x_t, timesteps, labels, self_cond=self_cond)
        if cfg.prediction_type == "x":
            # prediction and target share the clipped 1 / t, so a perfect prediction has zero loss at every t
            v_pred = self.velocity(out.prediction, x_t, timesteps, X_PRED_MIN_T)
            v_target = (x_t - x0) / t.clamp_min(X_PRED_MIN_T)
            x0_pred = out.prediction
        else:
            v_pred, v_target = out.prediction, noise - x0
            x0_pred = x_t - t * v_pred
        per_token = (v_pred - v_target).pow(2).mean(-1)
        weight = (timesteps > 0).to(per_token.dtype)  # tokens already at t = 0 are clean context, not targets
        denoise_loss = (per_token * weight).sum() / weight.sum().clamp_min(1)
        # The noise mask head learns log E|x0_hat - x0|^2 per token (exponential likelihood; minimised at the log-mean)
        x0_error = (x0_pred - x0).pow(2).mean(-1).detach()
        u = out.log_error
        uncertainty_loss = (((x0_error * torch.exp(-u) + u) * weight).sum() / weight.sum().clamp_min(1))
        loss = denoise_loss + cfg.uncertainty_loss_weight * uncertainty_loss
        if cfg.contrastive_weight and self.training:
            # Contrastive flow matching: move away from the target of another image in the batch
            other = torch.roll(torch.arange(b, device=x0.device), 1)
            if cfg.prediction_type == "x":
                v_other = (x_t - x0[other]) / t.clamp_min(X_PRED_MIN_T)
            else:
                v_other = noise[other] - x0[other]
            contrast = ((v_pred - v_other).pow(2).mean(-1) * weight).sum() / weight.sum().clamp_min(1)
            loss = loss - cfg.contrastive_weight * contrast
        if cfg.dispersive_weight and self.training:
            # Dispersive loss (InfoNCE with squared distances, no positives): spread the image representations apart.
            # On the unit sphere: the residual stream has no fixed scale, and unnormalised features let the model drive
            # the loss to minus infinity by growing them (training diverged that way)
            z = F.normalize(out.features.float(), dim=-1)
            dist = torch.cdist(z, z).pow(2)
            off = ~torch.eye(b, dtype=torch.bool, device=z.device)
            dispersive = torch.logsumexp(-dist[off] / DISPERSIVE_TEMPERATURE, 0) - math.log(off.sum().item())
            loss = loss + cfg.dispersive_weight * dispersive
        return HydroDiffOutput(
            loss=loss, denoise_loss=denoise_loss, uncertainty_loss=uncertainty_loss, prediction=out.prediction,
            timesteps=timesteps,
        )

    # ---------------------------------------------------------------------------------------------------- sampling

    def _velocity_and_error(self, x, t, labels, index, caches, guidance, self_cond=None, first_step=False):
        """
        The velocity of the active tokens, with guidance: `v_ref + scale * (v - v_ref)`. The reference is the
        unconditional model (classifier-free guidance), optionally with some blocks left out, or a weaker model with the
        same label (autoguidance). Outside `interval` (per-token noise levels) the plain conditional velocity is used.
        """
        scale, interval, skip_layers, guide_model, schedule, cfg_zero = guidance
        if scale == 1 or (labels is None and guide_model is None):
            out = self.model(x, t, labels, index, caches[0], self_cond=self_cond)
            return self.velocity(out.prediction, x, t), out.log_error
        if guide_model is None and not skip_layers:  # plain classifier-free guidance: one batched pass
            null = torch.full_like(labels, self.config.num_classes)
            both = lambda z: None if z is None else torch.cat([z, z])
            out = self.model(both(x), both(t), torch.cat([labels, null]), both(index), caches[0], self_cond=both(self_cond))
            v, v_ref = self.velocity(out.prediction, both(x), both(t)).chunk(2)
            log_error = out.log_error.chunk(2)[0]
        else:
            out = self.model(x, t, labels, index, caches[0], self_cond=self_cond)
            v, log_error = self.velocity(out.prediction, x, t), out.log_error
            if guide_model is not None:
                guide_cond = self_cond if guide_model.config.self_condition else None
                ref = guide_model.model(x, t, labels, index, caches[1], self_cond=guide_cond)
                v_ref = guide_model.velocity(ref.prediction, x, t)
            else:
                null = torch.full_like(labels, self.config.num_classes)
                ref = self.model(x, t, null, index, caches[1], skip_layers=skip_layers, self_cond=self_cond)
                v_ref = self.velocity(ref.prediction, x, t)
        if cfg_zero:  # CFG-Zero*: scale the reference by its projection on the conditional velocity, per image
            dims = tuple(range(1, v.ndim))
            proj = (v * v_ref).sum(dims, keepdim=True) / v_ref.pow(2).sum(dims, keepdim=True).clamp_min(1e-8)
            v_ref = proj * v_ref
            if first_step:
                return torch.zeros_like(v), log_error
        weight = scale if schedule == "constant" else 1 + (scale - 1) * 2 * (1 - t[..., None])  # "linear": rises as t falls
        guided = v_ref + weight * (v - v_ref)
        if interval is not None:
            inside = ((t >= interval[0]) & (t <= interval[1]))[..., None]
            guided = torch.where(inside, guided, v)
        return guided, log_error

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor | None = None,
        batch_size: int | None = None,
        steps: int = 16,
        guidance_scale: float | None = None,
        adaptive: bool = True,
        max_rate: float | None = None,
        warmup: float = 0.25,
        min_token_steps: int | None = None,
        solver: str = "dpm2m",
        churn: float = 0.0,
        guidance_interval: tuple[float, float] | None = None,
        uncond_skip_layers: tuple[int, ...] = (),
        guide_model: "HydroDiffForImageGeneration | None" = None,
        guidance_schedule: str = "constant",
        cfg_zero: bool = False,
        time_shift: float | None = None,
        reuse: float = 0.0,
        reuse_by: str = "uncertainty",
        skip_finished: bool = True,
        refresh_finished: bool = True,
        generator: torch.Generator | None = None,
        return_masks: bool = False,
    ) -> HydroDiffGenerationOutput:
        """
        Samples images in `steps` steps.

        Every step first predicts the noise mask - each token's next noise level - and from it the update mask - the
        tokens that need computing. A token's speed is set by how uncertain the model is about it: the token the model
        is surest of moves `max_rate` times faster than the schedule, the least sure moves with it, and the rest are
        spread between by rank (geometrically). Tokens that reach `t = 0` leave the update mask, so later steps run on
        fewer tokens, and the hardest tokens are finished last against an already clean context. With
        `adaptive=False` every token follows the same schedule.

        `min_token_steps` caps the speed-up so that no token takes fewer steps than that: a fast token is denoised on a
        coarser time grid, and below the step count at which the dense sampler stops improving that costs quality. The
        default is 16 for `euler` (16 steps or fewer: no speed-up; 20 steps: 1.36x at most; 32 steps: not binding)
        and 10 for `dpm2m`, whose second-order steps tolerate a coarser grid (16 steps: the full 2x).
        `reuse` obeys a floor too: a token only skips a step while it can still be run 16 times (or `min_token_steps`
        times when that is given) - a reused token's next step is first order, whatever the solver.

        `solver="dpm2m"` (default) is DPM-Solver++(2M) on the clean-patch estimates, run per token so that every token
        uses its own step sizes; `"euler"` is the first-order step, which did better on briefly trained models.
        `churn` > 0 replaces that share of each token's noise with fresh noise every step (`euler` only), scaled per
        token by the same uncertainty rank: the tokens the model cannot predict get the most fresh noise.

        Guidance (`guidance_scale` > 1) mixes in a reference velocity: the unconditional model (classifier-free
        guidance), the unconditional model without the blocks in `uncond_skip_layers` (as MOSS-Image does), or a
        weaker `guide_model` given the same labels (autoguidance). `guidance_interval=(lo, hi)` applies it only to
        tokens whose noise level is in that range; both default to the configuration (scale 3 on `[0.2, 0.85]`; pass
        `guidance_scale=1` for unguided sampling at half the cost); `guidance_schedule="linear"` raises the scale from 1 at pure noise to
        `2 * guidance_scale - 1` at the clean end (same mean); `cfg_zero=True` applies CFG-Zero* (the reference is
        scaled by its projection on the conditional velocity, and the first step is skipped). `time_shift` overrides
        the configuration's schedule shift.

        `reuse` > 0 widens the update mask: that share of the moving tokens - the surest ones - skips the network in a
        step and moves with its estimate from the step before (never twice in a row, never on its last step);
        `reuse_by="random"` picks those tokens at random instead (a control for the predicted mask).
        `refresh_finished` runs a token once more at `t = 0` in the step after it finishes, so that the cache holds it as
        clean context - what the model saw in training - rather than its last noisy state. `skip_finished=False` computes
        every token in every step (no speed-up; for measuring what the key/value cache costs).
        """
        cfg = self.config
        device = self.device
        if not 0 <= churn <= 1:
            raise ValueError("churn must be in [0, 1]")
        if churn and solver != "euler":
            raise ValueError("churn needs solver='euler'")
        if labels is None and cfg.num_classes:
            labels = torch.full((batch_size,), cfg.num_classes, device=device)
        b = labels.shape[0] if labels is not None else batch_size
        n = cfg.num_patches
        max_rate = cfg.max_rate if max_rate is None else max_rate
        # A reused token's next step is first order (its history has a gap), so reuse keeps the first-order floor
        reuse_floor = 16 if min_token_steps is None else min_token_steps
        if min_token_steps is None:
            min_token_steps = 16 if solver == "euler" else 10
        # A token at speed r takes k0 + (steps - k0) / r steps, k0 being the warmup steps at speed 1
        k0 = max(1, math.ceil(warmup * steps))
        if min_token_steps > k0:
            max_rate = max(1.0, min(max_rate, (steps - k0) / (min_token_steps - k0)))
        x = torch.randn(b, n, cfg.patch_dim, device=device, generator=generator, dtype=self.dtype)
        progress = torch.zeros(b, n, device=device)
        log_error = torch.zeros(b, n, device=device)
        x0 = torch.zeros_like(x)  # each token's latest clean-patch estimate
        prev_x0, prev_h = torch.zeros_like(x), torch.full((b, n), float("nan"), device=device)
        computed = torch.ones(b, n, dtype=torch.bool, device=device)
        evaluations = torch.zeros(b, n, device=device)  # steps in which each token was run while it still had noise
        just_finished = torch.zeros(b, n, dtype=torch.bool, device=device)
        caches = (HydroDiffTokenCache(), HydroDiffTokenCache())
        if guidance_scale is None:
            guidance_scale = cfg.guidance_scale
        if guidance_interval is None:
            guidance_interval = (cfg.guidance_min_t, cfg.guidance_max_t)
        guidance = (guidance_scale, guidance_interval, tuple(uncond_skip_layers), guide_model, guidance_schedule, cfg_zero)
        shift = cfg.time_shift if time_shift is None else time_shift
        to_t = lambda q: shift_time(1 - q, shift)
        evaluated, noise_masks, update_masks = 0, [to_t(progress)], []
        for k in range(steps):
            moving = progress < 1
            # Uncertainty rank among the moving tokens, from the last step that computed each: 0 = surest, 1 = least sure
            if k > 0:
                order = torch.where(moving, log_error, float("-inf")).argsort(-1).argsort(-1).float()
                finished, still = (~moving).sum(-1, keepdim=True), moving.sum(-1, keepdim=True)
                rank = ((order - finished) / (still - 1).clamp_min(1)).clamp(0, 1)
            else:
                rank = torch.full_like(progress, 0.5)
            shaping = k > 0 and k >= warmup * steps
            rate = max_rate ** (1 - rank) if adaptive and shaping else torch.ones_like(progress)
            target = (progress + rate / steps).clamp(max=1) if k < steps - 1 else torch.ones_like(progress)
            need = moving | (just_finished if refresh_finished else torch.zeros_like(moving))
            if reuse and shaping and k < steps - 1:
                # the surest moving tokens that were computed last step reuse that estimate instead (never twice in a
                # row, never for the step that finishes them, and only while the token can still be run `reuse_floor`
                # times if every later step computes it)
                top = max_rate if adaptive else 1.0
                fewest = torch.ceil((1 - progress) * steps / top - 1e-6).clamp(max=steps - k)  # steps left at top speed
                candidate = moving & computed & (target < 1) & (evaluations + fewest - 1 >= reuse_floor)
                skips = (reuse * moving.sum(-1, keepdim=True)).floor()
                score = rank if reuse_by == "uncertainty" else torch.rand(b, n, device=device, generator=generator)
                order = torch.where(candidate, score, score + 2).argsort(-1).argsort(-1)
                need &= ~(candidate & (order < skips))
            count = int(need.sum(-1).max())
            if k == 0 or count == n or not skip_finished:
                index = None
            else:  # the same number of tokens per image; images with fewer active tokens pad with finished ones
                index = (need.float() + torch.rand(b, n, device=device, generator=generator) * 0.5).topk(count, -1).indices
                index = index.sort(-1).values
            x_a, t_a = gather_tokens(x, index), to_t(gather_tokens(progress, index))
            has_estimate = k > (1 if cfg_zero and guidance_scale != 1 else 0)  # the CFG-Zero* first step estimates nothing
            self_cond = gather_tokens(x0, index) if cfg.self_condition and has_estimate else None
            v, err = self._velocity_and_error(
                x_a, t_a.to(x.dtype), labels, index, caches, guidance, self_cond, first_step=k == 0
            )
            x0 = scatter_tokens(x0, index, x_a - t_a[..., None].to(x.dtype) * v)
            log_error = scatter_tokens(log_error, index, err.float())
            computed = scatter_tokens(torch.zeros_like(need), index, torch.ones_like(t_a, dtype=torch.bool))
            evaluations += (computed & moving).float()
            evaluated += b * (n if index is None else count)

            # every moving token steps from t to t_next with its latest estimate; finished tokens stay as they are
            t, t_next = to_t(progress), to_t(target)
            t3, tn3 = t[..., None].to(x.dtype), t_next[..., None].to(x.dtype)
            ratio = tn3 / t3.clamp_min(1e-12)
            if solver == "dpm2m":
                h = log_snr(t_next) - log_snr(t)
                h = torch.where(t >= 1, torch.full_like(h, float("nan")), h)  # a step from pure noise has h = inf
                r = prev_h / h
                second = (computed & (r > 0) & torch.isfinite(r) & (t_next > 0))[..., None]
                d = torch.where(second, x0 + (x0 - prev_x0) / (2 * r[..., None].clamp_min(1e-6)), x0)
                x_new = ratio * x + ((1 - tn3) - (1 - t3) * ratio) * d
                prev_x0 = torch.where(computed[..., None], x0, prev_x0)
                prev_h = torch.where(computed, h, torch.full_like(h, float("nan")))
            else:
                noise = (x - (1 - t3) * x0) / t3.clamp_min(1e-12)  # the noise this estimate implies
                if churn:
                    eta = (churn * rank)[..., None].to(x.dtype)
                    fresh = torch.randn(noise.shape, device=device, generator=generator, dtype=x.dtype)
                    noise = (1 - eta**2).sqrt() * noise + eta * fresh
                x_new = (1 - tn3) * x0 + tn3 * noise
            x = torch.where(moving[..., None], x_new, x)
            progress = torch.where(moving, target, progress)
            just_finished = moving & (progress >= 1)
            if return_masks:
                update_masks.append(computed & moving)
                noise_masks.append(to_t(progress))
        images = unpatchify(x, cfg.patch_size, cfg.num_channels).clamp(-1, 1)
        return HydroDiffGenerationOutput(
            images=images,
            compute=evaluated / (b * n * steps),
            noise_masks=torch.stack(noise_masks) if return_masks else None,
            update_masks=torch.stack(update_masks) if return_masks else None,
        )


def register_hydrodiff_auto_classes() -> None:
    """Registers HydroDiff with `AutoConfig` and `AutoModel`, so checkpoints with `model_type="hydrodiff"` load."""
    from transformers import AutoConfig, AutoModel

    try:
        AutoConfig.register("hydrodiff", HydroDiffConfig)
    except ValueError:
        return  # already registered
    AutoModel.register(HydroDiffConfig, HydroDiffModel)


__all__ = [
    "HydroDiffPreTrainedModel",
    "HydroDiffModel",
    "HydroDiffForImageGeneration",
    "HydroDiffTokenCache",
    "HydroDiffModelOutput",
    "HydroDiffOutput",
    "HydroDiffGenerationOutput",
    "patchify",
    "unpatchify",
    "register_hydrodiff_auto_classes",
]
