from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig


# Pixel-space diffusion transformers over patch tokens. The `-moe` presets keep the active parameters of the dense
# preset they are named after and add routed experts (see test/RESEARCH.md for what each part bought on the bench).
HYDRODIFF_PRESETS = {
    "hydrodiff-s": dict(hidden_size=192, intermediate_size=512, num_hidden_layers=6, num_attention_heads=3),
    "hydrodiff-b": dict(hidden_size=256, intermediate_size=704, num_hidden_layers=8, num_attention_heads=4),
    "hydrodiff-l": dict(hidden_size=512, intermediate_size=1408, num_hidden_layers=12, num_attention_heads=8),
    "hydrodiff-s-moe": dict(
        hidden_size=192, intermediate_size=512, num_hidden_layers=6, num_attention_heads=3,
        num_experts=8, num_experts_per_tok=1, moe_intermediate_size=256, first_dense_layers=1,
    ),
    "hydrodiff-b-moe": dict(
        hidden_size=256, intermediate_size=704, num_hidden_layers=8, num_attention_heads=4,
        num_experts=8, num_experts_per_tok=1, moe_intermediate_size=352, first_dense_layers=1,
    ),
}


@strict
class HydroDiffConfig(PreTrainedConfig):
    r"""
    Configuration for HydroDiff models. The defaults describe `hydrodiff-b` on 32x32 RGB images; use
    [`HydroDiffConfig.from_preset`] for other sizes.

    HydroDiff is a rectified-flow diffusion transformer that works directly on pixel patches, with three additions:

    - Per-token noise levels (the *noise mask*). Every patch token carries its own noise level `t` in [0, 1], and the
      model is trained on images whose tokens sit at different levels. At sampling time a predicted, non-binary mask
      decides how fast each token is denoised: tokens the model is sure about move ahead, tokens it is unsure about
      (fire, grain, foliage - texture it cannot predict) keep their noise longer and get every step, so they are
      generated last, against an already clean context.
    - Active tokens (the *update mask*). A binary mask selects the tokens computed in a step. The model gathers only
      those tokens; the others take part as attention keys and values from a cache and are never run through the
      layers. Tokens that have reached `t = 0` drop out for good, so an image costs fewer token-steps than
      `steps * tokens`.
    - Routed experts. MLPs can be fine-grained mixtures of experts with one always-on shared expert, sigmoid top-k
      routing and an auxiliary-loss-free balancing bias. The router sees the token's own noise level through the
      modulation, so experts can specialise by noise level and by content. At the bench scale they did not beat dense
      MLPs at equal active compute, so they are off by default.

    Args:
        image_size (`int`), num_channels (`int`), patch_size (`int`): Image and patch shape.
        num_classes (`int`): Class labels for conditioning (0 = unconditional). One extra "null" class is added for
            classifier-free guidance; `class_dropout` is the rate at which training uses it.
        long_skips (`bool`): U-ViT long skip connections from each block in the first half to its mirror block.
        prediction_type (`str`): `"x"` (the network outputs the clean patch, the loss is on velocity) or `"v"`
            (the network outputs the velocity). Pixel patches are high-dimensional, which favours `"x"`.
        time_shift (`float`): Resolution shift of the noise schedule, `t' = s t / (1 + (s - 1) t)`.
        timestep_mean (`float`): Mean of the logit-normal training noise level (0: centred on t = 0.5; positive
            values train more at high noise, as MOSS-Image does).
        hetero_prob (`float`): Fraction of training images whose tokens get different noise levels.
        hetero_start (`float`): Progress (0 = pure noise, 1 = clean) from which those tokens' speeds differ. Matching
            the sampler's `warmup` keeps training on the states the sampler produces and nothing else.
        max_rate (`float`): Largest per-token speed-up, both in training (the spread of noise levels) and at sampling.
        uncertainty_loss_weight (`float`): Weight of the head that predicts each token's denoising error; its output
            is the noise mask.
        self_condition (`bool`): Feed the model its previous clean-patch estimate (half of the training steps; every
            step after the first at sampling).
        contrastive_weight (`float`): Contrastive flow matching - push the prediction away from another image's target.
        dispersive_weight (`float`): Dispersive loss on the mid-network image representation.
        guidance_scale (`float`), guidance_min_t (`float`), guidance_max_t (`float`): Default classifier-free guidance
            of [`~HydroDiffForImageGeneration.generate`]: the scale, applied only to tokens whose noise level is in
            `[guidance_min_t, guidance_max_t]` (guidance at the ends of the trajectory cost quality on the bench).
        num_experts (`int`): Routed experts per MoE layer (0 = dense MLPs).
        num_experts_per_tok (`int`), moe_intermediate_size (`int`), first_dense_layers (`int`),
        router_bias_update_rate (`float`): MoE shape and balancing-bias step.
        moe_routing (`str`): `"token"` (each token picks its experts), `"image"` (the whole image picks, per layer and
            step) or `"time"` (no router: each token goes to the expert of its own noise level).

    Example:

    ```python
    >>> from hydrodiff import HydroDiffConfig, HydroDiffForImageGeneration
    >>> model = HydroDiffForImageGeneration(HydroDiffConfig.from_preset("hydrodiff-s"))
    >>> images = model.generate(labels=torch.arange(10)).images
    ```"""

    model_type = "hydrodiff"

    image_size: int = 32
    num_channels: int = 3
    patch_size: int = 4
    hidden_size: int = 256
    intermediate_size: int = 704
    num_hidden_layers: int = 8
    num_attention_heads: int = 4
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float | int = 100.0
    initializer_range: float = 0.02
    num_classes: int = 10
    class_dropout: float | int = 0.1
    long_skips: bool = False

    prediction_type: str = "x"
    time_shift: float | int = 1.0
    timestep_mean: float | int = 0.0
    hetero_prob: float | int = 0.25
    hetero_start: float | int = 0.25
    max_rate: float | int = 2.0
    uncertainty_loss_weight: float | int = 0.1
    self_condition: bool = False
    contrastive_weight: float | int = 0.0
    dispersive_weight: float | int = 0.0
    guidance_scale: float | int = 3.0
    guidance_min_t: float | int = 0.2
    guidance_max_t: float | int = 0.85

    num_experts: int = 0
    num_experts_per_tok: int = 1
    moe_intermediate_size: int = 352
    first_dense_layers: int = 1
    router_bias_update_rate: float | int = 1e-3
    moe_routing: str = "token"

    def __post_init__(self, **kwargs):
        if self.image_size % self.patch_size:
            raise ValueError(f"image_size {self.image_size} is not divisible by patch_size {self.patch_size}")
        if self.hidden_size % self.num_attention_heads or (self.hidden_size // self.num_attention_heads) % 4:
            raise ValueError("head_dim must be a multiple of 4 (two rotary axes)")
        if self.prediction_type not in ("x", "v"):
            raise ValueError("prediction_type must be 'x' or 'v'")
        if self.num_experts and not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if self.moe_routing not in ("token", "image", "time"):
            raise ValueError("moe_routing must be 'token', 'image' or 'time'")
        if not 0 <= self.hetero_start < 1:
            raise ValueError("hetero_start must be in [0, 1)")
        if self.max_rate < 1:
            raise ValueError("max_rate must be >= 1")
        super().__post_init__(**kwargs)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patches_per_side**2

    @property
    def patch_dim(self) -> int:
        return self.num_channels * self.patch_size**2

    def is_moe_layer(self, layer_idx: int) -> bool:
        return bool(self.num_experts) and layer_idx >= self.first_dense_layers

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "HydroDiffConfig":
        if name not in HYDRODIFF_PRESETS:
            raise ValueError(f"Unknown preset {name!r}; choose from {sorted(HYDRODIFF_PRESETS)}")
        return cls(**{**HYDRODIFF_PRESETS[name], **overrides})


__all__ = ["HydroDiffConfig", "HYDRODIFF_PRESETS"]
