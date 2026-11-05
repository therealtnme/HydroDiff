"""
A text/image -> image/video pipeline with the interface of other generative pipelines:

    pipe = HydroDiffPipeline.from_pretrained("runs/coco/pipeline")
    image = pipe("a red bus on a city street").images[0]                         # text -> image
    image = pipe("the same street at night", image=photo, strength=0.6).images[0] # text + image -> image
    video = pipe("waves on a beach", num_frames=16).frames[0]                     # text -> video (video models)
    video = pipe("the dog runs", image=photo, num_frames=16).frames[0]           # text + image -> video

Inside it is HydroDiff: per-token noise levels set by the predicted noise mask, an update mask that removes finished
(and optionally reused) tokens from the computation, self-conditioning and interval classifier-free guidance. What the
pipeline can do follows from `config.modalities` (what the model was built and trained for): an image-only model refuses
`num_frames > 1`, a video-only model refuses single images, a model with both does both.

Text comes from a frozen CLIP text encoder; images are generated in pixel space unless the checkpoint ships a `vae/`
(then `config.vae_scale_factor` > 1 and the model works on its latents).
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .modeling_hydrodiff import HydroDiffForImageGeneration


@dataclass
class HydroDiffPipelineOutput:
    images: list | None = None  # PIL images (or a tensor with output_type="pt"), one per generated image
    frames: list | None = None  # per video, a list of PIL frames (or a (frames, 3, h, w) tensor)
    compute: float | None = None  # sampler token evaluations relative to a dense sampler with the same steps


def round_to(value: int, multiple: int) -> int:
    return max(multiple, int(value / multiple + 0.5) * multiple)  # half up (round() rounds halves to even)


def pil_to_tensor(image: Image.Image, multiple: int, size: tuple[int, int] | None = None) -> torch.Tensor:
    """RGB in [-1, 1], (3, h, w). The whole image is kept (no crop): each side is scaled to the nearest multiple."""
    image = image.convert("RGB")
    w, h = image.size if size is None else size
    target = (round_to(w, multiple), round_to(h, multiple))
    if target != image.size:
        image = image.resize(target, Image.BICUBIC)
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1)


def tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    array = ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in array]


def export_to_gif(frames: list[Image.Image], path: str, fps: int = 8) -> str:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return path


class HydroDiffPipeline:
    def __init__(self, transformer: HydroDiffForImageGeneration, text_encoder=None, tokenizer=None, vae=None):
        self.transformer = transformer.eval()
        self.text_encoder = None if text_encoder is None else text_encoder.eval()
        self.tokenizer = tokenizer
        self.vae = None if vae is None else vae.eval()
        cfg = transformer.config
        if cfg.text_dim and (text_encoder is None or tokenizer is None):
            raise ValueError("this model is text-conditioned; pass its text_encoder and tokenizer")
        if cfg.vae_scale_factor > 1 and vae is None:
            raise ValueError("this model works on latents; pass its vae")

    # ------------------------------------------------------------------------------------------------ loading/saving

    @classmethod
    def from_pretrained(cls, path: str, device: str | torch.device | None = None, dtype: torch.dtype = torch.float32):
        """Loads `transformer/` and, when present, `text_encoder/` + `tokenizer/` and `vae/`. Weights stored in half
        precision are upcast to `dtype` (float32 is fastest on CPUs without native bf16/fp16 arithmetic)."""
        transformer = HydroDiffForImageGeneration.from_pretrained(os.path.join(path, "transformer"), dtype=dtype)
        text_encoder = tokenizer = vae = None
        if os.path.isdir(os.path.join(path, "text_encoder")):
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer

            text_encoder = CLIPTextModelWithProjection.from_pretrained(os.path.join(path, "text_encoder"), dtype=dtype)
            tokenizer = CLIPTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        if os.path.isdir(os.path.join(path, "vae")):
            from diffusers import AutoencoderTiny

            vae = AutoencoderTiny.from_pretrained(os.path.join(path, "vae"), torch_dtype=dtype)
        pipe = cls(transformer, text_encoder, tokenizer, vae)
        return pipe.to(device) if device is not None else pipe

    def save_pretrained(self, path: str, half: bool = True):
        """Saves every part; `half=True` stores the weights in float16 (half the disk space, upcast when loading)."""
        def save(module, sub):
            target = os.path.join(path, sub)
            if half:
                original = next(module.parameters()).dtype
                module.to(torch.float16).save_pretrained(target)
                module.to(original)
            else:
                module.save_pretrained(target)

        save(self.transformer, "transformer")
        if self.text_encoder is not None:
            save(self.text_encoder, "text_encoder")
            self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        if self.vae is not None:
            save(self.vae, "vae")

    def to(self, device):
        for part in (self.transformer, self.text_encoder, self.vae):
            if part is not None:
                part.to(device)
        return self

    @property
    def device(self):
        return self.transformer.device

    @property
    def config(self):
        return self.transformer.config

    @property
    def multiple(self) -> int:
        """Pixel sizes must be multiples of this (one patch token)."""
        return self.config.patch_size * self.config.vae_scale_factor

    # ------------------------------------------------------------------------------------------------ encoders

    @torch.inference_mode()
    def encode_prompt(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, text_max_length, text_dim) token embeddings and (batch, text_dim) pooled embeddings."""
        tokens = self.tokenizer(
            prompts, padding="max_length", max_length=self.config.text_max_length, truncation=True, return_tensors="pt"
        ).to(self.text_encoder.device)
        out = self.text_encoder(**tokens)
        dtype = self.transformer.dtype
        return out.last_hidden_state.to(dtype), out.text_embeds.to(dtype)

    @torch.inference_mode()
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """[-1, 1] pixels (batch, 3, h, w) -> the model's input space."""
        cfg = self.config
        if self.vae is None:
            return pixels.to(self.device, self.transformer.dtype)
        latents = self.vae.encode(pixels.to(self.device, self.vae.dtype)).latents
        return ((latents - cfg.latent_shift) * cfg.latent_scale).to(self.transformer.dtype)

    @torch.inference_mode()
    def decode(self, samples: torch.Tensor) -> torch.Tensor:
        """The model's output space -> [-1, 1] pixels; videos (b, c, f, h, w) are decoded frame by frame."""
        cfg = self.config
        if self.vae is None:
            return samples.clamp(-1, 1)
        video = samples.ndim == 5
        if video:
            b, c, f, h, w = samples.shape
            samples = samples.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        pixels = self.vae.decode((samples / cfg.latent_scale + cfg.latent_shift).to(self.vae.dtype)).sample.clamp(-1, 1)
        if video:
            pixels = pixels.reshape(b, f, *pixels.shape[1:]).permute(0, 2, 1, 3, 4)
        return pixels

    # ------------------------------------------------------------------------------------------------ generation

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        image: Image.Image | list[Image.Image] | None = None,
        negative_prompt: str | list[str] | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 16,
        guidance_scale: float | None = None,
        strength: float = 0.6,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | None = None,
        output_type: str = "pil",
        **sampler_kwargs,
    ) -> HydroDiffPipelineOutput:
        """
        prompt: text (or a list); None generates unconditionally.
        image: with `num_frames` 1, image-to-image from a noised copy (`strength`: share of the trajectory re-run);
            with `num_frames` > 1, the first frame of the video (image-to-video).
        num_frames: 1 = image (default for models that make images), > 1 = video (default 16 for video-only models).
        height, width: output size in pixels (rounded to the nearest multiple of `self.multiple`; default: the
            image's size, else the configured size).
        guidance_scale: classifier-free guidance (default from the configuration; 1 = off, half the cost).
        sampler_kwargs: any option of `HydroDiffForImageGeneration.generate` (`adaptive`, `reuse`, `solver`,
            `min_token_steps`, `max_rate`, `guidance_interval`, ...).
        """
        cfg = self.config
        if num_frames is None:
            num_frames = 1 if cfg.supports_image else 16
        prompts = [prompt] if isinstance(prompt, str) or prompt is None else list(prompt)
        if image is not None and not isinstance(image, (list, tuple)):
            image = [image]
        if image is not None and len(prompts) == 1 and len(image) > 1:
            prompts = prompts * len(image)
        prompts = [p for p in prompts for _ in range(num_images_per_prompt)]
        b = len(prompts)
        if image is not None:
            image = [im for im in image for _ in range(num_images_per_prompt)]
            if len(image) != b:
                raise ValueError("pass one image, or one image per prompt")

        multiple, scale = self.multiple, cfg.vae_scale_factor
        if image is not None:
            size = (width, height) if width and height else image[0].size
            pixels = torch.stack([pil_to_tensor(im, multiple, size) for im in image])
            height, width = pixels.shape[-2:]
        else:
            height = round_to(height, multiple) if height else cfg.image_size * scale
            width = round_to(width, multiple) if width else cfg.image_size * scale

        kwargs = dict(
            batch_size=b, steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator,
            height=height // scale, width=width // scale, num_frames=num_frames, **sampler_kwargs,
        )
        if cfg.text_dim and prompts[0] is not None:
            kwargs["text"], kwargs["text_pool"] = self.encode_prompt(prompts)
            if negative_prompt is not None:
                negatives = [negative_prompt] * b if isinstance(negative_prompt, str) else list(negative_prompt)
                kwargs["negative_text"], kwargs["negative_text_pool"] = self.encode_prompt(negatives)
        elif prompts[0] is not None:
            raise ValueError("this model has no text conditioning; call it with prompt=None")
        if image is not None:
            given = self.encode_pixels(pixels)
            if num_frames == 1:
                kwargs.update(init=given, strength=strength)
            else:
                kwargs["cond_frames"] = given.unsqueeze(2)
        out = self.transformer.generate(**kwargs)
        pixels = self.decode(out.images)
        if pixels.ndim == 5:  # (b, c, f, h, w)
            frames = [p.permute(1, 0, 2, 3) for p in pixels]
            if output_type == "pil":
                frames = [tensor_to_pil(f) for f in frames]
            return HydroDiffPipelineOutput(frames=frames, compute=out.compute)
        return HydroDiffPipelineOutput(images=tensor_to_pil(pixels) if output_type == "pil" else pixels, compute=out.compute)


__all__ = ["HydroDiffPipeline", "HydroDiffPipelineOutput", "export_to_gif", "pil_to_tensor", "tensor_to_pil"]
