"""
Tests for text conditioning, any-size images, video, image conditioning and the pipeline. Tiny configs, one thread:

    python -m pytest -q test/test_multimodal.py
"""

import pytest
import torch


torch.set_num_threads(1)

from hydrodiff import HydroDiffConfig, HydroDiffForImageGeneration  # noqa: E402
from hydrodiff.modeling_hydrodiff import HydroDiffTokenCache, gather_tokens, patchify, rotary_nd, unpatchify  # noqa: E402
from hydrodiff.pipeline_hydrodiff import HydroDiffPipeline  # noqa: E402


TEXT_DIM, TEXT_LEN = 16, 5
TINY = dict(
    hidden_size=64, intermediate_size=128, num_attention_heads=2, num_hidden_layers=2, image_size=16, patch_size=4,
    num_classes=0, text_dim=TEXT_DIM, text_max_length=TEXT_LEN, guidance_scale=1,
)


def make(seed=0, **overrides):
    torch.manual_seed(seed)
    model = HydroDiffForImageGeneration(HydroDiffConfig.from_preset("hydrodiff-s", **{**TINY, **overrides}))
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return model.eval()


def prompts(b, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, TEXT_LEN, TEXT_DIM, generator=g), torch.randn(b, TEXT_DIM, generator=g)


def test_patchify_round_trip_for_images_and_videos():
    image, video = torch.randn(2, 3, 8, 12), torch.randn(2, 3, 3, 8, 12)
    assert torch.equal(unpatchify(patchify(image, 4), 4, 3, (1, 2, 3)), image)
    assert torch.equal(unpatchify(patchify(video, 4), 4, 3, (3, 2, 3)), video)
    # frame-major: the first 6 tokens are the first frame
    assert torch.equal(patchify(video, 4)[:, :6], patchify(video[:, :, 0], 4))


def test_image_rotary_is_unchanged_and_video_adds_a_time_axis():
    image_cfg = HydroDiffConfig.from_preset("hydrodiff-s", **TINY)
    video_cfg = HydroDiffConfig.from_preset("hydrodiff-s", **TINY, modalities=["image", "video"])
    cos, _ = rotary_nd(image_cfg, (1, 4, 4), "cpu")
    assert cos.shape == (16, image_cfg.head_dim)
    cos_v, sin_v = rotary_nd(video_cfg, (2, 4, 4), "cpu")
    assert cos_v.shape == (32, video_cfg.head_dim)
    assert not torch.allclose(cos_v[:16], cos_v[16:])  # the two frames differ in position
    # frame 0 of a video has the same (zero) time angle as an image of a video model
    torch.testing.assert_close(cos_v[:16], rotary_nd(video_cfg, (1, 4, 4), "cpu")[0])


def test_text_changes_the_prediction_and_null_prompt_is_unconditional():
    model = make()
    x, t = torch.randn(2, 16, 48), torch.rand(2, 16)
    text, pool = prompts(2)
    with torch.no_grad():
        a = model.model(x, t, text=text, text_pool=pool).prediction
        b = model.model(x, t, text=text.flip(0), text_pool=pool.flip(0)).prediction
        null = model.model(x, t).prediction
        explicit = model.model(x, t, text=model.model.null_text.expand(2, -1, -1), text_pool=model.model.null_text_pool.expand(2, -1)).prediction
    assert not torch.allclose(a, b)
    torch.testing.assert_close(null, explicit)


def test_active_tokens_match_full_run_with_text_and_odd_grid():
    model = make()
    grid = (1, 3, 5)
    n = 15
    g = torch.Generator().manual_seed(0)
    x, t = torch.randn(2, n, 48, generator=g), torch.rand(2, n, generator=g)
    text, pool = prompts(2)
    index = torch.stack([torch.randperm(n, generator=g)[:6] for _ in range(2)])
    cache = HydroDiffTokenCache()
    with torch.no_grad():
        full = model.model(x, t, None, None, cache, grid=grid, text=text, text_pool=pool)
        active = model.model(gather_tokens(x, index), gather_tokens(t, index), None, index, cache, grid=grid, text=text, text_pool=pool)
    torch.testing.assert_close(active.prediction, gather_tokens(full.prediction, index), atol=1e-5, rtol=1e-4)


def test_training_on_any_size_and_text_dropout_trains_the_null_prompt():
    model = make().train()
    text, pool = prompts(4)
    out = model(torch.rand(4, 3, 12, 20) * 2 - 1, text=text, text_pool=pool)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert out.prediction.shape == (4, 15, 48)
    model.zero_grad()
    model.config.text_dropout = 1.0
    model(torch.rand(4, 3, 12, 20) * 2 - 1, text=text, text_pool=pool).loss.backward()
    assert model.model.null_text.grad.abs().sum() > 0


def test_video_training_gives_some_first_frames_clean():
    model = make(modalities=["video"], image_cond_prob=1.0).train()
    text, pool = prompts(2)
    out = model(torch.rand(2, 3, 3, 8, 8) * 2 - 1, text=text, text_pool=pool)
    assert out.timesteps.shape == (2, 12)
    assert (out.timesteps[:, :4] == 0).all() and (out.timesteps[:, 4:] > 0).any()
    out.loss.backward()


def test_generate_text_to_image_with_guidance_any_size():
    model = make(guidance_scale=3)
    text, pool = prompts(2)
    out = model.generate(text=text, text_pool=pool, height=12, width=20, steps=8, min_token_steps=0,
                         generator=torch.Generator().manual_seed(0))
    assert out.images.shape == (2, 3, 12, 20) and torch.isfinite(out.images).all()
    assert out.compute < 1  # the noise mask finishes sure tokens early
    # guidance off, on and a negative prompt give different images
    off = model.generate(text=text, text_pool=pool, steps=8, guidance_scale=1, generator=torch.Generator().manual_seed(0))
    on = model.generate(text=text, text_pool=pool, steps=8, generator=torch.Generator().manual_seed(0))
    neg_t, neg_p = prompts(2, seed=5)
    neg = model.generate(text=text, text_pool=pool, steps=8, negative_text=neg_t, negative_text_pool=neg_p,
                         generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(off.images, on.images) and not torch.allclose(on.images, neg.images)


def test_modalities_are_enforced():
    image_model, video_model = make(), make(modalities=["video"])
    text, pool = prompts(1)
    with pytest.raises(ValueError, match="not built for video"):
        image_model.generate(text=text, text_pool=pool, num_frames=4, steps=4)
    with pytest.raises(ValueError, match="video-only"):
        video_model.generate(text=text, text_pool=pool, steps=4)
    both = make(modalities=["image", "video"])
    assert both.generate(text=text, text_pool=pool, steps=4).images.shape == (1, 3, 16, 16)
    assert both.generate(text=text, text_pool=pool, steps=4, num_frames=3).images.shape == (1, 3, 3, 16, 16)


def test_image_to_video_keeps_the_given_first_frame():
    model = make(modalities=["video"])
    text, pool = prompts(2)
    first = torch.rand(2, 3, 1, 8, 12) * 2 - 1
    out = model.generate(text=text, text_pool=pool, cond_frames=first, num_frames=3, steps=8, min_token_steps=0,
                         return_masks=True)
    assert out.images.shape == (2, 3, 3, 8, 12)
    torch.testing.assert_close(out.images[:, :, :1], first)
    assert not out.update_masks[1:, :, :6].any()  # the given frame is never denoised


def test_image_to_image_strength():
    model = make()
    text, pool = prompts(1)
    init = torch.rand(1, 3, 16, 16) * 2 - 1
    gen = lambda s: model.generate(text=text, text_pool=pool, init=init, strength=s, steps=8, min_token_steps=0,  # noqa: E731
                                   generator=torch.Generator().manual_seed(0), return_masks=True)
    weak, strong = gen(0.2), gen(1.0)
    torch.testing.assert_close(weak.noise_masks[0], torch.full_like(weak.noise_masks[0], 0.2))
    assert weak.noise_masks[-1].max() == 0 and weak.compute < 1
    plain = model.generate(text=text, text_pool=pool, steps=8, min_token_steps=0, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(strong.images, plain.images)  # strength 1 forgets the image: text-to-image
    # an oracle that always predicts `init` returns it exactly at any strength
    oracle = make()
    with torch.no_grad():
        for p in oracle.model.proj_out.parameters():
            p.zero_()
    zero = oracle.generate(text=text, text_pool=pool, init=torch.zeros_like(init), strength=0.3, steps=8, min_token_steps=0)
    torch.testing.assert_close(zero.images, torch.zeros_like(init))


def test_bottleneck_patch_embedding_trains():
    model = make(patch_embed_dim=8).train()
    text, pool = prompts(2)
    model(torch.rand(2, 3, 16, 16) * 2 - 1, text=text, text_pool=pool).loss.backward()
    assert model.model.patch_embed[0].weight.grad.abs().sum() > 0


class FakeTokenizer:
    def __call__(self, texts, max_length, **kwargs):
        ids = torch.tensor([[hash(t) % 97 + i for i in range(max_length)] for t in texts])
        return _Batch(input_ids=ids)


class _Batch(dict):
    def to(self, device):
        return self


class FakeTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(200, TEXT_DIM)

    @property
    def device(self):
        return self.embed.weight.device

    def forward(self, input_ids):
        h = self.embed(input_ids)
        return type("Out", (), dict(last_hidden_state=h, text_embeds=h.mean(1)))


def test_pipeline_text_image_video_modes():
    from PIL import Image

    pipe = HydroDiffPipeline(make(modalities=["image", "video"]), FakeTextEncoder(), FakeTokenizer())
    photo = Image.new("RGB", (21, 13), (200, 30, 30))
    out = pipe("a red square", num_inference_steps=4)
    assert len(out.images) == 1 and out.images[0].size == (16, 16)
    out = pipe(["a", "b"], height=10, width=26, num_inference_steps=4)
    assert [im.size for im in out.images] == [(28, 12)] * 2  # nearest multiples of the patch, no crop
    out = pipe("edit", image=photo, strength=0.5, num_inference_steps=4)
    assert out.images[0].size == (20, 12)
    out = pipe("move", image=photo, num_frames=3, num_inference_steps=4)
    assert len(out.frames) == 1 and len(out.frames[0]) == 3 and out.frames[0][0].size == (20, 12)


def test_stream_counts_bytes_and_stops_before_the_budget(tmp_path):
    import io
    import os
    import sys

    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    sys.path.insert(0, "test")
    from train_stream import captions_of, decode, stream

    def jpeg(i):
        buffer = io.BytesIO()
        Image.new("RGB", (40 + i, 30), (i * 20, 0, 0)).save(buffer, "JPEG")
        return buffer.getvalue()

    # an incompressible unused column, and a file well past pyarrow's 64 KB speculative footer read
    rows = [dict(image=dict(bytes=jpeg(i), path=None), sentences_raw=str([f"photo {i}", "x"]), extra=os.urandom(40000))
            for i in range(10)]
    folder = tmp_path / "datasets" / "toy" / "data"
    folder.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), folder / "train-00000-of-00001.parquet", row_group_size=4)
    fs, root = fsspec.filesystem("file"), str(tmp_path / "datasets").replace("\\", "/")

    counter = [0]
    got = list(stream("toy", "train", float("inf"), counter, fs=fs, root=root))
    assert [captions_of(r)[0] for r in got] == [f"photo {i}" for i in range(10)]
    assert "extra" not in got[0]  # only the image and caption columns are fetched
    full = counter[0]
    assert full < 0.5 * (folder / "train-00000-of-00001.parquet").stat().st_size  # the unused column was never read
    counter = [0]
    some = list(stream("toy", "train", full - 1, counter, fs=fs, root=root))
    assert len(some) == 8 and counter[0] <= full - 1  # the last row group (2 rows) would pass the budget: not fetched
    assert [captions_of(r)[0] for r in stream("toy", "train", float("inf"), [0], skip_rows=6, fs=fs, root=root)] == [
        f"photo {i}" for i in range(6, 10)
    ]
    x, nbytes = decode(got[3], multiple=8)
    assert x.dtype == torch.uint8 and x.shape == (3, 32, 40) and nbytes == len(rows[3]["image"]["bytes"])


def test_terminal_prob_trains_some_images_at_pure_noise():
    model = make(terminal_prob=1.0).train()
    t = model.sample_timesteps(4, "cpu", num_tokens=16)
    assert torch.equal(t, torch.ones(4, 16))
    assert (make().sample_timesteps(64, "cpu", num_tokens=16) < 1).all()  # off by default: logit-normal never hits 1
    text, pool = prompts(4)
    out = model(torch.rand(4, 3, 16, 16) * 2 - 1, text=text, text_pool=pool)
    assert torch.isfinite(out.loss)
