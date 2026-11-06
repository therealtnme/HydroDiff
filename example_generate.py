"""
Interactive generation.

* A pipeline folder (with `transformer/`, as test/train_stream.py --save-pipeline writes it): type a prompt and get an
  image - or a GIF for video models. `image: <path> | <prompt>` uses an image as well (image-to-image for image
  models, the first frame for video models); `frames: 16 | <prompt>` asks a model that does both for a video.
* A class-conditional checkpoint (CIFAR-10, test/train.py): type a class name and get a grid of 16 images.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from hydrodiff import HydroDiffForImageGeneration  # noqa: E402
from hydrodiff.pipeline_hydrodiff import HydroDiffPipeline, export_to_gif  # noqa: E402


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_pipeline(path):
    pipe = HydroDiffPipeline.from_pretrained(path, device=DEVICE)
    modes = " and ".join(pipe.config.modalities)
    print(f"Generates {modes} from text{' (and an image)' if pipe.config.text_dim else ''}. 'exit' quits.")
    count = 0
    while True:
        line = input("Prompt: ").strip()
        if line.lower() in ("exit", "quit"):
            break
        image, frames = None, None
        while "|" in line and line.split("|", 1)[0].strip().split(":")[0] in ("image", "frames"):
            option, line = (part.strip() for part in line.split("|", 1))
            key, value = (s.strip() for s in option.split(":", 1))
            if key == "image":
                image = Image.open(value)
            else:
                frames = int(value)
        out = pipe(line, image=image, num_frames=frames)
        count += 1
        if out.frames is not None:
            name = export_to_gif(out.frames[0], f"video_{count}.gif")
        else:
            name = f"image_{count}.png"
            out.images[0].save(name)
        print(f"Saved {name} ({out.compute:.0%} of the compute of a dense sampler)")


def run_classes(path):
    from torchvision.utils import save_image

    model = HydroDiffForImageGeneration.from_pretrained(path).to(DEVICE).eval()
    while True:
        name = input("Class: ").strip().lower()
        if name in ("exit", "quit"):
            break
        if name not in CIFAR10_CLASSES:
            print(f"Unknown class. Choose from: {', '.join(CIFAR10_CLASSES)}")
            continue
        labels = torch.full((16,), CIFAR10_CLASSES.index(name), device=DEVICE)
        out = model.generate(labels=labels, steps=16)
        save_image((out.images + 1) / 2, f"{name}.png", nrow=4)
        print(f"Saved 16 images to {name}.png ({out.compute:.0%} of the compute of a dense sampler)")


def main():
    path = input("Path: ").strip()
    if os.path.isdir(os.path.join(path, "transformer")):
        run_pipeline(path)
    else:
        run_classes(path)


if __name__ == "__main__":
    main()
