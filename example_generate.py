import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from hydrodiff import HydroDiffForImageGeneration
from torchvision.utils import save_image

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_model(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HydroDiffForImageGeneration.from_pretrained(path)

    model.to(device)
    model.eval()
    return model, device


def generate(model, device, prompt, count=16, steps=32):
    name = prompt.strip().lower()
    if name not in CIFAR10_CLASSES:
        print(f"Unknown class. Choose from: {', '.join(CIFAR10_CLASSES)}")
        return

    labels = torch.full((count,), CIFAR10_CLASSES.index(name), device=device)
    out = model.generate(labels=labels, steps=steps)

    filename = f"{name}.png"
    save_image((out.images + 1) / 2, filename, nrow=4)
    print(f"Saved {count} images to {filename} ({out.compute:.0%} of the compute of a dense sampler)")


def main():
    path = input("Path: ")
    model, device = load_model(path)

    while True:
        prompt = input("Class: ")
        if prompt.strip().lower() in ["exit", "quit"]:
            break

        generate(model, device, prompt)


if __name__ == "__main__":
    main()
