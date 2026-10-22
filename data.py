import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

transform = transforms.Compose([
    transforms.ToTensor(),                           
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))  
])


def get_loaders(batch_size=64, val_size=6000, seed=42):
    full_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)  
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()

    print(f"Train      : {len(train_loader.dataset)} görüntü")
    print(f"Validation : {len(val_loader.dataset)} görüntü")
    print(f"Test       : {len(test_loader.dataset)} görüntü")

    images, labels = next(iter(train_loader))
    print(f"Bir batch  : images {tuple(images.shape)}, labels {tuple(labels.shape)}")
    print(f"Piksel aralığı (normalize sonrası): {images.min():.2f} .. {images.max():.2f}")

    
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
    for ax, img, label in zip(axes.flat, images[:16], labels[:16]):
        ax.imshow(img.squeeze() * MNIST_STD + MNIST_MEAN, cmap="gray")
        ax.set_title(int(label))
        ax.axis("off")
    plt.suptitle("MNIST örnekleri (train)")
    plt.tight_layout()
    plt.savefig("results/mnist_samples.png", dpi=120)
    print("Örnek görseller kaydedildi: results/mnist_samples.png")
