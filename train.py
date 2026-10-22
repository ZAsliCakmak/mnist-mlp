import copy
import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import get_loaders
from model import MLP, count_parameters


CONFIG = {
    "experiment": "deney1_temel_mlp",
    "hidden_sizes": [256, 128],
    "activation": "ReLU",
    "loss": "CrossEntropyLoss",
    "optimizer": "Adam",      
    "learning_rate": 0.001,
    "momentum": 0.9,          
    "batch_size": 64,
    "epochs": 15,
    "val_size": 6000,
    "seed": 42,
}



def run_epoch(model, loader, criterion, device, optimizer=None):
    
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def plot_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], "o-", label="Train loss")
    ax1.plot(epochs, history["val_loss"], "s-", label="Validation loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train accuracy")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Validation accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=CONFIG["batch_size"], val_size=CONFIG["val_size"], seed=CONFIG["seed"]
    )

    model = MLP(hidden_sizes=tuple(CONFIG["hidden_sizes"])).to(device)
    criterion = nn.CrossEntropyLoss()
    if CONFIG["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG["learning_rate"],
                                    momentum=CONFIG["momentum"])

    print(f"Cihaz: {device} | Parametre sayısı: {count_parameters(model):,}")
    print(f"Hiperparametreler: {CONFIG}\n")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = 0.0, 0, None

    start = time.time()
    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:2d}/{CONFIG['epochs']} | "
              f"train loss {tr_loss:.4f}  acc {tr_acc*100:.2f}% | "
              f"val loss {va_loss:.4f}  acc {va_acc*100:.2f}%")
    train_time = time.time() - start

    
    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)

    print("\n================ SONUÇLAR ================")
    print(f"Eğitim süresi              : {train_time:.1f} sn")
    print(f"Son epoch train loss / acc : {history['train_loss'][-1]:.4f} / {history['train_acc'][-1]*100:.2f}%")
    print(f"Son epoch val   loss / acc : {history['val_loss'][-1]:.4f} / {history['val_acc'][-1]*100:.2f}%")
    print(f"En iyi val acc             : {best_val_acc*100:.2f}% (epoch {best_epoch})")
    print(f"TEST loss / acc            : {test_loss:.4f} / {test_acc*100:.2f}%")

    plot_history(history, "results/training_curves.png")
    torch.save(model.state_dict(), "results/mlp_model.pth")
    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": CONFIG,
            "device": str(device),
            "train_time_sec": round(train_time, 2),
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "history": history,
        }, f, indent=2, ensure_ascii=False)

    print("\nKaydedildi: results/training_curves.png, results/mlp_model.pth, results/results.json")


if __name__ == "__main__":
    main()
