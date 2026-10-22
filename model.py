
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=(256, 128), num_classes=10):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))

        self.flatten = nn.Flatten()          
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MLP()
    print(model)
    print(f"\nEğitilebilir parametre sayısı: {count_parameters(model):,}")

    #
    dummy = torch.randn(64, 1, 28, 28)
    out = model(dummy)
    print(f"Girdi şekli : {tuple(dummy.shape)}")
    print(f"Çıktı şekli : {tuple(out.shape)}  (64 görüntü x 10 sınıf skoru)")
