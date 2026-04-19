import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_fraction(self, threshold=1e-2):
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()


class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = PrunableLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = PrunableLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

    def prunable_layers(self):
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def sparsity_loss(self):
        return sum(
            torch.sigmoid(layer.gate_scores).sum()
            for layer in self.prunable_layers()
        )

    def overall_sparsity(self, threshold=1e-2):
        total = pruned = 0
        for layer in self.prunable_layers():
            gates = layer.get_gates()
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
        return pruned / total if total > 0 else 0.0

    def all_gate_values(self):
        vals = [layer.get_gates().cpu().numpy().ravel() for layer in self.prunable_layers()]
        return np.concatenate(vals)


def get_cifar10_loaders(batch_size=128):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, lam):
    model.train()
    total_loss = clf_loss_sum = sp_loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        clf_loss = F.cross_entropy(logits, labels)
        sp_loss = model.sparsity_loss()
        loss = clf_loss + lam * sp_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        clf_loss_sum += clf_loss.item()
        sp_loss_sum += sp_loss.item()
    n = len(loader)
    return total_loss / n, clf_loss_sum / n, sp_loss_sum / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total, loss_sum / total


def run_experiment(lam, train_loader, test_loader, device, epochs=30, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"  Lambda = {lam}  |  Epochs = {epochs}")
    print(f"{'='*60}")
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(1, epochs + 1):
        tr_loss, clf_loss, sp_loss = train_one_epoch(model, train_loader, optimizer, device, lam)
        val_acc, _ = evaluate(model, test_loader, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            sparsity = model.overall_sparsity()
            print(f"  Epoch {epoch:3d}/{epochs} | Loss={tr_loss:.4f} | Acc={val_acc:.2f}% | Sparsity={100*sparsity:.1f}%")
    test_acc, _ = evaluate(model, test_loader, device)
    sparsity = model.overall_sparsity()
    gate_vals = model.all_gate_values()
    print(f"\n  Final Test Accuracy : {test_acc:.2f}%")
    print(f"  Final Sparsity Level: {100*sparsity:.2f}%")
    return test_acc, sparsity, gate_vals, model


def plot_gate_distributions(results, best_lam):
    lambdas = sorted(results.keys())
    n = len(lambdas)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, lam in zip(axes, lambdas):
        gate_vals = results[lam]["gates"]
        color = "darkorange" if lam == best_lam else "steelblue"
        ax.hist(gate_vals, bins=60, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"λ={lam} | Acc={results[lam]['accuracy']:.1f}% | Sparsity={100*results[lam]['sparsity']:.1f}%")
        ax.set_xlabel("Gate Value")
        ax.set_ylabel("Count")
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5, label="Threshold")
        ax.legend()
    plt.suptitle("Gate Value Distributions – Self-Pruning Network (CIFAR-10)", y=1.02)
    plt.tight_layout()
    plt.savefig("gate_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: gate_distributions.png")


def plot_tradeoff(results):
    lambdas = sorted(results.keys())
    accs = [results[l]["accuracy"] for l in lambdas]
    sparsities = [100 * results[l]["sparsity"] for l in lambdas]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.set_xlabel("Lambda (λ)")
    ax1.set_ylabel("Test Accuracy (%)", color="steelblue")
    ax1.plot(lambdas, accs, "o-", color="steelblue", linewidth=2, markersize=8, label="Test Accuracy")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Sparsity Level (%)", color="darkorange")
    ax2.plot(lambdas, sparsities, "s--", color="darkorange", linewidth=2, markersize=8, label="Sparsity")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    plt.title("Accuracy vs Sparsity Trade-off")
    plt.tight_layout()
    plt.savefig("tradeoff_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: tradeoff_curve.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, test_loader = get_cifar10_loaders(batch_size=256)
    lambda_values = [1e-5, 1e-4, 1e-3]
    results = {}
    for lam in lambda_values:
        acc, sparsity, gate_vals, _ = run_experiment(lam, train_loader, test_loader, device, epochs=30)
        results[lam] = {"accuracy": acc, "sparsity": sparsity, "gates": gate_vals}
    print("\n" + "=" * 50)
    print(f"  {'Lambda':>10}  |  {'Accuracy':>10}  |  {'Sparsity':>10}")
    print("=" * 50)
    for lam in lambda_values:
        r = results[lam]
        print(f"  {lam:>10}  |  {r['accuracy']:>9.2f}%  |  {100*r['sparsity']:>9.2f}%")
    print("=" * 50)
    best_lam = max(results, key=lambda l: results[l]["accuracy"])
    plot_gate_distributions(results, best_lam)
    plot_tradeoff(results)


if __name__ == "__main__":
    main()
