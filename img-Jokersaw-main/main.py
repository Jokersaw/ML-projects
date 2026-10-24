import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

batch_size = 128

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

mnist_train = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
mnist_test = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)

fashion_train = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transform
)
fashion_test = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transform
)

mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

fashion_train_loader = DataLoader(fashion_train, batch_size=batch_size, shuffle=True)
fashion_test_loader = DataLoader(fashion_test, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # СВЁРТОЧНАЯ ЧАСТЬ
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 → 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 7x7 → 7x7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 7x7 -> 1x1
        )

        # КЛАССИФИКАТОР
        self.mnist_head = nn.Sequential(
            nn.Flatten(),  # 128 x 1 x 1 → вектор
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        self.fashion_mnist_head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x, head):
        x = self.features(x)
        if head == "mnist":
            x = self.mnist_head(x)
        elif head == "fashion_mnist":
            x = self.fashion_mnist_head(x)
        else:
            raise ValueError("head must be mnist or fashion_mnist")
        return x


def train_one_epoch(model, loader, criterion, optimizer, device, head):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, head=head)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, head):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images, head=head)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def fit(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    epochs,
    head,
    extra_eval=None,
):
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Если сразу две головы тестим
    if extra_eval is not None:
        _, _, extra_name = extra_eval
        history[f"{extra_name}_test_loss"] = []
        history[f"{extra_name}_test_acc"] = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, head=head
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, head=head)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        log = (
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

        if extra_eval is not None:
            extra_loader, extra_head, extra_name = extra_eval
            extra_loss, extra_acc = evaluate(
                model, extra_loader, criterion, device, head=extra_head
            )

            history[f"{extra_name}_test_loss"].append(extra_loss)
            history[f"{extra_name}_test_acc"].append(extra_acc)

            log += f" | {extra_name}_test_acc={extra_acc:.4f}"

        print(log)

    return history


def plot_history(history, title=""):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["test_loss"], label="test loss")

    # Если сразу две головы тестим
    for key in history:
        if key.endswith("_test_loss") and key != "test_loss":
            plt.plot(epochs, history[key], label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["test_acc"], label="test acc")

    for key in history:
        if key.endswith("_test_acc") and key != "test_acc":
            plt.plot(epochs, history[key], label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def set_requires_grad(module, flag):
    for param in module.parameters():
        param.requires_grad = flag


def plot_histories_comparison(histories, labels, metric, title=""):
    plt.figure(figsize=(8, 6))

    for history, label in zip(histories, labels):
        epochs = range(1, len(history[metric]) + 1)
        plt.plot(epochs, history[metric], marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()

set_requires_grad(model.fashion_mnist_head, False)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

history_mnist = fit(
    model,
    mnist_train_loader,
    mnist_test_loader,
    criterion,
    optimizer,
    device,
    epochs=8,
    head="mnist",
)

# Кривые обучения для задачи MNIST
plot_history(history_mnist, "MNIST")

# Сохраняем параметры
os.makedirs("checkpoints", exist_ok=True)

checkpoint = {
    "features": copy.deepcopy(model.features.state_dict()),
    "mnist_head": copy.deepcopy(model.mnist_head.state_dict()),
    "fashion_mnist_head_untrained": copy.deepcopy(
        model.fashion_mnist_head.state_dict()
    ),
}

torch.save(checkpoint, "checkpoints/model_after_mnist.pth")
print("Params saved.")

# Замораживаем общую часть и первую голову
set_requires_grad(model.features, False)
set_requires_grad(model.mnist_head, False)
set_requires_grad(model.fashion_mnist_head, True)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

history_fashion_mnist_head_frozen = fit(
    model,
    fashion_train_loader,
    fashion_test_loader,
    criterion,
    optimizer,
    device,
    epochs=8,
    head="fashion_mnist",
)

plot_history(history_fashion_mnist_head_frozen, "FashionMNIST (frozen features)")

# Размораживаем общую часть, первую голову оставляем замороженной (п. 7)
set_requires_grad(model.features, True)
set_requires_grad(model.mnist_head, False)
set_requires_grad(model.fashion_mnist_head, True)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)

history_fashion_mnist_unfrozen = fit(
    model,
    fashion_train_loader,
    fashion_test_loader,
    criterion,
    optimizer,
    device,
    epochs=6,
    head="fashion_mnist",
    extra_eval=(mnist_test_loader, "mnist", "mnist"),
)

plot_history(history_fashion_mnist_unfrozen, "FashionMNIST (unfrozen features)")

# Восстанавливаем модель с чекпоинта (п.8)
model_from_checkpoint = CNN().to(device)

checkpoint = torch.load("checkpoints/model_after_mnist.pth", map_location=device)

model_from_checkpoint.features.load_state_dict(checkpoint["features"])
model_from_checkpoint.mnist_head.load_state_dict(checkpoint["mnist_head"])
model_from_checkpoint.fashion_mnist_head.load_state_dict(
    checkpoint["fashion_mnist_head_untrained"]
)

# Размораживаем features и вторую голову
set_requires_grad(model_from_checkpoint.features, True)
set_requires_grad(model_from_checkpoint.mnist_head, False)
set_requires_grad(model_from_checkpoint.fashion_mnist_head, True)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_from_checkpoint.parameters()), lr=1e-4
)

history_fashion_mnist_from_checkpoint_unfrozen = fit(
    model_from_checkpoint,
    fashion_train_loader,
    fashion_test_loader,
    criterion,
    optimizer,
    device,
    epochs=8,
    head="fashion_mnist",
    extra_eval=(mnist_test_loader, "mnist", "mnist"),
)

plot_history(
    history_fashion_mnist_from_checkpoint_unfrozen,
    "FashionMNIST (checkpoint + unfrozen from start)",
)

model_random = CNN().to(device)

for layer in model_random.features[:6]:  # Conv + ReLU + Pool (2 слоя)
    set_requires_grad(layer, False)

set_requires_grad(model_random.mnist_head, False)
set_requires_grad(model_random.fashion_mnist_head, True)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_random.parameters()), lr=1e-3
)

history_random_partial = fit(
    model_random,
    fashion_train_loader,
    fashion_test_loader,
    criterion,
    optimizer,
    device,
    epochs=8,
    head="fashion_mnist",
)

plot_history(history_random_partial, "Random init + partial freeze")

# Отображаем всё вместе
# По точности
plot_histories_comparison(
    histories=[
        history_fashion_mnist_head_frozen,
        history_fashion_mnist_unfrozen,
        history_fashion_mnist_from_checkpoint_unfrozen,
        history_random_partial,
    ],
    labels=[
        "Frozen features",
        "Unfreeze after head training",
        "Checkpoint + unfrozen from start",
        "Random init + partial freeze",
    ],
    metric="test_acc",
    title="FashionMNIST: test accuracy comparison",
)

# По лоссу
plot_histories_comparison(
    histories=[
        history_fashion_mnist_head_frozen,
        history_fashion_mnist_unfrozen,
        history_fashion_mnist_from_checkpoint_unfrozen,
        history_random_partial,
    ],
    labels=[
        "Frozen features",
        "Unfreeze after head training",
        "Checkpoint + unfrozen from start",
        "Random init + partial freeze",
    ],
    metric="test_loss",
    title="FashionMNIST: test loss comparison",
)

best_model = model


@torch.no_grad()
def find_most_similar_images(
    model, loader, device, head="fashion_mnist", num_classes=10
):
    model.eval()

    best_scores = np.full((num_classes, num_classes), -np.inf)
    best_images = [[None for _ in range(num_classes)] for _ in range(num_classes)]
    best_true_labels = np.full((num_classes, num_classes), -1)
    best_pred_labels = np.full((num_classes, num_classes), -1)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images, head=head)
        preds = logits.argmax(dim=1)

        for i in range(images.size(0)):
            true_class = labels[i].item()

            for target_class in range(num_classes):
                score = logits[i, target_class].item()

                if score > best_scores[true_class, target_class]:
                    best_scores[true_class, target_class] = score
                    best_images[true_class][target_class] = images[i].detach().cpu()
                    best_true_labels[true_class, target_class] = true_class
                    best_pred_labels[true_class, target_class] = preds[i].item()

    return best_scores, best_images, best_true_labels, best_pred_labels


best_scores, best_images, best_true_labels, best_pred_labels = find_most_similar_images(
    best_model, fashion_test_loader, device, head="fashion_mnist", num_classes=10
)


def denormalize(image):
    return image * 0.5 + 0.5


class_names = fashion_test.classes
print(class_names)

fig, axes = plt.subplots(10, 10, figsize=(20, 20))

for c in range(10):
    for t in range(10):
        ax = axes[c, t]

        img = best_images[c][t]
        if img is not None:
            img = denormalize(img).squeeze(0)
            ax.imshow(img, cmap="gray")

            pred_label = best_pred_labels[c, t]
            score = best_scores[c, t]

            ax.set_title(f"pred={class_names[pred_label]}\n{score:.2f}", fontsize=7)

        if c == 0:
            ax.set_xlabel(class_names[t], fontsize=8)
            ax.xaxis.set_label_position("top")
        if t == 0:
            ax.set_ylabel(class_names[c], fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()
