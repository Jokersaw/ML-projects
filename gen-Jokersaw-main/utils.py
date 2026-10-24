import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion_mnist(batch_size=128):
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


def print_dataset_info(train_dataset, test_dataset):
    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    x, y = train_dataset[0]
    print("Image shape:", x.shape)
    print("Label:", y)


def train_autoencoder(
    model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10
):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0

        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()

            x_reconstructed = model(x)
            loss = criterion(x_reconstructed, x)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()

        total_test_loss = 0

        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)

                x_reconstructed = model(x)
                loss = criterion(x_reconstructed, x)

                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"train loss: {avg_train_loss:.6f}, "
            f"test loss: {avg_test_loss:.6f}"
        )

    return train_losses, test_losses


def plot_losses(train_losses, test_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    plt.plot(epochs, test_losses, marker="o", label="Test loss")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder training curve")
    plt.legend()
    plt.grid()
    plt.show()


def show_reconstructions(model, loader, device, n=10):
    model.eval()

    x, labels = next(iter(loader))
    x = x[:n].to(device)
    labels = labels[:n]

    with torch.no_grad():
        x_reconstructed = model(x)

    x = x.cpu()
    x_reconstructed = x_reconstructed.cpu()

    plt.figure(figsize=(2 * n, 4))

    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(x[i, 0], cmap="gray")
        plt.title(CLASS_NAMES[labels[i].item()])
        plt.axis("off")

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_reconstructed[i, 0], cmap="gray")
        plt.axis("off")

    plt.suptitle("Original images / Reconstructed images")
    plt.tight_layout()
    plt.show()


def show_samples(samples, title, nrow=8):
    n = len(samples)

    plt.figure(figsize=(1.7 * nrow, 2 * ((n + nrow - 1) // nrow)))

    for i in range(n):
        plt.subplot((n + nrow - 1) // nrow, nrow, i + 1)
        plt.imshow(samples[i, 0], cmap="gray")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def get_class_images(dataset, class_id, max_count=1000):
    images = []

    for x, y in dataset:
        if y == class_id:
            images.append(x)

        if len(images) >= max_count:
            break

    return torch.stack(images)


def get_class_image_vectors(dataset, class_id, max_count=1000):
    images = get_class_images(dataset, class_id, max_count=max_count)
    vectors = images.view(images.size(0), -1)

    return vectors


def image_vectors_to_images(vectors):
    images = vectors.clamp(0, 1)
    images = images.view(vectors.size(0), 1, 28, 28)

    return images


def get_class_latent_vectors(model, dataset, class_id, device, max_count=1000):
    model.eval()

    latents = []

    with torch.no_grad():
        for x, y in dataset:
            if y == class_id:
                x = x.unsqueeze(0).to(device)
                z = model.encode(x)
                latents.append(z.cpu().squeeze(0))

            if len(latents) >= max_count:
                break

    return torch.stack(latents)


def latent_vectors_to_images(model, vectors, device):
    model.eval()

    vectors = vectors.to(device)

    with torch.no_grad():
        images = model.decode(vectors)

    images = images.cpu().clamp(0, 1)

    return images


# В пространстве изображений (784-мерном) ковариационная
# матрица близка к вырожденной из-за линейной зависимости пикселей.
# Регуляризация eps спасает разложение Холецкого, но раздувает
# дисперсию в незначимых направлениях. В латентном пространстве (32D)
# эта проблема отсутствует.
def gaussian_sample_vectors(vectors, n=16, eps=1e-4):
    mean = vectors.mean(dim=0)

    centered = vectors - mean

    cov = centered.T @ centered / (vectors.size(0) - 1)

    cov = cov + eps * torch.eye(cov.size(0), device=vectors.device)

    L = torch.linalg.cholesky(cov)

    noise = torch.randn(n, vectors.size(1), device=vectors.device)

    samples = mean + noise @ L.T

    return samples


def smote_sample_vectors(vectors, n=16, k=5):
    distances = torch.cdist(vectors, vectors)

    distances.fill_diagonal_(float("inf"))

    nearest_neighbors = distances.topk(k, largest=False).indices

    samples = []

    for _ in range(n):
        i = np.random.randint(0, len(vectors))

        neighbor_pos = np.random.randint(0, k)
        j = nearest_neighbors[i, neighbor_pos].item()

        alpha = np.random.rand()

        new_vector = vectors[i] + alpha * (vectors[j] - vectors[i])

        samples.append(new_vector)

    return torch.stack(samples)


def train_conditional_gan(
    generator,
    discriminator,
    train_loader,
    device,
    noise_dim=100,
    num_classes=10,
    num_epochs=30,
    lr_g=2e-4,
    lr_d=1e-4,
    g_steps=2,
):
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999)
    )

    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        total_g_loss = 0
        total_d_loss = 0
        total_g_steps = 0

        for real_images, labels in train_loader:
            real_images = real_images.to(device)
            labels = labels.to(device)

            # [0, 1] -> [-1, 1]
            real_images = real_images * 2 - 1

            batch_size = real_images.size(0)

            y_vec = (
                torch.nn.functional.one_hot(labels, num_classes=num_classes)
                .float()
                .to(device)
            )

            real_targets = torch.ones(batch_size, 1).to(device) * 0.9
            fake_targets = torch.zeros(batch_size, 1).to(device)

            # Discriminator

            optimizer_d.zero_grad()

            real_logits = discriminator(real_images, y_vec)
            d_real_loss = criterion(real_logits, real_targets)

            z = torch.randn(batch_size, noise_dim).to(device)
            fake_images = generator(z, y_vec)

            fake_logits = discriminator(fake_images.detach(), y_vec)
            d_fake_loss = criterion(fake_logits, fake_targets)

            # Делим на 2, чтобы не усиливать градиент дискриминатора слишком сильно
            d_loss = 0.5 * (d_real_loss + d_fake_loss)

            d_loss.backward()
            optimizer_d.step()

            # Generator

            for _ in range(g_steps):
                optimizer_g.zero_grad()

                z = torch.randn(batch_size, noise_dim).to(device)
                fake_images = generator(z, y_vec)

                fake_logits = discriminator(fake_images, y_vec)

                g_loss = criterion(fake_logits, torch.ones(batch_size, 1).to(device))

                g_loss.backward()
                optimizer_g.step()

                total_g_loss += g_loss.item()
                total_g_steps += 1

            total_d_loss += d_loss.item()

        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / total_g_steps

        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"D loss: {avg_d_loss:.6f}, "
            f"G loss: {avg_g_loss:.6f}"
        )

    return g_losses, d_losses


def plot_gan_losses(g_losses, d_losses):
    epochs = range(1, len(g_losses) + 1)

    plt.figure(figsize=(7, 4))

    plt.plot(epochs, d_losses, marker="o", label="Discriminator loss")
    plt.plot(epochs, g_losses, marker="o", label="Generator loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Conditional GAN training curves")
    plt.legend()
    plt.grid()
    plt.show()


def generate_by_classes(
    generator, device, noise_dim=100, num_classes=10, n_per_class=8
):
    generator.eval()

    labels = torch.arange(num_classes).repeat_interleave(n_per_class).to(device)

    y_vec = (
        torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)
    )

    z = torch.randn(len(labels), noise_dim).to(device)

    with torch.no_grad():
        samples = generator(z, y_vec)

    # generator выдает [-1, 1], переводим обратно в [0, 1]
    samples = (samples + 1) / 2
    samples = samples.cpu().clamp(0, 1)

    return samples


def show_generated_by_classes(samples, class_names, n_per_class=8):
    num_classes = len(class_names)

    plt.figure(figsize=(1.5 * n_per_class, 1.5 * num_classes))

    for class_id in range(num_classes):
        for j in range(n_per_class):
            idx = class_id * n_per_class + j

            plt.subplot(num_classes, n_per_class, idx + 1)
            plt.imshow(samples[idx, 0], cmap="gray")
            plt.axis("off")

            if j == 0:
                plt.ylabel(class_names[class_id], rotation=0, labelpad=40)

    plt.suptitle("Conditional GAN: generated images by class")
    plt.tight_layout()
    plt.show()


def generate_class_mixture(
    generator, class_a, class_b, device, noise_dim=100, num_classes=10, steps=9
):
    generator.eval()

    alphas = torch.linspace(0, 1, steps).to(device)

    y_vec = torch.zeros(steps, num_classes).to(device)
    y_vec[:, class_a] = 1 - alphas
    y_vec[:, class_b] = alphas

    # Один и тот же шум для всех картинок,
    # чтобы менялось именно условие класса, а не случайный шум
    z = torch.randn(1, noise_dim).to(device)
    z = z.repeat(steps, 1)

    with torch.no_grad():
        samples = generator(z, y_vec)

    # [-1, 1] -> [0, 1]
    samples = (samples + 1) / 2
    samples = samples.cpu().clamp(0, 1)

    return samples, alphas.cpu()


def show_class_mixture(samples, alphas, class_a, class_b, class_names):
    steps = len(samples)

    plt.figure(figsize=(1.7 * steps, 2.5))

    for i in range(steps):
        plt.subplot(1, steps, i + 1)
        plt.imshow(samples[i, 0], cmap="gray")
        plt.axis("off")
        plt.title(f"{alphas[i].item():.2f}")

    plt.suptitle(f"Class mixture: {class_names[class_a]} → {class_names[class_b]}")
    plt.tight_layout()
    plt.show()
