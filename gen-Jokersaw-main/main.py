import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ConvAutoencoder, ConditionalGenerator, ConditionalDiscriminator

from utils import (
    CLASS_NAMES,
    load_fashion_mnist,
    print_dataset_info,
    train_autoencoder,
    plot_losses,
    show_reconstructions,
    show_samples,
    get_class_images,
    get_class_image_vectors,
    image_vectors_to_images,
    get_class_latent_vectors,
    latent_vectors_to_images,
    gaussian_sample_vectors,
    smote_sample_vectors,
    train_conditional_gan,
    plot_gan_losses,
    generate_by_classes,
    show_generated_by_classes,
    generate_class_mixture,
    show_class_mixture,
)


def main():
    torch.manual_seed(21)
    np.random.seed(21)

    # PART 1. Данные

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dataset, test_dataset, train_loader, test_loader = load_fashion_mnist(
        batch_size=128
    )

    print_dataset_info(train_dataset, test_dataset)

    # PART 2. Автокодировщик

    autoencoder = ConvAutoencoder(latent_dim=32).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    train_losses, test_losses = train_autoencoder(
        model=autoencoder,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
    )

    plot_losses(train_losses, test_losses)

    show_reconstructions(model=autoencoder, loader=test_loader, device=device, n=10)

    # PART 3. Gaussian и SMOTE

    class_id = 7

    real_images = get_class_images(train_dataset, class_id=class_id, max_count=16)

    show_samples(real_images, title=f"Real images: {CLASS_NAMES[class_id]}", nrow=8)

    # Image space

    image_vectors = get_class_image_vectors(
        train_dataset, class_id=class_id, max_count=1000
    )

    gaussian_image_vectors = gaussian_sample_vectors(image_vectors, n=16)

    gaussian_images = image_vectors_to_images(gaussian_image_vectors)

    show_samples(
        gaussian_images,
        title=f"Full Gaussian in image space: {CLASS_NAMES[class_id]}",
        nrow=8,
    )

    smote_image_vectors = smote_sample_vectors(image_vectors, n=16, k=5)

    smote_images = image_vectors_to_images(smote_image_vectors)

    show_samples(
        smote_images, title=f"SMOTE in image space: {CLASS_NAMES[class_id]}", nrow=8
    )

    # Latent space

    latent_vectors = get_class_latent_vectors(
        model=autoencoder,
        dataset=train_dataset,
        class_id=class_id,
        device=device,
        max_count=1000,
    )

    gaussian_latent_vectors = gaussian_sample_vectors(latent_vectors, n=16)

    gaussian_latent_images = latent_vectors_to_images(
        model=autoencoder, vectors=gaussian_latent_vectors, device=device
    )

    show_samples(
        gaussian_latent_images,
        title=f"Gaussian in latent space: {CLASS_NAMES[class_id]}",
        nrow=8,
    )

    smote_latent_vectors = smote_sample_vectors(latent_vectors, n=16, k=5)

    smote_latent_images = latent_vectors_to_images(
        model=autoencoder, vectors=smote_latent_vectors, device=device
    )

    show_samples(
        smote_latent_images,
        title=f"SMOTE in latent space: {CLASS_NAMES[class_id]}",
        nrow=8,
    )

    # PART 4. Conditional GAN

    noise_dim = 100
    num_classes = 10

    generator = ConditionalGenerator(noise_dim=noise_dim, num_classes=num_classes).to(
        device
    )

    discriminator = ConditionalDiscriminator(num_classes=num_classes).to(device)

    g_losses, d_losses = train_conditional_gan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        device=device,
        noise_dim=noise_dim,
        num_classes=num_classes,
        num_epochs=30,
        lr_g=2e-4,
        lr_d=1e-4,
        g_steps=2,
    )

    plot_gan_losses(g_losses=g_losses, d_losses=d_losses)

    generated_samples = generate_by_classes(
        generator=generator,
        device=device,
        noise_dim=noise_dim,
        num_classes=num_classes,
        n_per_class=8,
    )

    show_generated_by_classes(
        samples=generated_samples, class_names=CLASS_NAMES, n_per_class=8
    )

    # PART 5. Class mixture generation

    mixture_samples, alphas = generate_class_mixture(
        generator=generator,
        class_a=7,  # Sneaker
        class_b=9,  # Ankle boot
        device=device,
        noise_dim=noise_dim,
        num_classes=num_classes,
        steps=9,
    )

    show_class_mixture(
        samples=mixture_samples,
        alphas=alphas,
        class_a=7,
        class_b=9,
        class_names=CLASS_NAMES,
    )


if __name__ == "__main__":
    main()
