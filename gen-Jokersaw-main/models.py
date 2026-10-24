import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
        )

        self.decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        h = self.decoder_input(z)
        x_reconstructed = self.decoder(h)
        return x_reconstructed

    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed


class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes

        self.input_layer = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.LeakyReLU(0.2),
        )

        self.net = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 7x7 -> 14x14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),  # 14x14 -> 28x28
            nn.Tanh(),
        )

    def forward(self, z, y_vec):
        x = torch.cat([z, y_vec], dim=1)
        x = self.input_layer(x)
        x = self.net(x)

        return x


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.num_classes = num_classes

        self.label_projection = nn.Linear(num_classes, 28 * 28)

        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1),
        )

    def forward(self, x, y_vec):
        label_map = self.label_projection(y_vec)
        label_map = label_map.view(-1, 1, 28, 28)

        x = torch.cat([x, label_map], dim=1)

        logits = self.net(x)

        return logits
