
#!/usr/bin/env python
# coding: utf-8

# # GAN utils
# # ================================================

# # Imports
# # ================================================

import torch.nn as nn
import torch
import torchvision.utils as vutils
import os
from outputs.outputs_utils import save_image_grid
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_g=64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (B, z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, feature_g * 8, 4, 1, 0),     # → (B, 512, 4, 4)
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1),  # → (B, 256, 8, 8)
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1),  # → (B, 128, 16, 16)
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1),      # → (B, 64, 32, 32)
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, feature_g, 4, 2, 1),          # → (B, 64, 64, 64)
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, feature_g, 4, 2, 1),          # → (B, 64, 128, 128)
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, feature_g, 4, 2, 1),          # → (B, 64, 256, 256)
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1),       # → (B, 3, 512, 512)
            nn.Tanh()
        )

    def forward(self, z):
        out = self.net(z)  # → (B, 3, 512, 512)
        out = F.interpolate(out, size=(300, 300), mode="bilinear", align_corners=False)  # Resize → 300×300
        return out

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1),       # → (B, 64, 150, 150)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1),       # → (B, 128, 75, 75)
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1),   # → (B, 256, 37, 37)
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1),   # → (B, 512, 18, 18)
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 8, 1, 4, 2, 1),               # → (B, 1, 9, 9)
            nn.AdaptiveAvgPool2d(1),                            # → (B, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1)
    


class WassersteinLoss(nn.Module):
    """
    Implements the Wasserstein loss used in WGAN.
    For real samples: want high scores (maximize D(x))
    For fake samples: want low scores (minimize D(G(z)))
    """
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, predictions, is_real):
        if is_real:
            # D(real) -> maximize => loss = -mean(D(real))
            return -predictions.mean()
        else:
            # D(fake) -> minimize => loss = +mean(D(fake))
            return predictions.mean()


def train_gan_model(G, D, train_loader, val_loader, optimizer_G, optimizer_D, device, loss_fn,
                    num_epochs=1, early_stopping=False, patience=5, save_dir="outputs",
                    n_critic=4):  # ← stratégie 1 : nombre d'updates D pour 1 update G

    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    os.makedirs(save_dir, exist_ok=True)

    best_G_loss = float("inf")
    best_D_loss = float("inf")
    stop_counter = 0

    for epoch in range(num_epochs):
        G.train()
        D.train()
        total_G_loss = 0
        total_D_loss = 0

        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # noisy_real = real_imgs + torch.randn_like(real_imgs) * 0.05

            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # real_labels = torch.ones(B) * 0.9  # 

            """ WassersteinLoss
            
            # Pour entraîner le discriminateur
            D_loss = loss_fn(D(real_imgs), is_real=True) + loss_fn(D(fake_imgs.detach()), is_real=False)

            # Pour entraîner le générateur
            G_loss = loss_fn(D(fake_imgs), is_real=True)

             À ne pas oublier pour WGAN classique
                Enlever les Sigmoid() dans la dernière couche du Discriminateur.

                Clip les poids du Discriminateur (WGAN original) :

            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)
            

            """

            ### === Train Discriminator === ###
            if batch_idx % n_critic == 0:
                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                fake_imgs = G(noise).detach()
                D_loss_real = loss_fn(D(real_imgs), real_labels)
                D_loss_fake = loss_fn(D(fake_imgs), fake_labels)
                D_loss = D_loss_real + D_loss_fake

                optimizer_D.zero_grad()
                D_loss.backward()
                optimizer_D.step()

                total_D_loss += D_loss.item()

            ### === Train Generator  === ###
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_imgs = G(noise)
            G_loss = loss_fn(D(fake_imgs), real_labels)

            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

            total_G_loss += G_loss.item()

        # Moyennes
        avg_D = total_D_loss / len(train_loader)
        avg_G = total_G_loss / (len(train_loader) // n_critic + 1)

        print(f"[Epoch {epoch+1}/{num_epochs}] D_loss: {avg_D:.4f} | G_loss: {avg_G:.4f}")

        # Visualisation
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
            save_image_grid(fake, save_path=os.path.join(save_dir, f"epoch_{epoch+1:03d}.png"), title=f"Epoch {epoch+1}")

        # === Sauvegarde des modèles ===
        model_save_dir = os.path.join("models\\generation", "checkpoints")
        os.makedirs(model_save_dir, exist_ok=True)

        if avg_G < best_G_loss - 1e-4:
            best_G_loss = avg_G
            torch.save(G.state_dict(), os.path.join(model_save_dir, "best_generator.pth"))

        if avg_D < best_D_loss - 1e-4:
            best_D_loss = avg_D
            torch.save(D.state_dict(), os.path.join(model_save_dir, "best_discriminator.pth"))

        # === Early Stopping sur G ===
        if early_stopping:
            if avg_G < best_G_loss - 1e-4:
                stop_counter = 0
            else:
                stop_counter += 1
                if stop_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break