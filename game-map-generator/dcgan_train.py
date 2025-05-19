
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parametreler
latent_size = 100
image_size = 64
batch_size = 64
num_epochs = 100
lr = 0.0002
dataset_path = "C:\\Users\\serha\\OneDrive\\Masaüstü\\üretken_yapay_zeka\\game-map-generator\\map_dataset"




# Veri dönüşümleri
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Veri yükleyici
dataset = ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Model nesneleri
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Kayıp ve optimizasyon
criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Etiketler
real_label = 1.
fake_label = 0.

# ... tüm kod başı (import, class, model tanımları vs.) aynen kalsın ...

# Loss takibi için listeler
G_losses = []
D_losses = []

# Eğitim
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)

        # === Discriminator - Gerçek ===
        d_optimizer.zero_grad()
        real_output = discriminator(images)
        real_loss = criterion(real_output.flatten(), torch.full_like(real_output.flatten(), real_label))
        real_loss.backward()

        # === Discriminator - Sahte ===
        noise = torch.randn(images.size(0), latent_size, 1, 1, device=device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images.detach())
        fake_loss = criterion(fake_output.flatten(), torch.full_like(fake_output.flatten(), fake_label))
        fake_loss.backward()
        d_optimizer.step()

        # === Generator ===
        g_optimizer.zero_grad()
        fake_output = discriminator(fake_images)
        g_loss = criterion(fake_output.flatten(), torch.full_like(fake_output.flatten(), real_label))
        g_loss.backward()
        g_optimizer.step()

        # ✅ Loss'ları kaydet
        D_losses.append(real_loss.item() + fake_loss.item())
        G_losses.append(g_loss.item())

        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(dataloader)}] "
                  f"D Loss: {real_loss.item() + fake_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Örnek görsel kaydet
    if (epoch + 1) % 5 == 0:
        save_image(fake_images[:25], f"fake_images_epoch_{epoch+1}.png", nrow=5, normalize=True)

# 🔐 Modelleri kaydet
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

# 🎨 Loss Grafiğini çiz
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.title("DCGAN Loss Grafiği")
plt.plot(G_losses, label="G Loss")
plt.plot(D_losses, label="D Loss")
plt.xlabel("İterasyon")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()


