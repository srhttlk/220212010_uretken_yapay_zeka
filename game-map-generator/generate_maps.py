
import torch
from torchvision.utils import save_image
from dcgan_train import Generator, latent_size
import os

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Klasör
os.makedirs("generated_maps", exist_ok=True)

# Generator yükle
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# Rastgele görseller üret
num_images = 25
noise = torch.randn(num_images, latent_size, 1, 1, device=device)
fake_images = generator(noise)

# Görselleri kaydet
for i in range(num_images):
    save_image(fake_images[i], f"generated_maps/map_{i+1:03d}.png", normalize=True)
print(f"{num_images} görsel 'generated_maps/' klasörüne kaydedildi.")
