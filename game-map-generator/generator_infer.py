import os
from torchvision.utils import save_image
from generator_model import Generator
import uuid
import torch

def generate_fake_map_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_size = 100

    model = Generator(latent_size).to(device)
    model.load_state_dict(torch.load("generator.pth", map_location=device))
    model.eval()

    noise = torch.randn(1, latent_size, 1, 1, device=device)

    # 🔧 Static klasörü = bulunduğun dizindeki "static/"
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)

    filename = f"generated_{uuid.uuid4().hex[:8]}.png"
    full_path = os.path.join(static_dir, filename)

    print("📁 Kaydedilen yol:", full_path)

    save_image(model(noise), full_path, normalize=True)

    return filename  # sadece dosya adı dön
