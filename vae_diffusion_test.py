import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# HYPERPARAMETERS
latent_dim = 128
style_dim = 64
num_steps = 1000
num_epochs = 50
fixed_size = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, style_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)
        self.style_encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, style_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def encode_style(self, x):
        return self.style_encoder(x)

    def decode(self, z, style):
        decoded = self.decoder(z)
        batch_size, channels, height, width = decoded.size()
        style = style.view(batch_size, -1, 1, 1)
        style = style.repeat(1, 1, height, width)
        decoded = decoded + style
        return decoded

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        style = self.encode_style(x)
        recon_x = self.decode(z, style)
        return recon_x, mu, log_var, style

class DiffusionModel(nn.Module):
    def __init__(self, latent_dim, style_dim, num_steps):
        super(DiffusionModel, self).__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.num_steps = num_steps
        self.betas = self._get_betas()
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.denoise_fn = nn.Sequential(
            nn.Linear(latent_dim + style_dim, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, latent_dim + style_dim)
        )

    def _get_betas(self):
        return torch.linspace(1e-4, 0.02, self.num_steps)

    def forward(self, z, t):
        t = t.to(self.alpha_bars.device) # Move t to the same device as self.alpha_bars
        alpha_bar = self.alpha_bars[t]
        alpha_bar = alpha_bar.to(z.device) # Move alpha_bar to the same device as z
        sqrt_alpha_bar = torch.sqrt(alpha_bar).unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar).unsqueeze(-1).unsqueeze(-1).to(z.device)

        noise = torch.randn_like(z) # Generate noise on the same device as z

        noisy_z = sqrt_alpha_bar * z + sqrt_one_minus_alpha_bar * noise
        batch_size, _, _ = noisy_z.size()
        noisy_z = noisy_z.view(batch_size, self.latent_dim + self.style_dim)
        denoised_z = self.denoise_fn(noisy_z)
        denoised_z = denoised_z.view(batch_size, self.latent_dim, self.style_dim)
        return denoised_z, noise

# Load and preprocess the dataset
transform = transforms.Compose([
    transforms.Resize((fixed_size, fixed_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

import os
from torch.utils.data import Dataset

class CustomFlickr30k(Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.root = root
        self.ann_file = ann_file
        self.transform = transform
        self.annotations = self.load_annotations()

    def load_annotations(self):
        annotations = {}
        with open(self.ann_file, 'r', encoding='utf-8') as file:
            next(file)  # Skip the header row
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split("|")
                    image_id = parts[0].strip()
                    caption = parts[-1].strip()
                    if image_id not in annotations:
                        annotations[image_id] = []
                    annotations[image_id].append(caption)
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_id = list(self.annotations.keys())[index]
        image_path = os.path.join(self.root, image_id)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        captions = self.annotations[image_id]
        return image, captions

if __name__ == '__main__':
    print("Loading dataset...")
    dataset = CustomFlickr30k(root='flickr30k_images/flickr30k_images', ann_file='flickr30k_images/results.csv', transform=transform)
    subset_indices = list(range(0, len(dataset), 50))  # Use every Nth sample
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    print("Initializing models...")
    vae = VAE(input_dim=3, latent_dim=latent_dim, style_dim=style_dim).to(device)
    diffusion_model = DiffusionModel(latent_dim, style_dim, num_steps).to(device)

    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss()

    vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-3)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title('Style Transfer Result')
    axs[1].set_title('Loss Graph')
    plt.tight_layout()
    print("Starting training...")

    for epoch in range(num_epochs):
        vae_losses = []
        style_losses = []
        diffusion_losses = []

        for images, _ in dataloader:
            images = images.to(device)

            # VAE training
            recon_images, mu, log_var, style = vae(images)
            recon_loss = mse_loss(recon_images, images)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            style_loss = mse_loss(style, vae.encode_style(images))
            vae_loss = recon_loss + kl_div + style_loss

            vae_optimizer.zero_grad()
            vae_loss.backward()
            vae_optimizer.step()

            # Diffusion model training
            content_latent, _ = vae.encode(images)
            style_latent = vae.encode_style(images)
            combined_latent = torch.cat([content_latent, style_latent], dim=1)  # Concatenate along the channel dimension
            combined_latent = combined_latent.to(device)
            denoised_latent, noise = diffusion_model(combined_latent, torch.randint(0, num_steps, (images.size(0),), device=device))
            denoised_images = vae.decode(denoised_latent[:, :latent_dim])  # Extract the denoised content latent
            diffusion_loss = mse_loss(denoised_images, images)

            diffusion_optimizer.zero_grad()
            diffusion_loss.backward()
            diffusion_optimizer.step()

            vae_losses.append(vae_loss.item())
            style_losses.append(style_loss.item())
            diffusion_losses.append(diffusion_loss.item())

        # Live style transfer result
        with torch.no_grad():
            example_content = images[0].unsqueeze(0)
            example_style = images[1].unsqueeze(0)

            content_latent, _ = vae.encode(example_content)
            style_latent = vae.encode_style(example_style)
            combined_latent = content_latent + style_latent
            transferred_latent, _ = diffusion_model(combined_latent, torch.tensor([num_steps - 1], device=device))
            transferred_image = vae.decode(transferred_latent, style_latent)

        transferred_image = (transferred_image.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2
        axs[0].imshow(transferred_image)

        axs[1].clear()
        axs[1].plot(vae_losses, label='VAE Loss')
        axs[1].plot(style_losses, label='Style Loss')
        axs[1].plot(diffusion_losses, label='Diffusion Loss')
        axs[1].legend()

        fig.canvas.draw()
        plt.pause(0.001)

        print(f"Epoch [{epoch+1}/{num_epochs}], VAE Loss: {np.mean(vae_losses):.4f}, Style Loss: {np.mean(style_losses):.4f}, Diffusion Loss: {np.mean(diffusion_losses):.4f}")

    print("Saving models...")
    torch.save(vae.state_dict(), 'vae.pth')
    torch.save(diffusion_model.state_dict(), 'diffusion_model.pth')
    print("Models saved.")

    print("Starting inference...")
    content_image = Image.open('path/to/content/image.jpg')
    style_image = Image.open('path/to/style/image.jpg')

    content_tensor = transform(content_image).unsqueeze(0).to(device)
    style_tensor = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_latent, _ = vae.encode(content_tensor)
        style_latent = vae.encode_style(style_tensor)
        combined_latent = content_latent + style_latent
        transferred_latent, _ = diffusion_model(combined_latent, torch.tensor([num_steps - 1], device=combined_latent.device))
        transferred_image = vae.decode(transferred_latent)

    transferred_image = (transferred_image.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2
    transferred_image = (transferred_image * 255).astype(np.uint8)
    transferred_image = Image.fromarray(transferred_image)
    transferred_image.save('transferred_image.jpg')