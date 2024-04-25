import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import plotly.graph_objects as go
import numpy as np

N_MELS = 256
MEL_N_FFT = 2048
FIXED_SPECT_LENGTH = 1024


# Data preprocessing
def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
    spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=N_MELS, n_fft=MEL_N_FFT, hop_length=1024)(waveform)
    spectrogram = T.AmplitudeToDB()(spectrogram)
    spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()  # Normalize spectrogram
    return spectrogram.squeeze(0), sample_rate

# Plot example spectrogram
def plot_spectrogram(spectrogram):
    fig = go.Figure(data=go.Heatmap(z=spectrogram.numpy(), colorscale='Viridis'))
    fig.update_layout(title='Mel Spectrogram', xaxis_title='Time', yaxis_title='Mel Frequency')
    fig.show()

# VAE Encoder (using Conv2d)
class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, stride=2, padding=1)
        self.flat_dim = (N_MELS // 4) * (FIXED_SPECT_LENGTH // 4) * hidden_channels*2
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        #print(f"Encoder input shape: {x.shape}")
        x = F.relu(self.conv1(x))
        #print(f"Encoder conv1 output shape: {x.shape}")
        x = F.relu(self.conv2(x))
        #print(f"Encoder conv2 output shape: {x.shape}")
        x = x.view(x.size(0), -1)
        #print(f"Encoder flattened shape: {x.shape}")
        #print(f"Encoder fc_mu weight shape: {self.fc_mu.weight.shape}")
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

# VAE Decoder (using ConvTranspose2d)
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_channels, output_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, (N_MELS // 4) * (FIXED_SPECT_LENGTH // 4) * hidden_channels*2)
        self.deconv1 = nn.ConvTranspose2d(hidden_channels*2, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        #print(f"Decoder input shape: {z.shape}")
        x = self.fc(z)
        #print(f"Decoder fc output shape: {x.shape}")
        x = x.view(x.size(0), -1, N_MELS // 4, FIXED_SPECT_LENGTH // 4)
        #print(f"Decoder reshaped output shape: {x.shape}")
        x = F.relu(self.deconv1(x))
        #print(f"Decoder deconv1 output shape: {x.shape}")
        x = torch.sigmoid(self.deconv2(x))
        #print(f"Decoder deconv2 output shape: {x.shape}")
        return x

# VAE
class VAE(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, hidden_channels, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_channels, input_channels)

    def forward(self, x):
        #print(f"VAE input shape: {x.shape}")
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std




# Training loop
def train(model, dataloader, optimizer):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (noisy_spectrograms, clean_spectrograms, _) in enumerate(dataloader):
            noisy_spectrograms = noisy_spectrograms.to(device)
            clean_spectrograms = clean_spectrograms.to(device)
            
            optimizer.zero_grad()
            reconstructed_spectrograms, mu, log_var = model(noisy_spectrograms)
            reconstruction_loss = F.mse_loss(reconstructed_spectrograms, clean_spectrograms)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + kl_divergence
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

# Inference
def inference(model, spectrogram):
    with torch.no_grad():
        print(f"Inference input shape: {spectrogram.shape}")
        spectrogram = spectrogram.to(device)
        print(f"Inference input shape after moving to device: {spectrogram.shape}")
        
        # Pad the spectrogram to match the fixed_size used during training
        pad_length = FIXED_SPECT_LENGTH - spectrogram.size(-1)
        print(f"Pad length: {pad_length}")
        padded_spectrogram = F.pad(spectrogram, (0, pad_length), mode='constant', value=0)
        print(f"Padded spectrogram shape: {padded_spectrogram.shape}")
        
        padded_spectrogram = padded_spectrogram.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        print(f"Inference input shape after unsqueeze: {padded_spectrogram.shape}")
        reconstructed_spectrogram, _, _ = model(padded_spectrogram)
        print(f"Inference output shape: {reconstructed_spectrogram.shape}")
        
        # Remove padding from the reconstructed spectrogram
        reconstructed_spectrogram = reconstructed_spectrogram.squeeze(0).squeeze(0)[:, :spectrogram.size(-1)]
        print(f"Inference output shape after removing padding: {reconstructed_spectrogram.shape}")
    return reconstructed_spectrogram

# Custom collate function
def collate_fn(batch):
    noisy_spectrograms, clean_spectrograms, sample_rates = zip(*batch)
    
    # Pad the spectrograms to a fixed size
    padded_noisy_batch = []
    padded_clean_batch = []
    for noisy_spec, clean_spec in zip(noisy_spectrograms, clean_spectrograms):
        pad_length = FIXED_SPECT_LENGTH - noisy_spec.size(-1)
        padded_noisy_spec = F.pad(noisy_spec, (0, pad_length), mode='constant', value=0)
        padded_clean_spec = F.pad(clean_spec, (0, pad_length), mode='constant', value=0)
        padded_noisy_batch.append(padded_noisy_spec)
        padded_clean_batch.append(padded_clean_spec)
    
    # Stack the padded spectrograms into single tensors
    stacked_noisy_spectrograms = torch.stack(padded_noisy_batch)
    stacked_clean_spectrograms = torch.stack(padded_clean_batch)
    
    #print(f"Collate function output shapes: {stacked_noisy_spectrograms.shape}, {stacked_clean_spectrograms.shape}")
    
    return stacked_noisy_spectrograms.unsqueeze(1), stacked_clean_spectrograms.unsqueeze(1), sample_rates[0]

# Load and preprocess data
def load_data(data_path, noise_ratio=0.6):
    spectrograms = []
    for root, dirs, files in os.walk(data_path):
        for file in files[:8]:
            if file.endswith(".wav"):
                audio_path = os.path.join(root, file)
                spectrogram, sample_rate = preprocess_audio(audio_path)
                
                # Add noise to the spectrogram
                noise = torch.randn_like(spectrogram) * noise_ratio * spectrogram.abs().mean()
                noisy_spectrogram = spectrogram + noise
                
                spectrograms.append((noisy_spectrogram, spectrogram, sample_rate))
    return spectrograms

if __name__ == "__main__":
    print("Starting...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 200
    batch_size = 60000
    learning_rate = 0.8e-4
    latent_dim = 256
    hidden_channels = 32

    # Create VAE model
    input_channels = 1
    model = VAE(input_channels, hidden_channels, latent_dim).to(device)

    # Create dataloader
    data_path = "26_29_09_2017_KCL\\26-29_09_2017_KCL"
    spectrograms = load_data(data_path)
    print(f"Number of spectrograms: {len(spectrograms)}")

    # Plot an example spectrogram
    for i in range(0):
        example_spectrogram, _ = spectrograms[i]
        plot_spectrogram(example_spectrogram)

    dataloader = torch.utils.data.DataLoader(spectrograms, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True if torch.cuda.is_available() else False)






    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train(model, dataloader, optimizer)





    # Perform inference
    input_spectrogram, sample_rate = preprocess_audio("26_29_09_2017_KCL\\26-29_09_2017_KCL\\ReadText\\PD\\ID02_pd_2_0_0.wav")
    print(f"Input spectrogram shape: {input_spectrogram.shape}")
    output_spectrogram = inference(model, input_spectrogram)
    print(f"Output spectrogram shape: {output_spectrogram.shape}")

    plot_spectrogram(output_spectrogram.cpu()) # Plot the output spectrogram

    print(f"Minimum value of output spectrogram before normalization: {output_spectrogram.min()}")
    print(f"Maximum value of output spectrogram before normalization: {output_spectrogram.max()}")
    output_spectrogram = torch.pow(10.0, output_spectrogram / 20.0) # Convert the spectrogram from decibel scale back to amplitude
    print(f"Output spectrogram after power conversion: {output_spectrogram}")
    print(f"Minimum value of output spectrogram before normalization: {output_spectrogram.min()}")
    print(f"Maximum value of output spectrogram before normalization: {output_spectrogram.max()}")
    output_spectrogram = (output_spectrogram - output_spectrogram.min()) / (output_spectrogram.max() - output_spectrogram.min()) # Normalize the spectrogram to the range [0, 1]
    print(f"Output spectrogram after normalization: {output_spectrogram}")

    # Create a GriffinLim object with matching parameters
    print(f"Inference n_fft: {MEL_N_FFT}")
    hop_length = 1023
    win_length = 1023

    class InverseMelScale(T.InverseMelScale): # Invert the Mel scale
        def forward(self, melspec):
            self.fb = self.fb.to(melspec.device)  # Move the Mel filterbank to the same device as the input
            return super().forward(melspec)

    inverse_mel_scale = InverseMelScale(sample_rate=sample_rate, n_stft=MEL_N_FFT, n_mels=N_MELS)
    output_spectrogram = inverse_mel_scale(output_spectrogram)
    print(f"Output spectrogram shape after inverse Mel scale: {output_spectrogram.shape}")

    output_spectrogram = output_spectrogram.cpu() # Move the spectrogram to the CPU before applying GriffinLim
    griffinlim = T.GriffinLim(n_fft=MEL_N_FFT*2-1, hop_length=hop_length, win_length=None) # Create a GriffinLim object

    output_waveform = griffinlim(output_spectrogram.squeeze()) # Convert spectrogram back to audio
    print(f"Output waveform shape: {output_waveform.shape}")

    output_waveform = output_waveform.unsqueeze(0) # Add channel dimension

    torchaudio.save("output_audio.wav", output_waveform, sample_rate)

    print("Done! File saved.")