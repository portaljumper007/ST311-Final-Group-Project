import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import plotly.graph_objects as go
import numpy as np

N_MELS = 96
MEL_N_FFT = 4096
FIXED_SPECT_LENGTH = 4096+2048+1024
HOP_LENGTH = 4096//16

DTYPE = torch.float32

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING DEVICE: {device}")

# Set default data type to float32 or bfloat32
torch.set_default_dtype(DTYPE)

# Data preprocessing
def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
    waveform = waveform.to(torch.float32)  # Cast waveform to float32
    torch.set_default_dtype(torch.float32)
    spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=N_MELS, n_fft=MEL_N_FFT, hop_length=HOP_LENGTH)(waveform)
    spectrogram = T.AmplitudeToDB()(spectrogram)
    spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()  # Normalize spectrogram
    spectrogram = spectrogram.to(DTYPE)
    torch.set_default_dtype(DTYPE)
    return spectrogram.squeeze(0), sample_rate

def mel_frequencies(n_mels, sample_rate):
    """Compute the Mel frequencies for a given number of Mel bins."""
    min_mel = 0
    max_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_freqs = np.linspace(min_mel, max_mel, n_mels)
    mel_freqs = 700 * (10 ** (mel_freqs / 2595) - 1)
    return mel_freqs

import plotly.subplots as subplots
def plot_spectrogram(input_spectrogram, output_spectrogram):
    input_spectrogram = input_spectrogram.to(torch.float32)
    output_spectrogram = output_spectrogram.to(torch.float32)

    # Pad the shorter spectrogram to match the length of the longer one
    if input_spectrogram.size(-1) < output_spectrogram.size(-1):
        pad_length = output_spectrogram.size(-1) - input_spectrogram.size(-1)
        input_spectrogram = F.pad(input_spectrogram, (0, pad_length), mode='constant', value=0)
    elif output_spectrogram.size(-1) < input_spectrogram.size(-1):
        pad_length = input_spectrogram.size(-1) - output_spectrogram.size(-1)
        output_spectrogram = F.pad(output_spectrogram, (0, pad_length), mode='constant', value=0)

    diff_spectrogram = output_spectrogram - input_spectrogram

    # Compute the mel frequencies
    mel_freqs = mel_frequencies(N_MELS, sample_rate)

    # Create subplots
    fig = subplots.make_subplots(rows=3, cols=1, vertical_spacing=0.1, subplot_titles=('Input Spectrogram', 'Output Spectrogram', 'Difference Spectrogram'))

    # Input spectrogram
    fig.add_trace(go.Heatmap(
        z=input_spectrogram.cpu().numpy(),
        colorscale='Hot',
        y=mel_freqs,  # Use mel frequencies for y-axis
        zmin=input_spectrogram.min().item(),
        zmax=input_spectrogram.max().item(),
        colorbar=dict(title='Amplitude (dB)'),
    ), row=1, col=1)

    # Output spectrogram
    fig.add_trace(go.Heatmap(
        z=output_spectrogram.cpu().numpy(),
        colorscale='Hot',
        y=mel_freqs,  # Use mel frequencies for y-axis
        zmin=output_spectrogram.min().item(),
        zmax=output_spectrogram.max().item(),
        colorbar=dict(title='Amplitude (dB)'),
    ), row=2, col=1)

    # Difference spectrogram
    fig.add_trace(go.Heatmap(
        z=diff_spectrogram.cpu().numpy(),
        colorscale='RdBu',
        y=mel_freqs,  # Use mel frequencies for y-axis
        zmin=diff_spectrogram.min().item(),
        zmax=diff_spectrogram.max().item(),
        colorbar=dict(title='Amplitude Difference (dB)'),
    ), row=3, col=1)

    fig.update_layout(
        title='Mel Spectrograms',
        xaxis_title='Time chunks of length ' + str(MEL_N_FFT),
        yaxis_title='Mel Frequency (Hz)',
        height=1200,
        width=2400,
        coloraxis_colorbar_y=0.2,  # Move colorbars to the right side
    )

    fig.show()





# ARCHITECTURE

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x = x + residual
        x = self.activation(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=5, stride=2, padding=2)
        self.res_block1 = ResidualBlock(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=5, stride=2, padding=2, groups=hidden_channels*1)
        self.res_block2 = ResidualBlock(hidden_channels*2)
        self.conv3 = nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=5, stride=2, padding=2, groups=hidden_channels*2)
        self.res_block3 = ResidualBlock(hidden_channels*4)
        self.conv4 = nn.Conv2d(hidden_channels*4, hidden_channels*8, kernel_size=5, stride=2, padding=2)
        self.res_block4 = ResidualBlock(hidden_channels*8)
        self.conv5 = nn.Conv2d(hidden_channels*8, hidden_channels*16, kernel_size=5, stride=2, padding=2)
        self.res_block5 = ResidualBlock(hidden_channels*16)
        self.flat_dim = (N_MELS // 32) * (FIXED_SPECT_LENGTH // 32) * hidden_channels*16
        self.fc1 = nn.Linear(self.flat_dim, hidden_channels*32)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_channels*32, hidden_channels*16)
        self.dropout2 = nn.Dropout(0.1)
        self.fc_mu = nn.Linear(hidden_channels*16, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels*16, latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    @torch.autocast(device_type="cuda", dtype=DTYPE)
    def forward(self, x):
        residual = x
        x = self.leaky_relu(self.conv1(x))
        x = self.res_block1(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.res_block2(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.res_block3(x)
        x = self.leaky_relu(self.conv4(x))
        x = self.res_block4(x)
        x = self.leaky_relu(self.conv5(x))
        x = self.res_block5(x)
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var, residual

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_channels, output_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_channels*16)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_channels*16, hidden_channels*32)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(hidden_channels*32, (N_MELS // 32) * (FIXED_SPECT_LENGTH // 32) * hidden_channels*16)
        self.dropout3 = nn.Dropout(0.1)
        self.deconv1 = nn.ConvTranspose2d(hidden_channels*16, hidden_channels*16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.res_block1 = ResidualBlock(hidden_channels*16)
        self.pointwise_deconv1 = nn.ConvTranspose2d(hidden_channels*16, hidden_channels*8, kernel_size=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_channels*8, hidden_channels*8, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.res_block2 = ResidualBlock(hidden_channels*8)
        self.deconv3 = nn.ConvTranspose2d(hidden_channels*8, hidden_channels*4, kernel_size=5, stride=2, padding=2, output_padding=1, groups=hidden_channels*4)
        self.res_block3 = ResidualBlock(hidden_channels*4)
        self.pointwise_deconv3 = nn.ConvTranspose2d(hidden_channels*4, hidden_channels*4, kernel_size=1)
        self.deconv4 = nn.ConvTranspose2d(hidden_channels*4, hidden_channels*2, kernel_size=5, stride=2, padding=2, output_padding=1, groups=hidden_channels*2)
        self.res_block4 = ResidualBlock(hidden_channels*2)
        self.pointwise_deconv4 = nn.ConvTranspose2d(hidden_channels*2, hidden_channels, kernel_size=1)
        self.deconv5 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.res_block5 = ResidualBlock(hidden_channels)
        self.pointwise_deconv5 = nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=1)
        self.residual_conv = nn.Conv2d(output_channels, output_channels, kernel_size=1, groups=output_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

    @torch.autocast(device_type="cuda", dtype=DTYPE)
    def forward(self, z, residual):
        x = self.leaky_relu(self.fc1(z))
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout3(x)
        x = x.view(x.size(0), -1, N_MELS // 32, FIXED_SPECT_LENGTH // 32)
        x = self.leaky_relu(self.deconv1(x))
        x = self.res_block1(x)
        x = self.leaky_relu(self.pointwise_deconv1(x))
        x = self.leaky_relu(self.deconv2(x))
        x = self.res_block2(x)
        x = self.leaky_relu(self.deconv3(x))
        x = self.res_block3(x)
        x = self.leaky_relu(self.pointwise_deconv3(x))
        x = self.leaky_relu(self.deconv4(x))
        x = self.res_block4(x)
        x = self.leaky_relu(self.pointwise_deconv4(x))
        x = self.leaky_relu(self.deconv5(x))
        x = self.res_block5(x)
        x = torch.sigmoid(self.pointwise_deconv5(x))
        
        # Incorporate the residual connection
        residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=False)
        residual_weight = torch.sigmoid(self.residual_conv(x))
        x = residual_weight * residual / x
        
        return x

# VAE
class VAE(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, hidden_channels, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_channels, input_channels)

    def forward(self, x):
        mu, log_var, residual = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z, residual)
        return x_recon, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

# Training loop
def train(model, dataloader, optimizer, scheduler=None):
    model.train()
    for conv_layer in [model.encoder.conv1, model.encoder.conv2, model.encoder.conv3, model.encoder.conv4, model.encoder.conv5]:
        nn.init.kaiming_normal_(conv_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (noisy_spectrograms, clean_spectrograms, _) in enumerate(dataloader):
            noisy_spectrograms = noisy_spectrograms.to(device).to(DTYPE)  # Convert input to float16
            clean_spectrograms = clean_spectrograms.to(device).to(DTYPE)  # Convert input to float16

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=DTYPE):
                reconstructed_spectrograms, mu, log_var = model(noisy_spectrograms)
                reconstruction_loss = F.l1_loss(reconstructed_spectrograms, clean_spectrograms)
                kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                beta = 0.1  # Beta-VAE loss coefficient
                loss = reconstruction_loss + beta * kl_divergence  # Combine losses with weighting factor

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
        if scheduler:
            scheduler.step(epoch_loss / len(dataloader))

# Inference
def inference(model, spectrogram):
    with torch.no_grad():
        spectrogram = spectrogram.to(device)
        
        # Pad the spectrogram to match the fixed_size used during training
        pad_length = FIXED_SPECT_LENGTH - spectrogram.size(-1)
        padded_spectrogram = F.pad(spectrogram, (0, pad_length), mode='constant', value=0)
        
        padded_spectrogram = padded_spectrogram.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        reconstructed_spectrogram, _, _ = model(padded_spectrogram)
        
        # Remove padding from the reconstructed spectrogram
        reconstructed_spectrogram = reconstructed_spectrogram.squeeze(0).squeeze(0)[:, :spectrogram.size(-1)]
        reconstructed_spectrogram = reconstructed_spectrogram.to(torch.float32)  # Cast back to float32
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
    
    return stacked_noisy_spectrograms.unsqueeze(1), stacked_clean_spectrograms.unsqueeze(1), sample_rates[0]

# Load and preprocess data
def load_data(data_path, noise_ratio=0.6):
    spectrograms = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav") and "ID30_pd_2_1_1.wav" not in file:
                print(file)
                audio_path = os.path.join(root, file)
                spectrogram, sample_rate = preprocess_audio(audio_path)
                
                # Add noise to the spectrogram
                noise = torch.randn_like(spectrogram) * noise_ratio * torch.var(spectrogram)
                noisy_spectrogram = spectrogram + noise
                
                spectrograms.append((noisy_spectrogram, spectrogram, sample_rate))
    return spectrograms

def spectrogram_to_waveform(spectrogram, sample_rate):
    spectrogram = spectrogram.to(torch.float32)  # Cast back to float32
    spectrogram = torch.pow(10.0, spectrogram / 20.0)  # Convert the spectrogram from decibel scale back to amplitude
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())  # Normalize the spectrogram to the range [0, 1]

    # Create a GriffinLim object with matching parameters
    hop_length = HOP_LENGTH - 1
    win_length = HOP_LENGTH - 1

    class InverseMelScale(T.InverseMelScale):  # Invert the Mel scale
        def forward(self, melspec):
            self.fb = self.fb.to(melspec.device)  # Move the Mel filterbank to the same device as the input
            return super().forward(melspec)

    inverse_mel_scale = InverseMelScale(sample_rate=sample_rate, n_stft=MEL_N_FFT // 2 + 1, n_mels=N_MELS)
    spectrogram = inverse_mel_scale(spectrogram)

    spectrogram = spectrogram.cpu()  # Move the spectrogram to the CPU before applying GriffinLim
    griffinlim = T.GriffinLim(n_fft=MEL_N_FFT*2-1, hop_length=HOP_LENGTH, win_length=None, power=1, n_iter=100)  # Create a GriffinLim object

    waveform = griffinlim(spectrogram.squeeze())  # Convert spectrogram back to audio
    waveform = waveform.unsqueeze(0)  # Add channel dimension

    return waveform

if __name__ == "__main__":
    print("Starting...")

    # Hyperparameters
    num_epochs = 120
    batch_size = 4096
    learning_rate = 4e-4
    latent_dim = 4096
    hidden_channels = 6

    # Create VAE model
    input_channels = 1
    model = VAE(input_channels, hidden_channels, latent_dim).to(device)

    # Create dataloader
    data_path = "26_29_09_2017_KCL\\26-29_09_2017_KCL"
    spectrograms = load_data(data_path)
    print(f"Number of spectrograms: {len(spectrograms)}")

    # Plot an example spectrogram
    #_, example_spectrogram, _ = spectrograms[0]
    #plot_spectrogram(example_spectrogram.cpu())
    input_spectrogram, sample_rate = preprocess_audio("26_29_09_2017_KCL\\26-29_09_2017_KCL\\ReadText\\PD\\ID30_pd_2_1_1.wav")

    dataloader = torch.utils.data.DataLoader(spectrograms, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True if torch.cuda.is_available() else False)

    # Train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5, amsgrad=True)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    train(model, dataloader, optimizer)#, scheduler)

    # Perform inference
    output_spectrogram = inference(model, input_spectrogram)
    plot_spectrogram(input_spectrogram.cpu(), output_spectrogram.cpu()) # Plot the input, output, and difference spectrograms

    output_waveform = spectrogram_to_waveform(output_spectrogram, sample_rate)
    torchaudio.save("output_audio.wav", output_waveform, sample_rate)

    input_waveform = spectrogram_to_waveform(input_spectrogram, sample_rate)
    torchaudio.save("test_unaltered_input_pipeline_audio.wav", input_waveform, sample_rate)

    print("Done! File saved.")