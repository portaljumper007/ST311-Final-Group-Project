import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import plotly.graph_objects as go
import numpy as np

N_MELS = 64
MEL_N_FFT = 4096
FIXED_SPECT_LENGTH = 8192


# Data preprocessing
def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
    spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=N_MELS, n_fft=MEL_N_FFT, hop_length=1024)(waveform)
    spectrogram = T.AmplitudeToDB()(spectrogram)
    return spectrogram.squeeze(0), sample_rate

# Plot example spectrogram
def plot_spectrogram(spectrogram):
    fig = go.Figure(data=go.Heatmap(z=spectrogram.numpy(), colorscale='Viridis'))
    fig.update_layout(title='Mel Spectrogram', xaxis_title='Time', yaxis_title='Mel Frequency')
    fig.show()

# VAE Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

# VAE Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        x_recon = self.fc3(z)
        return x_recon

# VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
    def forward(self, x):
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
        for i, (spectrograms, _) in enumerate(dataloader):
            spectrograms = spectrograms.to(device)
            spectrograms_flat = spectrograms.view(spectrograms.size(0), -1)
            
            optimizer.zero_grad()
            reconstructed_spectrograms, mu, log_var = model(spectrograms_flat)
            reconstruction_loss = F.mse_loss(reconstructed_spectrograms, spectrograms_flat)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + kl_divergence
            loss.backward()
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Inference
def inference(model, spectrogram):
    with torch.no_grad():
        print("Original spectrogram shape:", spectrogram.shape)
        spectrogram = spectrogram.to(device)
        print("Spectrogram shape after moving to device:", spectrogram.shape)
        
        # Pad the spectrogram to match the fixed_size used during training
        pad_length = FIXED_SPECT_LENGTH - spectrogram.size(-1)
        padded_spectrogram = F.pad(spectrogram, (0, pad_length), mode='constant', value=0)
        
        padded_spectrogram = padded_spectrogram.unsqueeze(0)  # Add batch dimension
        print("Padded spectrogram shape:", padded_spectrogram.shape)
        
        padded_spectrogram_flat = padded_spectrogram.reshape(padded_spectrogram.size(0), -1)
        print("Flattened padded spectrogram shape:", padded_spectrogram_flat.shape)
        
        reconstructed_spectrogram_flat, _, _ = model(padded_spectrogram_flat)
        print("Reconstructed flattened spectrogram shape:", reconstructed_spectrogram_flat.shape)
        
        reconstructed_spectrogram = reconstructed_spectrogram_flat.reshape(padded_spectrogram.size(0), padded_spectrogram.size(1), padded_spectrogram.size(2))
        print("Reconstructed spectrogram shape:", reconstructed_spectrogram.shape)
        
        # Remove the padding from the reconstructed spectrogram
        reconstructed_spectrogram = reconstructed_spectrogram[:, :, :spectrogram.size(-1)]
        print("Final reconstructed spectrogram shape:", reconstructed_spectrogram.shape)
        
    return reconstructed_spectrogram.squeeze(0)  # Remove batch dimension

# Custom collate function
def collate_fn(batch):
    spectrograms, sample_rates = zip(*batch)
    
    # Pad the spectrograms to a fixed size
    padded_batch = []
    for spec in spectrograms:
        pad_length = FIXED_SPECT_LENGTH - spec.size(-1)
        padded_spec = F.pad(spec, (0, pad_length), mode='constant', value=0)
        padded_batch.append(padded_spec.unsqueeze(0))
    
    # Stack the padded spectrograms into a single tensor
    stacked_spectrograms = torch.stack(padded_batch)
    
    return stacked_spectrograms, sample_rates[0]  # Assume all audio files have the same sample rate

# Load and preprocess data
def load_data(data_path):
    spectrograms = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                audio_path = os.path.join(root, file)
                spectrogram, sample_rate = preprocess_audio(audio_path)
                spectrograms.append((spectrogram, sample_rate))
    return spectrograms

if __name__ == "__main__":
    print("Starting...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Hyperparameters
    num_epochs = 60
    batch_size = 20000
    learning_rate = 1e-7
    latent_dim = 128
    hidden_dim = 256

    # Create VAE model
    input_dim = N_MELS * FIXED_SPECT_LENGTH  # Adjust based on the fixed spectrogram size
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)

    # Create dataloader
    data_path = "26_29_09_2017_KCL\\26-29_09_2017_KCL"
    spectrograms = load_data(data_path)

    # Plot an example spectrogram
    for i in range(0):
        example_spectrogram, _ = spectrograms[i]
        plot_spectrogram(example_spectrogram)

    dataloader = torch.utils.data.DataLoader(spectrograms, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True if torch.cuda.is_available() else False)






    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, dataloader, optimizer)





    # Perform inference
    input_spectrogram, sample_rate = preprocess_audio("26_29_09_2017_KCL\\26-29_09_2017_KCL\\ReadText\\PD\\ID02_pd_2_0_0.wav")
    print(np.shape(input_spectrogram))
    output_spectrogram = inference(model, input_spectrogram)
    print(output_spectrogram)

    print("Minimum value of spectrogram before normalization:", output_spectrogram.min())
    print("Maximum value of spectrogram before normalization:", output_spectrogram.max())
    output_spectrogram = torch.pow(10.0, output_spectrogram / 20.0) # Convert the spectrogram from decibel scale back to amplitude
    print(output_spectrogram)
    print("Minimum value of spectrogram before normalization:", output_spectrogram.min())
    print("Maximum value of spectrogram before normalization:", output_spectrogram.max())
    #output_spectrogram = (output_spectrogram - output_spectrogram.min()) / (output_spectrogram.max() - output_spectrogram.min()) # Normalize the spectrogram to the range [0, 1]
    print(output_spectrogram)

    # Create a GriffinLim object with matching parameters
    print("inference n_fft:", MEL_N_FFT)
    hop_length = 1023
    win_length = 1023

    class InverseMelScale(T.InverseMelScale): # Invert the Mel scale
        def forward(self, melspec):
            self.fb = self.fb.to(melspec.device)  # Move the Mel filterbank to the same device as the input
            return super().forward(melspec)

    inverse_mel_scale = InverseMelScale(sample_rate=sample_rate, n_stft=MEL_N_FFT, n_mels=N_MELS)
    output_spectrogram = inverse_mel_scale(output_spectrogram)
    print(np.shape(output_spectrogram))

    output_spectrogram = output_spectrogram.cpu() # Move the spectrogram to the CPU before applying GriffinLim
    griffinlim = T.GriffinLim(n_fft=MEL_N_FFT*2-1, hop_length=hop_length, win_length=None) # Create a GriffinLim object

    output_waveform = griffinlim(output_spectrogram.squeeze()) # Convert spectrogram back to audio
    print(output_waveform)

    output_waveform = output_waveform.unsqueeze(0) # Move the spectrogram to the CPU before applying GriffinLim

    torchaudio.save("output_audio.wav", output_waveform, sample_rate)

    print("Done! File saved.")