import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import plotly.graph_objects as go

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 100
batch_size = 8
learning_rate = 1e-4
num_embeddings = 64
embedding_dim = 16
num_features = 16
num_residual_layers = 2
num_residual_hiddens = 16

# Data preprocessing
def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
    spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128, n_fft=1024, hop_length=256)(waveform)
    spectrogram = T.AmplitudeToDB()(spectrogram)
    return spectrogram.squeeze(0), sample_rate

# Plot example spectrogram
def plot_spectrogram(spectrogram):
    fig = go.Figure(data=go.Heatmap(z=spectrogram.numpy(), colorscale='Viridis'))
    fig.update_layout(title='Mel Spectrogram', xaxis_title='Time', yaxis_title='Mel Frequency')
    fig.show()

# VQ-VAE Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, num_features, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=4, stride=2, padding=1)
        self.residual_stack = nn.Sequential(
            *[ResidualBlock(num_features, num_residual_hiddens) for _ in range(num_residual_layers)]
        )
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.residual_stack(x)
        return x

# VQ-VAE Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self.residual_stack = nn.Sequential(
            *[ResidualBlock(in_channels, num_residual_hiddens) for _ in range(num_residual_layers)]
        )
        self.transpose_conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.transpose_conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        x = self.residual_stack(x)
        x = F.relu(self.transpose_conv1(x))
        x = self.transpose_conv2(x)
        return x

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.conv(x))
        x = x + residual
        return x

# VQ-VAE
class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_features):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(1, num_features, num_residual_layers, num_residual_hiddens)
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.decoder = Decoder(num_features, 1, num_residual_layers, num_residual_hiddens)
        
    def forward(self, x):
        encoded = self.encoder(x)
        flattened = encoded.view(-1, encoded.size(1))
        
        distances = (flattened.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(-1)
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).view(encoded.size())
        
        decoded = self.decoder(quantized)
        return decoded, indices
    
    def encode(self, x):
        encoded = self.encoder(x)
        flattened = encoded.view(-1, encoded.size(1))
        
        distances = (flattened.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(-1)
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).view(encoded.size())
        
        return quantized, indices
    
    def decode(self, quantized):
        return self.decoder(quantized)
    
# Training loop
def train(model, dataloader, optimizer):
    model.train()
    for epoch in range(num_epochs):
        for i, (spectrograms, _) in enumerate(dataloader):
            spectrograms = spectrograms.to(device)
            
            optimizer.zero_grad()
            reconstructed_spectrograms, _ = model(spectrograms)
            loss = F.mse_loss(reconstructed_spectrograms, spectrograms)
            loss.backward()
            
            if (i + 1) % 4 == 0:  # Accumulate gradients over 4 iterations
                optimizer.step()
                optimizer.zero_grad()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Inference
def inference(model, spectrogram):
    with torch.no_grad():
        spectrogram = spectrogram.to(device)
        quantized, _ = model.encode(spectrogram)
        reconstructed_spectrogram = model.decode(quantized)
    return reconstructed_spectrogram

# Create VQ-VAE model
model = VQVAE(num_embeddings, embedding_dim, num_features).to(device)

# Custom collate function
def collate_fn(batch):
    spectrograms, sample_rates = zip(*batch)
    
    # Find the maximum size among the spectrograms in the batch
    max_size = max(spec.size(-1) for spec in spectrograms)
    
    # Pad the spectrograms to the maximum size
    padded_batch = []
    for spec in spectrograms:
        pad_length = max_size - spec.size(-1)
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

# Create dataloader
data_path = "26_29_09_2017_KCL\\26-29_09_2017_KCL"
spectrograms = load_data(data_path)

# Plot an example spectrogram
example_spectrogram, _ = spectrograms[0]
plot_spectrogram(example_spectrogram)

dataloader = torch.utils.data.DataLoader(spectrograms, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Train model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train(model, dataloader, optimizer)

# Perform inference
input_spectrogram, sample_rate = preprocess_audio("ST311-Final-Group-Project-\\26_29_09_2017_KCL\\26-29_09_2017_KCL\\ReadText\\PD\\ID02_pd_2_0_0.wav")
output_spectrogram = inference(model, input_spectrogram)

# Convert spectrogram back to audio
output_waveform = T.GriffinLim()(output_spectrogram.squeeze())
torchaudio.save("output_audio.wav", output_waveform, sample_rate)