import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

N_MELS = 128
MEL_N_FFT = 4096
HOP_LENGTH = 4096

def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
    waveform = waveform.to(torch.float32)  # Cast waveform to float32
    spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=N_MELS, n_fft=MEL_N_FFT, hop_length=HOP_LENGTH)(waveform)
    spectrogram = T.AmplitudeToDB()(spectrogram)
    spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()  # Normalize spectrogram
    return spectrogram.squeeze(0), sample_rate

def spectrogram_to_waveform(spectrogram, sample_rate):
    spectrogram = torch.pow(10.0, spectrogram / 20.0)  # Convert the spectrogram from decibel scale back to amplitude
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())  # Normalize the spectrogram to the range [0, 1]

    # Create a GriffinLim object with matching parameters
    hop_length = HOP_LENGTH - 1
    win_length = HOP_LENGTH - 1

    class InverseMelScale(T.InverseMelScale):  # Invert the Mel scale
        def forward(self, melspec):
            self.fb = self.fb.to(melspec.device)  # Move the Mel filterbank to the same device as the input
            return super().forward(melspec)

    inverse_mel_scale = InverseMelScale(sample_rate=sample_rate, n_stft=MEL_N_FFT, n_mels=N_MELS)
    spectrogram = inverse_mel_scale(spectrogram)

    griffinlim = T.GriffinLim(n_fft=MEL_N_FFT*2-1, hop_length=HOP_LENGTH, win_length=None, power=1)  # Create a GriffinLim object

    waveform = griffinlim(spectrogram.squeeze())  # Convert spectrogram back to audio
    waveform = waveform.unsqueeze(0)  # Add channel dimension

    return waveform

if __name__ == "__main__":
    audio_path = "26_29_09_2017_KCL\\26-29_09_2017_KCL\\ReadText\\PD\\ID30_pd_2_1_1.wav"
    spectrogram, sample_rate = preprocess_audio(audio_path)
    waveform = spectrogram_to_waveform(spectrogram, sample_rate)
    torchaudio.save("test_audio.wav", waveform, sample_rate)
    print("Done! File saved.")