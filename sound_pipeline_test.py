import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

#N_MELS = 256
#MEL_N_FFT = 4096
#HOP_LENGTH = MEL_N_FFT // 8

N_MELS = 96
MEL_N_FFT = 4096*2
HOP_LENGTH = 4096//40

def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
    waveform = waveform.to(torch.float32)  # Cast waveform to float32
    waveform_std = torch.std(waveform)

    spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=N_MELS, n_fft=MEL_N_FFT, hop_length=HOP_LENGTH)(waveform)
    spectrogram = T.AmplitudeToDB()(spectrogram)
    #spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()  # Normalize spectrogram

    return spectrogram.squeeze(0), sample_rate, waveform_std

def spectrogram_to_waveform(spectrogram, sample_rate):
    print("Converting spectrogram back to a waveform...")

    spectrogram = torch.pow(10.0, spectrogram / 20.0)  # Convert the spectrogram from decibel scale back to amplitude
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())  # Normalize the spectrogram to the range [0, 1]

    # Create a GriffinLim object with matching parameters
    n_fft = MEL_N_FFT
    hop_length = HOP_LENGTH
    win_length = MEL_N_FFT
    window = torch.hann_window(win_length)  # Use Hann window

    class InverseMelScale(T.InverseMelScale):  # Invert the Mel scale
        def forward(self, melspec):
            self.fb = self.fb.to(melspec.device)  # Move the Mel filterbank to the same device as the input
            return super().forward(melspec)

    inverse_mel_scale = InverseMelScale(sample_rate=sample_rate, n_stft=n_fft // 2 + 1, n_mels=N_MELS)
    spectrogram = inverse_mel_scale(spectrogram)

    spectrogram = spectrogram.unsqueeze(0)  # Add a batch dimension
    griffinlim = T.GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=1, n_iter=10)  # Create a GriffinLim object

    waveform = griffinlim(spectrogram)  # Convert spectrogram back to audio

    print(spectrogram.shape)

    return waveform

if __name__ == "__main__":
    audio_path = "26_29_09_2017_KCL\\26-29_09_2017_KCL\\ReadText\\PD\\ID30_pd_2_1_1.wav"

    spectrogram, sample_rate, waveform_std = preprocess_audio(audio_path)
    waveform = spectrogram_to_waveform(spectrogram, sample_rate)
    output_waveform_std = torch.std(waveform)
    waveform *= waveform_std / output_waveform_std  # Scale the waveform back to its original amplitude

    torchaudio.save("sound_pipeline_test_audio.wav", waveform, sample_rate)
    print("Done! File saved.")