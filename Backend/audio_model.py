import librosa
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Preprocessing Functions ---
# Based on RAVDESSFinalAudioPipeline.ipynb, cells 2lQrtB0UKwKa & fJR2J3gKK0T2

def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y = y_trimmed / (np.max(np.abs(y_trimmed)) + 1e-8)
    return y

def highpass_filter(y, sr, cutoff=100):
    sos = scipy.signal.butter(10, cutoff, 'hp', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, y)

def pad_or_truncate(y, target_len=48000):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    else:
        return y[:target_len]

def extract_features_temporal(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features = np.vstack([
        mfcc, mfcc_delta, mfcc_delta2, mel_db,
        zcr, rmse, spectral_centroid, spectral_rolloff
    ])

    # Pad/truncate time dimension to match the training shape (94)
    max_time = 94
    if features.shape[1] < max_time:
        pad_width = max_time - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_time]

    return features

def preprocess_for_model(file_path, sr=16000, target_len=48000):
    y = load_audio(file_path, sr=sr)
    y = highpass_filter(y, sr=sr)
    y = pad_or_truncate(y, target_len=target_len)
    feats = extract_features_temporal(y, sr=sr) # (188, 94)
    return feats

def features_to_tensor(feats):
    arr = np.asarray(feats, dtype=np.float32)
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0) # (1, 1, 188, 94)
    return tensor

# --- 2. Model Architecture ---
# Based on RAVDESSFinalAudioPipeline.ipynb, cell JLOd9jPXK8LW

class CNN_GRU_Attention(nn.Module):
    def __init__(self, num_classes=7): # Defaulted to 7
        super(CNN_GRU_Attention, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.2)
        )
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.attn = nn.Linear(256, 1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, 1, features, time)
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.mean(dim=3)
        gru_out, _ = self.gru(x)
        attn_weights = F.softmax(self.attn(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        out = self.fc(context)
        return out