# %%
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from pathlib import Path
import torchaudio
import soundfile as sf
from IPython.display import Audio, display
from tqdm import tqdm
import math
import time

# %%
device = torch.device('mps')
PATCH_SIZE = 1 * 8 * 128  # for per-pixel error
TIME_STEPS = 8
FREQ_BINS = 128

sample_rate = 16000
win_length = int(0.025 * sample_rate)  # 400
hop_length = int(0.010 * sample_rate)  # 160
n_fft = win_length                     # 400 is fine
hidden1 = 256
hidden2 = 512


#%%
class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=13):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv32 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(8, 8),
            stride=(2, 2)
        )
        
        self.conv64 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(1, 8),
            stride=(1, 2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def reparameterize(self, mu, logvar):
        # clamp logvar to reasonable range
        logvar = logvar.clamp(min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar   

    def forward(self, x):
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv64(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        mu     = self.fc_mu(x)
        raw_logvar = self.fc_logvar(x)
        z, logvar = self.reparameterize(mu, raw_logvar)

        return z, mu, logvar
# %%
class Decoder(nn.Module):
    def __init__(self, latent_dim=13, out_channels=1, out_h=8, out_w=128):
        super().__init__()
        self.out_channels = out_channels
        self.out_h = out_h
        self.out_w = out_w

       
        self.fc1 = nn.Linear(latent_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_channels * out_h * out_w)
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.out_channels, self.out_h, self.out_w)
        x = torch.tanh(x)
        return x
# %%
class AudioVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=13):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim,
                               out_channels=in_channels,
                               out_h=8,
                               out_w=128)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# %%
def spectrogram_to_patches(spec, time_steps=8, freq_bins=128):
    """
    spec: (freq_bins_full, time_frames)
    returns: (num_patches, 1, 8, 128) for Conv2d input
    """
    spec = spec[:freq_bins, :]  # (128, T)
    num_frames = spec.size(1)
    patches = []

    for t in range(num_frames - time_steps + 1):
        patch = spec[:, t:t+time_steps]   
        patch = patch.transpose(0, 1)    
        # norm
        patch = torch.log1p(patch)

        # standardize per-patch
        patch_mean = patch.mean()
        patch_std  = patch.std()
        patch = (patch - patch_mean) / (patch_std + 1e-6)

        patches.append(patch)

    if not patches:
        return None

    patches = torch.stack(patches, dim=0)   
    patches = patches.unsqueeze(1)         

    return patches

def sample_random_patch_from_spec(spec, time_steps=TIME_STEPS, freq_bins=FREQ_BINS):
    """
    spec: (F_full, T)  -> we use only first freq_bins, and sample 1 patch (1, 8, 128)
    """
    spec = spec[:freq_bins, :]  
    F, T = spec.shape

    if T < time_steps:
        return None  

    # random start frame
    t0 = torch.randint(0, T - time_steps + 1, (1,)).item()

    patch = spec[:, t0:t0+time_steps]   
    patch = patch.transpose(0, 1)       

    # norm
    patch = torch.log1p(patch)
    patch_mean = patch.mean()
    patch_std  = patch.std()
    patch = (patch - patch_mean) / (patch_std + 1e-6)

    # add channel dim for Conv2d: 
    patch = patch.unsqueeze(0)

    return patch

stft_transform = torchaudio.transforms.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    window_fn=torch.hann_window,
    power=2.0,
    center=True,
    normalized=False
)

class LibriSpeechLikeDataset(Dataset):
    def __init__(self, subsets, root='LibriSpeech', target_sample_rate=16000):
        self.root = Path(root)
        self.subsets = subsets if isinstance(subsets, (list, tuple)) else [subsets]
        self.target_sample_rate = target_sample_rate
        self.entries = []

        for subset in self.subsets:
            subset_dir = self.root / subset
            if not subset_dir.is_dir():
                raise RuntimeError(f"Subset directory not found: {subset_dir}")

            for speaker_dir in subset_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                for chapter_dir in speaker_dir.iterdir():
                    if not chapter_dir.is_dir():
                        continue
                    trans_files = list(chapter_dir.glob("*.trans.txt"))
                    if len(trans_files) != 1:
                        continue
                    trans_file = trans_files[0]
                    with open(trans_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(" ", 1)
                            if len(parts) != 2:
                                continue
                            utt_id, text = parts
                            flac_path = chapter_dir / f"{utt_id}.flac"
                            if flac_path.is_file():
                                self.entries.append((flac_path, text, utt_id))

        if len(self.entries) == 0:
            raise RuntimeError(f"No utterances found in subsets: {self.subsets}")
        print(f"Loaded {len(self.entries)} utterances from {self.subsets}")

    def __len__(self):
        return len(self.entries)

    def _load_audio(self, flac_path):

        # Use soundfile to read FLAC, bypassing torchaudio.load / torchcodec
        data, sr = sf.read(str(flac_path), dtype="float32")  # data: (T,) or (T, C)

      
        if data.ndim == 1:
            
            data = data[None, :]
        else:
           
            data = data.T

        waveform = torch.from_numpy(data) 

        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
            sr = self.target_sample_rate

        # Force mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True) 

        return waveform, sr

    def __getitem__(self, idx):
        flac_path, text, utt_id = self.entries[idx]
        waveform, sr = self._load_audio(flac_path)   

        spec = stft_transform(waveform)[0]          

        patch = sample_random_patch_from_spec(spec) 
        if patch is None:
            # if too short get another index
            return self[(idx + 1) % len(self)]

        return {
            "patch": patch,               
            "sample_rate": sr,
            "utt_id": utt_id,
            "transcript": text,
            "path": str(flac_path),
        }
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # per-element MSE
    recon_mse = F.mse_loss(recon_x, x, reduction="mean")

    # KL (average over batch)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_mse + beta * kl
    return loss, recon_mse, kl

def format_metrics(loss, recon_mse, kl):
    loss_val  = loss
    recon_val = recon_mse
    kl_val    = kl

    if loss_val > 0:
        recon_pct = 100.0 * recon_val / loss_val
        kl_pct    = 100.0 * kl_val    / loss_val
    else:
        recon_pct = kl_pct = 0.0

    # root-mean-square error per element
    recon_rmse = math.sqrt(recon_val)

    return recon_pct, kl_pct, recon_rmse
# %%
def reconstruct_full_utterance(model, flac_path, device=device):
    """
    Reconstruct an entire utterance using the patch-wise VAE.

    Returns:
        sr: sample rate
        orig_wave: original mono waveform (numpy)
        recon_wave: reconstructed waveform (numpy)
    """
    # load original audio 
    data, sr = sf.read(str(flac_path), dtype="float32")  # (T,) or (T, C)
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo -> mono

    orig_wave = data

    # Resample to 16k if needed (to match training)
    if sr != sample_rate:
        wave_t = torch.from_numpy(data).unsqueeze(0)  # (1, T)
        resampler = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=sample_rate)
        wave_t = resampler(wave_t)
        sr = sample_rate
        orig_wave = wave_t.squeeze(0).numpy()
    else:
        wave_t = torch.from_numpy(orig_wave).unsqueeze(0)  # (1, T)

    #  compute spectrogram like in training
    spec = stft_transform(wave_t)[0] 

    F_full, T_frames = spec.shape
    F_use = min(FREQ_BINS, F_full)

    spec_use = spec[:F_use, :]  

    # buffers for overlap-add reconstruction in power domain
    recon_spec_use = torch.zeros_like(spec_use)
    count = torch.zeros_like(spec_use)

    model.eval()
    with torch.no_grad():
        for t0 in range(T_frames - TIME_STEPS + 1):
            # Original patch in power domain: (F_use, 8)
            patch_orig = spec_use[:, t0:t0+TIME_STEPS]

        
            patch = patch_orig.transpose(0, 1)

            # log1p + per-patch normalization 
            patch_log = torch.log1p(patch)
            patch_mean = patch_log.mean()
            patch_std = patch_log.std()
            patch_norm = (patch_log - patch_mean) / (patch_std + 1e-6)

            
            patch_in = patch_norm.unsqueeze(0).unsqueeze(0).to(device)

            # run through VAE
            recon_norm, mu, logvar = model(patch_in)
            recon_norm = recon_norm.squeeze(0).squeeze(0).cpu()  # (8, F_use)

            # denormalize & undo log
            recon_log = recon_norm * (patch_std + 1e-6) + patch_mean
            recon_patch = torch.expm1(recon_log)  # back to power

            
            recon_patch_ft = recon_patch.transpose(0, 1)

            # overlap-add into global spectrogram
            recon_spec_use[:, t0:t0+TIME_STEPS] += recon_patch_ft
            count[:, t0:t0+TIME_STEPS] += 1.0

    # avoid divide-by-zero
    recon_spec_use = recon_spec_use / count.clamp(min=1.0)

    # pad back to full FFT bins (n_fft//2 + 1)
    full_F = n_fft // 2 + 1
    recon_spec_full = torch.zeros(full_F, T_frames)
    recon_spec_full[:F_use, :] = recon_spec_use

    # convert power -> magnitude for Griffin-Lim
    recon_mag = recon_spec_full.clamp(min=0).sqrt()

    griffin = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        power=1.0,
        n_iter=32
    )

    recon_wave_t = griffin(recon_mag)  # (T_audio,)
    recon_wave = recon_wave_t.cpu().numpy()

    return sr, orig_wave, recon_wave

def play_original_and_full_recon(model, full_dataset, idx=0, device=device):
    """
    Pick utterance `idx` from `full_dataset`,
    play ORIGINAL then RECONSTRUCTED audio one after another.
    """
    flac_path, text, utt_id = full_dataset.entries[idx]
    print(f"Utterance idx={idx}, utt_id={utt_id}")
    print(f"Transcript: {text}")
    print(f"File: {flac_path}")

    sr, orig_wave, recon_wave = reconstruct_full_utterance(
        model, flac_path, device=device
    )

    print("\n--- ORIGINAL ---")
    display(Audio(orig_wave, rate=sr))
    time.sleep(0.5)

    print("\n--- RECONSTRUCTED (VAE) ---")
    display(Audio(recon_wave, rate=sr))

#%%
# def main():
#     full_dataset = LibriSpeechLikeDataset(
#         subsets=["train-clean-100"],
#         root="LibriSpeech",
#         target_sample_rate=16000
#     )

#     train_size = int(0.8 * len(full_dataset))
#     val_size   = len(full_dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, persistent_workers=True, num_workers=4)
#     val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, persistent_workers=True, num_workers=4)

#     model = AudioVAE(in_channels=1, latent_dim=13).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

   
#     num_epochs = 10

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         train_recon = 0.0
#         train_kl = 0.0

#         batch = next(iter(train_loader))
       

#         for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):

#             x = batch["patch"].to(device)         # (B, 1, 8, 128)
           
#             optimizer.zero_grad()
#             with torch.autocast(device_type="mps", dtype=torch.float16):
#                 recon_x, mu, logvar = model(x)
#                 beta = min(1.0, epoch / (num_epochs / 2.0))
#                 loss, recon_loss, kl = vae_loss(recon_x, x, mu, logvar, beta=beta)

#             # recon_x, mu, logvar = model(x)
#             # loss, recon_loss, kl = vae_loss(recon_x, x, mu, logvar, beta=1.0)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

#             optimizer.step()

#             train_loss  += loss.item()   * x.size(0)
#             train_recon += recon_loss.item() * x.size(0)
#             train_kl    += kl.item()    * x.size(0)
            

#         n_train = len(train_loader.dataset)
       

#         # ---- validation ----
#         model.eval()
#         val_loss = 0.0
#         val_recon = 0.0
#         val_kl = 0.0

#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc="Validation", leave=False):
#                 x = batch["patch"].to(device)
#                 recon_x, mu, logvar = model(x)
#                 loss, recon_loss, kl = vae_loss(recon_x, x, mu, logvar, beta=1.0)
#                 val_loss  += loss.item()   * x.size(0)
#                 val_recon += recon_loss.item() * x.size(0)
#                 val_kl    += kl.item()    * x.size(0)
       
#         n_val   = len(val_loader.dataset)

#         # convert sums → means
#         train_loss  /= n_train
#         train_recon /= n_train
#         train_kl    /= n_train

#         val_loss  /= n_val
#         val_recon /= n_val
#         val_kl    /= n_val

#         tr_recon_pct, tr_kl_pct, tr_rmse = format_metrics(train_loss, train_recon, train_kl)
#         va_recon_pct, va_kl_pct, va_rmse = format_metrics(val_loss, val_recon, val_kl)

        

#         print(
#             f"Epoch {epoch+1}/{num_epochs} | "
#             f"Train Loss: {train_loss:.4f} "
#             f"(Recon MSE {train_recon:.4f}, RMSE {tr_rmse:.4f}, "
#             f"{tr_recon_pct:.1f}% ; KL {train_kl:.4f}, {tr_kl_pct:.1f}%)\n"
#             f"           | Val   Loss: {val_loss:.4f} "
#             f"(Recon MSE {val_recon:.4f}, RMSE {va_rmse:.4f}, "
#             f"{va_recon_pct:.1f}% ; KL {val_kl:.4f}, {va_kl_pct:.1f}%)"
#         )
#         x = batch["patch"].to(device)

#         with torch.no_grad():
#             recon_x, mu, logvar = model(x)
#             mse_model = F.mse_loss(recon_x, x, reduction="mean").item()
#             mse_zero  = F.mse_loss(torch.zeros_like(x), x, reduction="mean").item()

#         #print(f"Model MSE: {mse_model:.4f}, Zero-baseline MSE: {mse_zero:.4f}")
    
#     model.eval()
#     batch = next(iter(train_loader))
#     x = batch["patch"]           # (B, 1, 8, 128)

#     B = x.shape[0]
#     input_per_example = x[0].numel()  # 1*8*128 = 1024

#     with torch.no_grad():
#         z, mu, logvar = model.encoder(x.to(device))

#     latent_per_example = z[0].numel()  # should be 13

#     print(f"Input shape: {x.shape}, values per example: {input_per_example}")
#     print(f"Latent shape: {z.shape}, values per example: {latent_per_example}")

#     compression = input_per_example / latent_per_example
#     print(f"Compression ratio: {compression:.2f}x")
#     print("\nNow doing full-utterance A/B test...")
#     play_original_and_full_recon(model, full_dataset, idx=0, device=device)
def train_one_vae(latent_dim, train_loader, val_loader, num_epochs=10, device=device):
    model = AudioVAE(in_channels=1, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    history = []

    for epoch in range(num_epochs):
        # ---------- TRAIN ----------
        model.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        n_train = len(train_loader.dataset)

        for batch in tqdm(train_loader, desc=f"[z={latent_dim}] Train Epoch {epoch+1}/{num_epochs}"):
            x = batch["patch"].to(device)  # (B, 1, 8, 128)

            optimizer.zero_grad()

            # forward
            with torch.autocast(device_type="mps", dtype=torch.float16):
                recon_x, mu, logvar = model(x)

                # 1) loss used for optimization: warmup beta
                beta = min(1.0, epoch / (num_epochs / 2.0))
                loss_opt, recon_loss_opt, kl_opt = vae_loss(recon_x, x, mu, logvar, beta=beta)

                # 2) loss used for logging: fixed beta=1.0 so train/val comparable
                loss_log, recon_log, kl_log = vae_loss(recon_x, x, mu, logvar, beta=1.0)

            loss_opt.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # accumulate *logging* metrics
            bs = x.size(0)
            train_loss_sum  += loss_log.item()  * bs
            train_recon_sum += recon_log.item() * bs
            train_kl_sum    += kl_log.item()    * bs

        train_loss  = train_loss_sum  / n_train
        train_recon = train_recon_sum / n_train
        train_kl    = train_kl_sum    / n_train

        # VALIDATION 
        model.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0
        n_val = len(val_loader.dataset)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[z={latent_dim}] Validation", leave=False):
                x = batch["patch"].to(device)
                recon_x, mu, logvar = model(x)
                loss, recon_loss, kl = vae_loss(recon_x, x, mu, logvar, beta=1.0)

                bs = x.size(0)
                val_loss_sum  += loss.item()       * bs
                val_recon_sum += recon_loss.item() * bs
                val_kl_sum    += kl.item()         * bs

        val_loss  = val_loss_sum  / n_val
        val_recon = val_recon_sum / n_val
        val_kl    = val_kl_sum    / n_val

        # metrics
        tr_recon_pct, tr_kl_pct, tr_rmse = format_metrics(train_loss, train_recon, train_kl)
        va_recon_pct, va_kl_pct, va_rmse = format_metrics(val_loss, val_recon, val_kl)

        print(
            f"z={latent_dim} | Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} "
            f"(Recon MSE {train_recon:.4f}, RMSE {tr_rmse:.4f}, {tr_recon_pct:.1f}% ; "
            f"KL {train_kl:.4f}, {tr_kl_pct:.1f}%)\n"
            f"           | Val   Loss: {val_loss:.4f} "
            f"(Recon MSE {val_recon:.4f}, RMSE {va_rmse:.4f}, {va_recon_pct:.1f}% ; "
            f"KL {val_kl:.4f}, {va_kl_pct:.1f}%)"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_recon_mse": train_recon,
            "train_kl": train_kl,
            "val_loss": val_loss,
            "val_recon_mse": val_recon,
            "val_kl": val_kl,
        })

    return model, history

def main():
    full_dataset = LibriSpeechLikeDataset(
        subsets=["train-clean-100"],
        root="LibriSpeech",
        target_sample_rate=16000
    )

    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    latent_dims = [13, 32, 64]
    num_epochs = 10

    results = {}

    for ld in latent_dims:
        print("\n" + "="*60)
        print(f"Training VAE with latent_dim = {ld}")
        print("="*60 + "\n")

        model, history = train_one_vae(
            latent_dim=ld,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            device=device
        )

        torch.save(model.state_dict(), f"latent_dim_{ld}")

        best_val_recon = min(h["val_recon_mse"] for h in history)
        final_val_recon = history[-1]["val_recon_mse"]
        print(f"\n[z={ld}] Best val recon MSE: {best_val_recon:.4f}, "
              f"final val recon MSE: {final_val_recon:.4f}")

        results[ld] = {
            "model": model,
            "history": history,
        }

        print(f"\n[z={ld}] Full-utterance A/B listening test:")
        play_original_and_full_recon(model, full_dataset, idx=0, device=device)

    print("\n===== SUMMARY ACROSS LATENT DIMS =====")
    for ld in latent_dims:
        best_val_recon = min(h["val_recon_mse"] for h in results[ld]["history"])
        print(f"z={ld}: best val recon MSE = {best_val_recon:.4f}")

if __name__ =="__main__":
    main()
    