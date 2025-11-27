import torch
import torch.nn as nn
import numpy as np

SAMPLE_RATE = 8000
FRAME_SIZE = 512
LATENT_DIM = 32
NUM_EMBEDDINGS = 512 # Dictionary of sound features
COMMITMENT_COST = 0.25
decay = 0.99

# --- 1. Define the Original 1D Model (To load your weights) ---
class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock1D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1)
        )
    def forward(self, x): return x + self.block(x)

class AudioVQVAE(nn.Module):
    def __init__(self):
        super(AudioVQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock1D(32),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock1D(64),
            nn.Conv1d(64, 32, kernel_size=4, stride=2, padding=1), # Latent dim 32
        )
        
# Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(LATENT_DIM, 64, kernel_size=4, stride=2, padding=1),
            ResidualBlock1D(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            ResidualBlock1D(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Output -1 to 1 (Mu-Law range)
        )
# --- 2. Define the "Shadow" 2D Model (For Export) ---
# We treat audio as an image with Height = 1
class ResidualBlock2D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2D, self).__init__()
        # Kernel: (1, 3) means height 1, width 3
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(1,1))
        )
    def forward(self, x): return x + self.block(x)

class Encoder2D(nn.Module):
    def __init__(self):
        super(Encoder2D, self).__init__()
        self.net = nn.Sequential(
            # Stride (1, 2) reduces width by half, keeps height 1
            nn.Conv2d(1, 32, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            ResidualBlock2D(32),
            nn.Conv2d(32, 64, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            ResidualBlock2D(64),
            nn.Conv2d(64, 32, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
        )
    def forward(self, x): return self.net(x)

class Decoder2D(nn.Module):
    def __init__(self):
        super(Decoder2D, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            ResidualBlock2D(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            ResidualBlock2D(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.Tanh()
        )
    def forward(self, x): return self.net(x)

# --- 3. Transfer Weights Function ---
def transfer_weights(layer_1d, layer_2d):
    # Copy Weight: [Out, In, K] -> [Out, In, 1, K]
    with torch.no_grad():
        if isinstance(layer_1d, nn.Conv1d):
            layer_2d.weight.copy_(layer_1d.weight.unsqueeze(2))
            if layer_1d.bias is not None:
                layer_2d.bias.copy_(layer_1d.bias)
        elif isinstance(layer_1d, nn.ConvTranspose1d):
            layer_2d.weight.copy_(layer_1d.weight.unsqueeze(2))
            if layer_1d.bias is not None:
                layer_2d.bias.copy_(layer_1d.bias)

def recursive_transfer(module_1d, module_2d):
    # Matches layers by order and transfers weights
    layers_1d = [m for m in module_1d.modules() if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d))]
    layers_2d = [m for m in module_2d.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    
    for l1, l2 in zip(layers_1d, layers_2d):
        transfer_weights(l1, l2)
        print(f"Transferred {l1} -> {l2}")

# --- 4. Main Execution ---
# A. Load Original
device = torch.device('cpu')
orig_model = AudioVQVAE().to(device)
state_dict = torch.load('vqvae_weights.pth', map_location=device)
orig_model.load_state_dict(state_dict, strict=False) # strict=False to ignore decoder keys if needed

# B. Create 2D Shadows
encoder_2d = Encoder2D()
decoder_2d = Decoder2D()

# C. Transfer Weights
print("--- Transferring Encoder ---")
recursive_transfer(orig_model.encoder, encoder_2d)
print("--- Transferring Decoder ---")
# Re-create original decoder structure to grab weights easily
orig_decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=4, stride=2, padding=1),
            ResidualBlock1D(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            ResidualBlock1D(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
orig_decoder.load_state_dict(orig_model.decoder.state_dict())
recursive_transfer(orig_decoder, decoder_2d)

# D. Export to ONNX (Opset 13 is safer for 2D)
# Encoder Input: [1, 1, 1, 512] (Batch, Channel, Height, Width)
dummy_in = torch.randn(1, 1, 1, 512)
torch.onnx.export(encoder_2d, dummy_in, "encoder_fixed.onnx", 
                  input_names=['input'], output_names=['latent'], opset_version=12)

# Decoder Input: [1, 32, 1, 64]
dummy_latent = torch.randn(1, 32, 1, 64)
torch.onnx.export(decoder_2d, dummy_latent, "decoder_fixed.onnx", 
                  input_names=['latent'], output_names=['audio'], opset_version=12)

print("\nSUCCESS: Generated 'encoder_fixed.onnx' and 'decoder_fixed.onnx'")