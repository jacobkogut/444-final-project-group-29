import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from tqdm.notebook import trange, tqdm
from datasets import load_dataset, Audio
import IPython.display as ipd

SAMPLE_RATE = 8000
FRAME_SIZE = 512
LATENT_DIM = 32
NUM_EMBEDDINGS = 512 # Dictionary of sound features
COMMITMENT_COST = 0.25
decay = 0.99

class ResidualBlock(nn.Module): # ResNet (CMSIS eq.: arm_add_f32)
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.block(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # The Codebook: A dictionary of 512 distinct "sound vectors"
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # inputs: [Batch, Channels (Latent), Time] -> Permute to [Batch, Time, Latent]
        inputs = inputs.permute(0, 2, 1).contiguous()

        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances between input and codebook
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding: Find the nearest codebook index per input
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Quantize: Replace input with the nearest codebook vector
        quantized = self._embedding(encoding_indices).view(input_shape)

        # Loss computation
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        # Permute back to [Batch, Channels, Time]
        return loss, quantized.permute(0, 2, 1).contiguous(), encoding_indices

class AudioVQVAE(nn.Module):
    def __init__(self):
        super(AudioVQVAE, self).__init__()

        # Encoder
        # Input: 1 x 512
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1), # -> 32 x 256
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1), # -> 64 x 128
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv1d(64, LATENT_DIM, kernel_size=4, stride=2, padding=1), # -> 32 x 64
        )

        # The Vector Quantizer Layer
        self.vq = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM, COMMITMENT_COST)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(LATENT_DIM, 64, kernel_size=4, stride=2, padding=1),
            ResidualBlock(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            ResidualBlock(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Output -1 to 1 (Mu-Law range)
        )

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, indices

    # Helper to decode from indices (Simulates Receiver Side)
    def decode_from_indices(self, indices):
        # indices shape: [Batch, Time_Steps]
        # Map indices to vectors
        quantized = self.vq._embedding(indices)
        # Reshape for decoder: [Batch, Time, Latent] -> [Batch, Latent, Time]
        quantized = quantized.permute(0, 2, 1)
        return self.decoder(quantized)
    
# 1. Load trained model
device = torch.device('cpu') # Export on CPU
model = AudioVQVAE().to(device)
state_dict = torch.load('vqvae_weights.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()

codebook_weights = model.vq._embedding.weight.detach().numpy() 

def export_codebook_to_c(weights, filename="codebook.h"):
    with open(filename, 'w') as f:
        f.write("#ifndef CODEBOOK_H\n#define CODEBOOK_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const float CODEBOOK[{weights.shape[0] * weights.shape[1]}] = {{\n")
        
        flat = weights.flatten()
        for i, val in enumerate(flat):
            f.write(f"{val:.8f}f, ")
            if (i+1) % 32 == 0: f.write("\n")
            
        f.write("};\n\n")
        f.write(f"#define CODEBOOK_ENTRIES {weights.shape[0]}\n")
        f.write(f"#define CODEBOOK_DIM {weights.shape[1]}\n")
        f.write("#endif\n")

export_codebook_to_c(codebook_weights)
print("Saved codebook.h")

dummy_input = torch.randn(1, 1, 512)

# 1. Export Encoder
encoder = model.encoder
torch.onnx.export(encoder,
                  dummy_input, 
                  f="encoder.onnx", 
                  input_names=['input_audio'], 
                  output_names=['latent_z'],
                  opset_version=18)

# 2. Export Decoder
dummy_latent = torch.randn(1, 32, 64) 
decoder = model.decoder
torch.onnx.export(decoder, 
                  dummy_input, 
                  f = "decoder.onnx", 
                  input_names=['quantized_z'], 
                  output_names=['reconstructed_audio'],
                  opset_version=18)

print("Saved encoder.onnx and decoder.onnx")