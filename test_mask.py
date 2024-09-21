import torch
import matplotlib.pyplot as plt

# Sample configuration for testing
class Config:
    num_hidden_layers = 4
    num_diffusion_layers = 2
    num_patches = 2

class Model:
    def __init__(self, config):
        self.config = config
        self.layer_idx = 3  # Example layer index

    def transfuser_mask(self, b, h, q_idx, kv_idx):
        # Causal mask for the normal layers
        if self.layer_idx <= self.config.num_hidden_layers - self.config.num_diffusion_layers:
            return q_idx >= kv_idx
        # Bidirectional mask for the diffusion layers
        else:
            q_type = type_tensor[b, q_idx]
            first_image_index = (type_tensor[b] != 0).nonzero(as_tuple=True)[0][0].item() if (type_tensor[b] != 0).any() else None
            if q_type != 0:
                return q_idx >= first_image_index + ((kv_idx - first_image_index) // self.config.num_patches) * self.config.num_patches
            else:
                return q_idx >= kv_idx

# Create sample data
batch_size = 1
seq_length = 10
num_heads = 2

# Sample type_tensor with some image and text tokens
type_tensor = torch.tensor([[0, 0, 2, 2, 1, 1, 1, 1, 2, 2]], dtype=torch.uint8)

# Create a model instance
model = Model(Config())

# Prepare q_idx and kv_idx for visualization
q_idx = torch.arange(seq_length).unsqueeze(0)  # Shape: (1, seq_length)
kv_idx = torch.arange(seq_length).unsqueeze(0)  # Shape: (1, seq_length)

# Generate the mask
mask = torch.zeros((seq_length, seq_length), dtype=torch.bool)
for b in range(batch_size):
    for k in range(seq_length):
        for q in range(seq_length):
            mask[q, k] = model.transfuser_mask(b, 0, q, k)

# Plot the mask
plt.imshow(mask.numpy(), cmap='gray', interpolation='nearest')
plt.colorbar(label='Mask Value (True/False)')
plt.title('Transfuser Mask Visualization')
plt.xlabel('kv_idx')
plt.ylabel('q_idx')
plt.xticks(range(seq_length))
plt.yticks(range(seq_length))
plt.savefig('mask_visualization.png')
