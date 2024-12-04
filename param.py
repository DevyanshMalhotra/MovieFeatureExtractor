import torch
from timm import create_model

# Initialize the EfficientNetB0 model
model = create_model('efficientnet_b0', pretrained=True)

# Count the number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Count the number of layers with trainable parameters
trainable_layers = sum(1 for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {trainable_params}")
print(f"Number of layers with trainable parameters: {trainable_layers}")
