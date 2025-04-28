import torch
from models.rgb2point import RGB2Point

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your model
model = RGB2Point().to(device)
model.load_state_dict(torch.load('outputs/best_model.pth'))
model.eval()

# Create a dummy RGB input
dummy_input = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    output = model(dummy_input)

print(f"[TEST] Model output shape: {output.shape}")
print(f"[TEST] Some output values: {output.view(-1, 3)[:5]}")
