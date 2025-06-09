import torch
from torchvision import transforms
from PIL import Image

# Ask for image path
path = input("Enter path of the image: ")

# Load model
model = torch.load("mosaic.pth")
model.eval()

# Load and process image
img = Image.open(path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])
input_tensor = preprocess(img).unsqueeze(0)

# Apply style
with torch.no_grad():
    output = model(input_tensor)

# Save and show result
result = transforms.ToPILImage()(output[0].clamp(0, 255) / 255)
result.save("stylized.jpg")
result.show()
