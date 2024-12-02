import torch
from torchvision import transforms
from PIL import Image

# Load the trained model (replace 'model.pth' with your model's path)
model = torch.load('model.pth')
model.eval()

# Define transformations using h and w from your training code
h, w = 256, 128
transform = transforms.Compose([
    transforms.Resize((h, w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = 'path_to_image.jpg'  # replace with your image path
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)  # create a mini-batch as expected by the model

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_tensor = input_tensor.to(device)

# Perform inference
with torch.no_grad():
    outputs = model(input_tensor)
    _, preds = torch.max(outputs, 1)

# Get class names (replace with your actual classes)
class_names = image_datasets['train'].classes  # ensure image_datasets is accessible
predicted_class = class_names[preds[0]]

print('Predicted class:', predicted_class)