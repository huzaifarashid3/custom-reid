import torch
from model import ft_net
from torchvision import datasets
from torch.autograd import Variable

from PIL import Image
import torchvision.transforms as transforms

# Define image transformations (same as used in training)
transform = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),  # Standard Market-1501 image size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
])


# image_path = "./Market/pytorch/query/0001/0001_c1s1_001051_00.jpg"
# image_path = "Market/pytorch/query/0024/0024_c1s1_002326_00.jpg"
# image = Image.open(image_path)
# image = transform(image)
# image = Variable(image)
# image = image.unsqueeze(0)  # Add batch dimension



dataset = datasets.ImageFolder('./Market/pytorch/query', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

image, label = next(iter(dataloader))
print(label.item())


model = ft_net()
model.load_state_dict(torch.load('model.pth', weights_only=True))

model.eval()


# for image, label in dataloader:
#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output.data, 1)
#     # write to a file
#     with open('output.txt', 'a') as f:
#         pred = predicted.item()
#         leb = label.item()
#         same = pred == leb
#         f.write(f'Predicted: {pred}, Label: {leb}, Same: {same}\n')



# output = model(image)
# _, predicted = torch.max(output.data, 1)

with torch.no_grad():
    outputs = model(image)
    print(outputs[0][0])
    _, preds = torch.max(outputs.data, 1)
print(preds.item(), label.item())

# print("Predicted Class: ", predicted_class)