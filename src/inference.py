import torch
from PIL import Image
from torchvision.transforms import transforms
import torchvision.models as models

# Set the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print('Using MPS')
else:
    print("MPS device not found.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model architecture (explicitly specifying weights)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Updated weights parameter
num_classes = 2  # Adjust for your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Move the model to the specified device
model.to(device)

# Load the model weights
model.load_state_dict(torch.load('/Users/seamus/universe/harvard/REspIrezE/inhaler_2_train/models/inhaler_2_model.v01.pth', weights_only=True, map_location=device))
model.to(device)

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the image
image_path = '/Users/seamus/universe/harvard/REspIrezE/inhaler_2_train/test_images/mdi/imageds.jpeg'
image = Image.open(image_path)

# Apply the transformations
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Move the tensor to the appropriate device
input_tensor = input_tensor.to(device)

# Perform inference
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    output = model(input_tensor)

# Interpret the output
probabilities = torch.nn.functional.softmax(output[0], dim=0)
predicted_class = torch.argmax(probabilities).item()

# Reverse the class_to_idx mapping to get idx_to_class
class_to_idx = {'ellipta': 0, 'mdi': 1}  # Define your class-to-index mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}
predicted_class_name = idx_to_class[predicted_class]

# Print the result
print(f"This inhaler is a {predicted_class_name.upper()}.")