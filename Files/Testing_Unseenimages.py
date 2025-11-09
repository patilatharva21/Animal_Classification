import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# classes used
classes = ["Buffalo", "Elephant", "Rhino", "Zebra"]

# loading the trained model 
def load_model(model_path='best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(classes)) 

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!\n")
    return model, device


# use same mean/std as used during training
mean = [0.5200, 0.4986, 0.4165]
std = [0.2550, 0.2467, 0.2490]

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
])


def classify(model, device, image_path, classes):
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(" Error: Image file not found.")
        return
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # Apply transforms and move to device
    image = image_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        confidence = probs[0][predicted.item()].item() * 100

    print(f"🦓 Predicted Class: {classes[predicted.item()]} ({confidence:.2f}% confidence)")


model_path = "/content/best_model.pth"  # Upload this to Colab
image_path = "/content/rhi3.jpeg"  # Upload any test image

model, device = load_model(model_path)
classify(model, device, image_path, classes)
