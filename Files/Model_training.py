import torchvision
import torch
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


train_data = r'C:\Users\atharva patil\Documents\Projects2.0\classification\animal_data\train_data'
test_data = r'C:\Users\atharva patil\Documents\Projects2.0\classification\animal_data\test_data'

# define normalization parameters
mean = [0.5200, 0.4986, 0.4165]
std = [0.2550, 0.2467, 0.2490]

# transforms metrics
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# load datasets
train_dataset = torchvision.datasets.ImageFolder(root=train_data, transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=test_data, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


def set_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epoch):
    device = set_device()
    model.to(device)
    best_acc = 0

    for epoch in range(n_epoch):
        print(f"Epoch {epoch + 1}/{n_epoch}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_acc = 100 * correct / total
        print(f"  Training Accuracy: {epoch_acc:.2f}%, Loss: {running_loss / len(train_loader):.4f}")

        test_acc = evaluate_model_on_test_set(model, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, epoch, optimizer, best_acc)
            print("  Checkpoint saved!")

    print("Training complete.")
    return model


def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    device = set_device()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"  Testing Accuracy: {accuracy:.2f}%")
    return accuracy


def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_accuracy': best_acc
    }
    torch.save(state, 'model_best_checkpoint.pth.tar')


def load_checkpoint(model_path, model):
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        print("Checkpoint loaded!")
    else:
        print("No checkpoint found, training from scratch.")


# Initialize model
resnet18_model = models.resnet18(weights=None)
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 4
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)

# Load checkpoint if available
load_checkpoint('model_best_checkpoint.pth.tar', resnet18_model)

# Define loss function and optimizer
device = set_device()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

# Train model
trained_model = train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, 40)

# Save final trained model
torch.save(trained_model.state_dict(), 'best_model.pth')
print("Final model saved.")
