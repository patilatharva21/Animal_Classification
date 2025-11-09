import os
import torch
import torchvision
import torchvision.transforms as transforms

os.listdir("C:\\Users\\atharva patil\\Documents\\Projects2.0\\classification\\animal_data\\train_data")


training_dataset_path ="C:\\Users\\atharva patil\\Documents\\Projects2.0\\classification\\animal_data\\train_data"
training_transform = transforms.Compose([transforms.Resize((244,244)), transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root = training_dataset_path, transform =training_transform)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=32, shuffle=False)


def get_mean_and_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        mean += images.mean([0, 2, 3]) * image_count_in_a_batch
        std += images.std([0, 2, 3]) * image_count_in_a_batch
        total_images_count += image_count_in_a_batch

    mean /= total_images_count
    std /= total_images_count

    return mean, std


mean, std = get_mean_and_std(train_loader)

print(mean, std)




