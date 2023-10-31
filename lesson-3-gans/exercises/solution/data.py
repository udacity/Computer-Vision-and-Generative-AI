from multiprocessing import cpu_count
from torchvision import datasets, transforms
import torch


def get_dataloader(root_path, image_size, batch_size, workers=cpu_count()):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.CIFAR10(
        root=root_path, download=True, train=True, transform=transform
    )

    # Get indices of samples with label 2 (birds)
    car_indices = [i for i, (_, label) in enumerate(dataset) if label == 1]

    # Create a subset using those indices
    car_subset = torch.utils.data.Subset(dataset, car_indices)

    dataloader = torch.utils.data.DataLoader(
        car_subset, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, persistent_workers=True
    )

    return dataloader
