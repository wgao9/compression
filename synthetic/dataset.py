import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os

def get(batch_size, model_raw, size, input_dims, output_dims, input_std=1.0, noise_std=0.0, **kwargs):
    input = input_std*torch.randn(size, input_dims).cuda()
    noise = noise_std*torch.randn(size, output_dims).cuda()
    target = model_raw.forward(input) + noise
    input = input.cpu()
    target = target.cpu().detach()

    synthetic_dataset = torch.utils.data.TensorDataset(input, target)

    print("Building synthetic data loader with 1 workers")
    loader = torch.utils.data.DataLoader(
            dataset=synthetic_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            )
    return loader

