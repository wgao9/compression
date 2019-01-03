import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os

def get(batch_size, renew,name, model=None, size=0, input_dims=100, n_hidden='[50,20]', output_dims=10, input_std=1.0, noise_std=0.0, **kwargs):
    if renew == False:
        #load data
        filename = "synthetic_"+name+"_data.pth"
        pathname = "sub_models/synthetic"
        pathname += "_"+str(input_dims)
        for dims in eval(n_hidden):
            pathname += "_"+str(dims)
        pathname += "_"+str(output_dims)
        pathname += "/data"
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        filepath = os.path.join(pathname, filename)
        with open(filepath, "rb") as f:
            data = torch.load(f)
            input = data['input']
            target = data['target']
    else:
        input = input_std*torch.randn(size, input_dims).cuda()
        noise = noise_std*torch.randn(size, output_dims).cuda()
        target = model.forward(input) + noise
        input = input.cpu()
        target = target.cpu().detach()

        
        #save data
        filename = "synthetic_"+name+"_data.pth"
        pathname = "sub_models/synthetic"
        pathname += "_"+str(input_dims)
        for dims in eval(n_hidden):
            pathname += "_"+str(dims)
        pathname += "_"+str(output_dims)
        pathname += "/data"
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        filepath = os.path.join(pathname, filename)
        with open(filepath, "wb") as f:
            torch.save({
                'input': input,
                'target': target,
                }, f)

    synthetic_dataset = torch.utils.data.TensorDataset(input, target)

    print("Building synthetic data loader with 1 workers")
    loader = torch.utils.data.DataLoader(
            dataset=synthetic_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            )
    return loader

