import torch.nn as nn
from collections import OrderedDict
from utee import misc
print = misc.logger.info

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, output_dims, dropout):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(dropout)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, output_dims)

        self.model= nn.Sequential(layers)
        print(self.model)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)

def synthetic(input_dims, n_hiddens, n_class, dropout, pretrained=False):
    model = MLP(input_dims, n_hiddens, n_class, dropout)
    return model
