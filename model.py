import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from xsampa import WickelFeature

# TODO: it'd be nice if we could reuse this class for the one-model-two-cases output
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # TODO: layer architecture

        self.activation = nn.Sigmoid()

    def forward(self, x):
        # TODO
        pass 


net_gen_sg = Net()
net_gen_pl = Net()

loss_criterion = nn.L1Loss()


def run_training(model, mapping_pairs,num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=10.)

    for epoch in range(num_epochs):
        for input_rep,target_rep in mapping_pairs:
            # TODO: this is copied from the problem set, do we actually need it?
            input_rep = input_rep.view(1,-1)
            target_rep = target_rep.view(1, -1)

            optimizer.zero_grad()
            output = model(input_rep)
            loss = loss_criterion(output, target_rep)
            loss.backward()
            optimizer.step()


def wickelfeature_to_tensor(feature: WickelFeature) -> torch.Tensor:
    # TODO
    pass 