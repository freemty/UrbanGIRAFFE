
import torch 
import torch.nn as nn
import torch.nn.functional as F 


class BaseLoss(nn.Module):
    def __init__(self):
        pass 

    def forward(self):
        pass 

class GANLoss(nn.Module):
    def __init__(self):
        pass

    # def r1_penalty(self, d_real, x_real):
    #     with conv2d_gradfix.no_weight_gradients():
    #         r1_grads = torch.autograd.grad(outputs=d_real.sum(), inputs=x_real, create_graph=True, only_inputs=True)
    #     r1_grads_image = r1_grads[0]
    #     r1_penalty = r1_grads_image.square().sum([1,2,3]) / 2
    #     return r1_penalty
    
    def forward(self):
        pass