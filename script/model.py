import torch
from script.generate import generate_dataset

class Prior_Theta:
    def sample(self, size):
        return torch.randn(2*size)*2
    
class P_Theta:
    def sample(self, theta):
        size = theta.size(0)
        return torch.randn(size)*2 + theta
    
class MNIST_Sampler:

    def sample(self, size):
        return generate_dataset(size)