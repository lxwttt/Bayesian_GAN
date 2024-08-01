import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, lr=0.0002, device='cpu'):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(4, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 2)
        # )
        self.noise_dim = 28*28
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output values between -1 and 1
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def generate_noise(self, batch_size):
        return torch.randn(batch_size)

    def loss_function(self, reference_table, discriminator):
        pred = self.forward(reference_table[:,:,:,0]).reshape(-1, 28, 28)
        loss_fake = discriminator.forward(torch.stack((reference_table[:,:,:,0], pred), 3))
        loss_real = discriminator.forward(reference_table)
        penalty = 0
        loss = torch.abs(torch.sum(loss_fake) - torch.sum(loss_real))
        return loss

    def update(self, reference_table, discriminator):
        self.optimizer.zero_grad()
        loss = self.loss_function(reference_table, discriminator)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def forward(self, x):
        batch_size = x.size(0)
        noise = self.generate_noise(batch_size*self.noise_dim).reshape(batch_size, 28, 28)
        input = torch.stack((noise, x), 3)
        return self.model(input.permute(0, 3, 1, 2))
    
class Discriminator(nn.Module):
    def __init__(self, lr=0.0002, device='cpu'):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(4, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid()
        # )
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),  # 4x4 is the spatial dimension after 3 conv layers with stride 2
            nn.Sigmoid()  # Output probability
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device


    def loss_function(self, reference_table, generator):
        theta_fake = generator.forward(reference_table[:,:,:,0]).reshape(-1, 28, 28)
        theta_real = reference_table[:,:,:,1]
        loss_fake = self.forward(torch.stack((reference_table[:,:,:,0], theta_fake), 3))
        loss_real = self.forward(torch.stack((reference_table[:,:,:,0], theta_real), 3))
        loss = -torch.abs(torch.sum(loss_fake) - torch.sum(loss_real))
        return loss

    def update(self, reference_table, generator):
        loss = self.loss_function(reference_table, generator)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def forward(self, input):
        return self.model(input.permute(0, 3, 1, 2))
