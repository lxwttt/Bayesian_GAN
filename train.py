import torch
import numpy as np
from script.model import MNIST_Sampler

def train_bayesian_gan(sampler,
                #  prior_theta, 
                #  P_theta,
                 discriminator,
                 generator,
                 table_size=2, epochs=1000, epochs_critic=5):
    reference_table = torch.tensor([])

    # theta = prior_theta.sample(table_size)
    # X = P_theta.sample(theta)
    dataset = sampler.sample(table_size)
    X = dataset[:,:,2,:]
    theta = dataset[:,:,0,:]

    reference_table = torch.stack((X, theta), 2).reshape(table_size, 28, 28, 2)

    discriminator.train()
    generator.train()
    for epoch in range(epochs):
        for epoch_critic in range(epochs_critic):
            discriminator.update(reference_table, generator)
        generator.update(reference_table, discriminator)
        loss = generator.loss_function(reference_table, discriminator)
        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch}/{epochs-1} loss: {loss}')

    return

if __name__ == '__main__':
    from script.gan import Generator, Discriminator
    from script.model import Prior_Theta, P_Theta

    # prior_theta = Prior_Theta()
    # P_theta = P_Theta()
    sampler = MNIST_Sampler()
    discriminator = Discriminator()
    generator = Generator()

    # train_bayesian_gan(prior_theta, P_theta, discriminator, generator)
    train_bayesian_gan(sampler, discriminator, generator)

    # Save the models
    torch.save(discriminator.state_dict(), 'weights/discriminator.pt')
    torch.save(generator.state_dict(), 'weights/generator.pt')