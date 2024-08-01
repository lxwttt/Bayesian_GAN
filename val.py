import torch
from scipy.stats import kstest

def evaluate_bayesian_gan(true_theta,
                     P_theta,
                     discriminator,
                     generator,
                     table_size=1000):
    reference_table = []
    for j in range(table_size):
        theta_j = true_theta.sample()
        X_j = P_theta.sample(theta_j)
        reference_table.append((X_j, theta_j))

    reference_table = torch.tensor(reference_table)
        
    discriminator.eval()
    generator.eval()
    
    loss = generator.loss_function(reference_table, discriminator)
    generator.update(reference_table, discriminator)

    return