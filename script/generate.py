import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def scale_mnist_images(image):
    # Scale pixel intensity values to be between 1 and 10
    scaled_image = image * 0.9 + 0.1
    return scaled_image

def solve_heat_conduction(image, b, size=28):
    scaled_image = scale_mnist_images(image.squeeze().numpy())
    kappa = torch.tensor(scaled_image, dtype=torch.float32)
    # Define the number of grid points
    n = size - 2
    
    # Create a grid of points
    h = 1.0 / (size - 1)
    
    # Initialize the coefficient matrix A and the right-hand side vector B
    A = torch.zeros((n*n, n*n), dtype=torch.float32)
    B = torch.full((n*n,), -b * h * h, dtype=torch.float32)
    
    # Fill the coefficient matrix A for the interior points
    for i_ in range(0, n):
        for j_ in range(0, n):
            i, j = i_ + 1, j_ + 1
            idx = i_*n + j_
            A[idx, idx] = - (kappa[i+1, j] + kappa[i-1, j]) / (2 * h * h)
            if i_ != 0:
                A[idx, idx-n] = kappa[i-1, j] / (2 * h * h)
            if i_ != n-1:
                A[idx, idx+n] = kappa[i+1, j] / (2 * h * h)
            if j_ != 0:
                A[idx, idx-1] = kappa[i, j-1] / (2 * h * h)
            if j_ != n-1:
                A[idx, idx+1] = kappa[i, j+1] / (2 * h * h)

    A.to_dense()
            
    u_ = torch.linalg.solve(A, B)
    res = A @ u_ - B
    res = res.reshape((n, n))
    u = torch.zeros((size, size), dtype=torch.float32)
    u[1:-1, 1:-1] = u_.reshape((n, n))
    u_noised = u + 1 * torch.randn_like(u)
    # print(f'Residual: {torch.norm(res)}')
    # print(kappa[25:28,6:9])
    # print(u[25:28,6:9])
    # print(res[25,6])
    # print(A[25*n+6,25*n+5:25*n+8],A[25*n+6,24*n+6],A[25*n+6,26*n+6])
    # print(res.shape)
    return torch.stack((kappa, u, u_noised),dim=2)

def generate_dataset(n_samples, b=1e3, size=28):
    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataset = []
    for i in range(n_samples):
        image, _ = mnist_dataset[i]
        dataset.append(solve_heat_conduction(image, b, size))
        # kappa, temperature_field, temperature_field_noised = solve_heat_conduction(image, b, size)
        # dataset.append([kappa, temperature_field, temperature_field_noised])
    dataset = torch.stack(dataset, dim=3)
    print(dataset.shape)
    return dataset


if __name__ == '__main__':
    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Select a single image for demonstration purposes
    image, _ = mnist_dataset[0]

    # Define the heat source term b(s)
    b = 1e3

    # Solve the steady-state heat conduction equation
    kappa, temperature_field, temperature_field_noised = solve_heat_conduction(image, b)

    # Visualize the conductivity field and the resulting temperature field
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Conductivity Field (Scaled MNIST Image)")
    plt.imshow(kappa, cmap='hot')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Temperature Field")
    plt.imshow(temperature_field.detach().numpy(), cmap='hot')
    plt.colorbar()

    plt.show()
