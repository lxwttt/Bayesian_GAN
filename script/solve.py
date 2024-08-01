import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(n):
    u = np.zeros((n, n))
    return u

def apply_boundary_conditions(u):
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    return u

def solve_heat_equation(n, kappa, b, max_iter=10000, tol=1e-6):
    u = initialize_grid(n)
    u = apply_boundary_conditions(u)
    
    # Heat source
    heat_source = np.ones((n, n)) * b
    
    # Time step
    dt = 0.1
    dx = 1.0
    dy = 1.0
    
    for _ in range(max_iter):
        u_new = u.copy()
        for i in range(1, n-1):
            for j in range(1, n-1):
                u_new[i, j] = u[i, j] + kappa * dt * (
                    (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 +
                    (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
                ) + dt * heat_source[i, j]
        
        # Check for convergence
        if np.linalg.norm(u_new - u, ord=2) < tol:
            break
        
        u = u_new
    
    return u

# Parameters
n = 28
kappa = 1.0
b = 10**3

# Solve the heat equation
u = solve_heat_equation(n, kappa, b)

# Plotting the result
plt.imshow(u, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Heat Equation Solution')
plt.show()
