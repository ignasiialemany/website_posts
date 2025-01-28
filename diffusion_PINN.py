import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the initial condition function

import numpy as np

def f_0(x, mu=0.5, sigma=1.0):
    """Gaussian function."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

def f_0_prime(x, mu=0.5, sigma=1.0):
    """First derivative of the Gaussian function."""
    return -((x - mu) / sigma**2) * f_0(x, mu, sigma)

def f_0_double_prime(x, mu=0.5, sigma=1.0):
    """Second derivative of the Gaussian function."""
    return ((x - mu)**2 - sigma**2) / sigma**4 * f_0(x, mu, sigma)


# Generate collocation points
def get_points(N_ic, N_internal, seed):
    np.random.seed(seed)
    # Initial condition points
    x_ic = np.random.uniform(0, 1, N_ic)
    t_0 = np.zeros_like(x_ic)

    # Internal points
    x_internal = np.random.uniform(0, 1, N_internal)
    t_internal = np.random.uniform(0, 1, N_internal)

    # Combine points
    initial_points = np.stack((x_ic, t_0), axis=1)
    internal_points = np.stack((x_internal, t_internal), axis=1)
    all_points = np.concatenate((internal_points, initial_points), axis=0)

    # Convert to tensors
    all_points = torch.tensor(all_points, dtype=torch.float32)
    
    x_from_all_points = all_points[:, 0]
    sorted_x_points = np.sort(x_from_all_points)
    
    
    
    plt.plot(sorted_x_points)
    plt.show()
    check = f_0(sorted_x_points)
    
    check_true = f_0(np.linspace(0, 1, 100))
    
    plt.plot(np.linspace(0, 1, 100), check_true)
    plt.plot(sorted_x_points, check)
    plt.show()
    
    h = torch.tensor(f_0(all_points[:, 0].numpy()), dtype=torch.float32)
    h_prime = torch.tensor(f_0_prime(all_points[:, 0].numpy()), dtype=torch.float32)
    h_double_prime = torch.tensor(f_0_double_prime(all_points[:, 0].numpy()), dtype=torch.float32)

    return all_points, h, h_prime, h_double_prime

# Define the PINN model
class PINN_diffusion(nn.Module):
    def __init__(self, layers):
        super(PINN_diffusion, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            nn.init.xavier_normal_(self.layers[-1].weight,gain=2.0)
            nn.init.zeros_(self.layers[-1].bias)
        self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        #add a layer that outputs from 0 to 1
        x = torch.sigmoid(x)
        return x

# Compute PDE loss

def compute_loss_pde_noD(model, points, D):
    points.requires_grad_(True)
    u = model(points)
    grads = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    u_t = grads[:, 1]
    return (u_t**2).mean()

def compute_loss_pde(model, points, D):
    points.requires_grad_(True)
    u = model(points)
    grads = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    u_t = grads[:, 1]
    u_x = grads[:, 0]
    u_xx = torch.autograd.grad(u_x.sum(), points, create_graph=True)[0][:, 0]
    pde_residual = u_t - D * u_xx
    return (pde_residual**2).mean()

# Compute initial condition loss

def compute_loss_spatial(model, points, h, h_prime, h_double_prime):
    points.requires_grad_(True)
    u = model(points)
    grads = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    u_x = grads[:, 0]
    u_xx = torch.autograd.grad(u_x.sum(), points, create_graph=True)[0][:, 0]
    
    
    f0_value = (u-h)**2
    f0_prime_value = (u_x-h_prime)**2
    f0_double_prime_value = (u_xx-h_double_prime)**2
    loss_f0 = f0_value.mean() + 0.1 * f0_prime_value.mean() + 0.01 * f0_double_prime_value.mean()
    
    return loss_f0
           

def compute_loss_ic(model, points, h):
    u_pred = model(points)
    return ((u_pred - h)**2).mean()

# Training function
def train(model, points, h, h_prime, h_double_prime, epochs, lr, weight_ic=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    points, h = points.to(device), h.to(device)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,momentum=0.1,alpha=0.9)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    #print lr
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                      mode='min',
                                                      factor=0.5,
                                                      patience=20,
                                                      min_lr=1e-8,
                                                      verbose=True)
    loss_ic_history, loss_pde_history = [], []

    ic_points = points[points[:, 1] == 0]
    internal_points = points[points[:, 1] > 0]
    ic_values = h[points[:, 1] == 0]
    h_prime_values = h_prime[points[:, 1] == 0]
    h_double_prime_values = h_double_prime[points[:, 1] == 0]
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        model.train()

        # Define closure function
        def closure():
            optimizer.zero_grad()
            loss_ic = compute_loss_spatial(model, ic_points, ic_values, h_prime_values, h_double_prime_values)
            loss_pde = compute_loss_pde(model, internal_points, D=0.)
            loss = weight_ic * loss_ic + loss_pde
            loss.backward()
            return loss

        # Perform optimization step
        loss = optimizer.step(closure)
        scheduler.step(loss)
        # Record losses
        loss_ic_value = compute_loss_spatial(model, ic_points, ic_values, h_prime_values, h_double_prime_values).item()
        loss_pde_value = compute_loss_pde(model, internal_points, D=0.).item()
        loss_ic_history.append(loss_ic_value)
        loss_pde_history.append(loss_pde_value)

        pbar.set_description(f"Epoch {epoch}, Loss PDE: {loss_pde_value:.6f}, Loss IC: {loss_ic_value:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        def validate(t_val=0.):
            x_val = torch.linspace(0, 1, 100, device=device)
            t_val = torch.full_like(x_val, t_val)
            val_points = torch.stack((x_val, t_val), dim=1)
            with torch.no_grad():
                return model(val_points).cpu().numpy()
        
        if epoch % 100 == 0:
            u_pred = validate()
            plt.ion()
            plt.figure(figsize=(6, 4))
            plt.plot(np.linspace(0, 1, 100), u_pred, label='Prediction')
            plt.plot(np.linspace(0, 1, 100), f_0(np.linspace(0, 1, 100)), '--', label='Target')
            plt.draw()
            plt.pause(1.0)
            plt.ioff()
            
    return model, loss_ic_history, loss_pde_history

# Main execution
if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Generate points
    N_ic = 1000
    N_internal = 1000
    points, h, h_prime, h_double_prime = get_points(N_ic, N_internal, seed=123)

    # Define model architecture
    layers = [2, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,32, 32, 32, 1]
    model = PINN_diffusion(layers)

    # Train the model
    trained_model, ic_loss_history, pde_loss_history = train(model, points, h, h_prime, h_double_prime, epochs=5000, lr=0.05, device=device)

    # Plot loss history
    plt.plot(ic_loss_history, label='Initial Condition Loss')
    plt.plot(pde_loss_history, label='PDE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot solution at t = 0.5
    x = np.linspace(0, 1, 100)
    t = np.full_like(x, 0.)
    #x_normalized = x / 10.0  # Normalize x to match the training input
    input_tensor = torch.tensor(np.column_stack((x, t)), dtype=torch.float32).to(device)

    # Predict using the trained model
    model.eval()
    with torch.no_grad():
        u_pred = trained_model(input_tensor).cpu().numpy()

    # Plot the predicted solution
    plt.figure(figsize=(8, 5))
    plt.plot(x, u_pred, label='PINN Prediction at t=0.5')
    plt.plot(x, f_0(x), label='True Solution')
    plt.xlabel('x')
    plt.ylabel('u(x, t=0.5)')
    plt.title('1D Diffusion Equation Solution at t=0.5')
    plt.legend()
    plt.grid(True)
    plt.show()

 
