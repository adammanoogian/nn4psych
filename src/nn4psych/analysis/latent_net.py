"""
Vendored from engellab/latentcircuit (Langdon & Engel 2025).
Modified: removed connectivity.py dependency, optimized for GPU (pre-allocated
forward pass, cached masks/buffers, removed per-step tensor allocations).
"""

import torch
import torch.nn as nn


class LatentNet(torch.nn.Module):
    '''
    Pytorch module for implementing a latent circuit model.
    '''
    def __init__(self, n, N, n_trials, sigma_rec=.15, input_size=6, output_size=2, device='cpu'):
        super(LatentNet, self).__init__()
        self.alpha = .2
        self.n = n
        self.N = N
        self.n_trials = n_trials
        self.input_size = input_size
        self.output_size = output_size
        self.activation = torch.nn.ReLU()
        self.device = device

        # sigma_rec as buffer so it moves with .to(device)
        self.register_buffer('sigma_rec', torch.tensor(sigma_rec))

        # Identity matrix cached as buffer (used in cayley_transform)
        self.register_buffer('_eye_N', torch.eye(self.N))

        # Initialize connectivity layers
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=False)
        self.recurrent_layer.weight.data.normal_(mean=0., std=0.025)
        self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
        self.input_layer.weight.data.normal_(mean=0.2, std=.1)
        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)
        self.output_layer.weight.data.normal_(mean=.2, std=0.1)

        # Pre-compute connectivity masks as buffers (created once, moved with .to())
        input_mask = torch.zeros(self.n, self.input_size)
        input_mask[:self.input_size, :self.input_size] = torch.eye(self.input_size)
        self.register_buffer('_input_mask', input_mask)

        output_mask = torch.zeros(self.output_size, self.n)
        output_mask[-self.output_size:, -self.output_size:] = torch.eye(self.output_size)
        self.register_buffer('_output_mask', output_mask)

        # Initialize embedding matrix, q
        self.a = torch.nn.Parameter(torch.rand(self.N, self.N), requires_grad=True)
        self.q = self.cayley_transform(self.a)

        # Apply connectivity masks to initialized connectivity
        self.connectivity_masks()

    def connectivity_masks(self):
        # Apply pre-computed masks (no tensor allocation per call)
        self.input_layer.weight.data = self._input_mask * torch.relu(self.input_layer.weight.data)
        self.output_layer.weight.data = self._output_mask * torch.relu(self.output_layer.weight.data)

    def cayley_transform(self, a):
        # Transform square matrix a into orthonormal matrix q.
        # Uses cached _eye_N and linalg.solve instead of inverse for stability.
        skew = (a - a.t()) / 2
        o = (self._eye_N - skew) @ torch.linalg.solve(self._eye_N + skew, self._eye_N)
        return o[:self.n, :]

    def forward(self, u):
        t = u.shape[1]
        batch_size = u.shape[0]

        # Noise for all timesteps (one allocation)
        noise = (torch.sqrt(torch.tensor(2.0 * self.alpha, device=u.device)) * self.sigma_rec
                 * torch.randn(batch_size, t, self.n, device=u.device))

        # Collect states in a list, stack once at end (autograd-safe, one final alloc)
        h = torch.zeros(batch_size, self.n, device=u.device)
        state_list = [h]
        for i in range(t - 1):
            h = (1 - self.alpha) * h + self.alpha * (
                self.activation(
                    self.recurrent_layer(h) + self.input_layer(u[:, i, :]) + noise[:, i, :]))
            state_list.append(h)
        return torch.stack(state_list, dim=1)  # (batch, t, n)

    def loss_function(self, x, z, y, l_y):
        return self.mse_z(x, z) + l_y * self.nmse_y(y, x)

    def mse_z(self, x, z):
        return torch.sum((self.output_layer(x) - z) ** 2) / x.shape[0] / x.shape[1]

    def nmse_x(self, y, x):
        mse = nn.MSELoss(reduction='mean')
        y_bar = y - torch.mean(y, dim=[0, 1], keepdim=True)
        return mse(y @ self.q.t(), x) / mse(y_bar, torch.zeros_like(y_bar))

    def nmse_q(self, y):
        mse = nn.MSELoss(reduction='mean')
        y_bar = y - torch.mean(y, dim=[0, 1], keepdim=True)
        return mse(y @ self.q.t() @ self.q, y) / mse(y_bar, torch.zeros_like(y_bar))

    def nmse_y(self, y, x):
        mse = nn.MSELoss(reduction='mean')
        y_bar = y - torch.mean(y, dim=[0, 1], keepdim=True)
        return mse(x @ self.q, y) / mse(y_bar, torch.zeros_like(y_bar))

    def fit(self, u, z, y, epochs, lr, l_y, weight_decay, verbose=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        n_samples = u.shape[0]
        batch_size = 128

        loss_history = []
        for i in range(epochs):
            # Shuffle indices on-device (avoids DataLoader Python overhead)
            indices = torch.randperm(n_samples, device=u.device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = indices[start:start + batch_size]
                u_batch = u[idx]
                z_batch = z[idx]
                y_batch = y[idx]

                optimizer.zero_grad()
                x_batch = self.forward(u_batch)
                loss = self.loss_function(x_batch, z_batch, y_batch, l_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach()  # stay on GPU, no sync
                n_batches += 1

                # Re-calculate q after a is updated
                self.q = self.cayley_transform(self.a)
                # Re-apply connectivity masks
                self.connectivity_masks()

            if verbose and i % 10 == 0:
                # Only sync to CPU for logging every 10 epochs
                avg_loss = (epoch_loss / n_batches).item()
                with torch.no_grad():
                    x = self.forward(u)
                    mse_z_val = self.mse_z(x, z).item()
                    nmse_y_val = self.nmse_y(y, x).item()
                print(f'Epoch: {i}/{epochs}............. mse_z: {mse_z_val:.4f} nmse_y: {nmse_y_val:.4f}')
                loss_history.append(avg_loss)
        return loss_history
