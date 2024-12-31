import torch
import torch.optim as optim
from pathlib import Path
import logging
from tqdm import tqdm


def find_sparse_dictionary(W, V, lambda_reg=0.1, num_epochs=1000, learning_rate=0.1):
    """
    Finds a sparse dictionary D such that V ≈ W D.

    Parameters:
    - W: torch.Tensor or numpy.ndarray of shape (m, n)
    - V: torch.Tensor or numpy.ndarray of shape (m, k)
    - lambda_reg: Regularization parameter for sparsity (default 0.1)
    - num_epochs: Number of iterations for optimization (default 1000)
    - learning_rate: Learning rate for the optimizer (default 0.1)

    Returns:
    - D: torch.Tensor of shape (n, k), the learned sparse dictionary
    """
    # Ensure W and V are torch tensors
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W, dtype=torch.float32)
    if not isinstance(V, torch.Tensor):
        V = torch.tensor(V, dtype=torch.float32)

    # Get dimensions
    m, n = W.shape
    m_V, k = V.shape

    # Check that the number of rows in V equals the number of rows in W
    if m != m_V:
        raise ValueError("Number of rows in W and V must be the same.")

    # Initialize the dictionary D to be learned
    D = torch.randn(n, k, requires_grad=True)

    # Define the optimizer
    optimizer = optim.Adam([D], lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Compute the reconstruction loss (Frobenius norm)
        reconstruction_loss = torch.norm(V - W @ D, p='fro') ** 2
        
        # Compute the L1 regularization term to enforce sparsity
        l1_regularization = lambda_reg * torch.norm(D, p=1)
        
        # Total loss
        loss = reconstruction_loss + l1_regularization
        
        # Backpropagation
        loss.backward()
        
        # Update the dictionary D
        optimizer.step()
        
        # Optional: Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}')
    
    # Return the learned dictionary D
    return D.detach()


class DictionaryLearner:
    def __init__(self):
        # Configure paths
        self.base_dir = Path('data')
        self.decomp_dir = self.base_dir / 'svd_decompositions'
        self.dict_dir = self.base_dir / 'dictionary_learning'
        self.vocab_sizes = [1000]
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

    def update_S_ISTA(self, D, S, X, lambda_reg, num_iters=50, lr=0.1, only_positive=True):
        """Update sparse codes using ISTA."""
        D = D.detach()
        S = S.clone().detach()
        
        for _ in range(num_iters):
            # Compute the gradient: X ≈ D @ S.T, so gradient is D.T @ (D @ S.T - X).T
            reconstruction = torch.matmul(D, S)  # S.t()
            grad = torch.matmul(D.t(), reconstruction - X)  #.t()
            # Gradient descent step
            S = S - lr * grad
            # Soft-thresholding (proximal operator for L1 norm)
            if only_positive:
                return torch.clamp(S - lr * lambda_reg, min=0.0)  # positive constraint
            else:
                return torch.sign(S) * torch.clamp(torch.abs(S) - lr * lambda_reg, min=0.0)

    def update_D(self, D, S, X, num_iters=10, lr=0.1):
        """Update dictionary using gradient descent."""
        S = S.detach()
        D = D.detach().clone()
        D.requires_grad = True
        optimizer_D = torch.optim.Adam([D], lr=lr)
        
        for _ in range(num_iters):
            optimizer_D.zero_grad()
            reconstruction = torch.matmul(D, S)  # S.t()
            loss = torch.norm(X - reconstruction, 'fro')**2
            loss.backward()
            optimizer_D.step()

            with torch.no_grad():
                D.data = D.data / torch.norm(D.data, dim=0, keepdim=True)
        
        return D.detach()

    def learn_dictionary(self, V, num_dict_atoms=5, lambda_reg=0.1, num_iterations=1000, fixed_dictionary=None, only_positive=True):
        """Perform dictionary learning on input matrix V. Optionally use a fixed dictionary."""
        N, M = V.shape  # N: number of words (1001), M: embedding dimension (15)
        K = num_dict_atoms if fixed_dictionary is None else fixed_dictionary.shape[1]
        V = V.t()  # Transpose V for easier matrix multiplication (M x N)
        
        # Use provided dictionary or initialize new one
        if fixed_dictionary is not None:
            D = fixed_dictionary.to(self.device)
            if D.shape[0] != M:
                raise ValueError(f"Dictionary dimension {D.shape[0]} doesn't match input dimension {M}")
        else:
            D = torch.randn(M, K, device=self.device)
            with torch.no_grad():
                D = D / torch.norm(D, dim=0, keepdim=True)
        
        # Initialize S with small positive values
        S = torch.abs(torch.randn(K, N, device=self.device) * 0.1)
        
        # Main loop for alternating updates
        pbar = tqdm(range(num_iterations), desc="Dictionary Learning")
        for i in pbar:
            # Update S using ISTA with non-negativity constraint
            S = self.update_S_ISTA(D, S, V, lambda_reg, num_iters=20, lr=0.01, only_positive=only_positive)
            
            # Update D only if no fixed dictionary is provided
            if fixed_dictionary is None:
                D = self.update_D(D, S, V, num_iters=20, lr=0.01)
            
            if i % 10 == 0:
                with torch.no_grad():
                    reconstruction = torch.matmul(D, S)
                    loss = torch.norm(V - reconstruction, 'fro')**2 + lambda_reg * torch.norm(S, 1)
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                    logging.info(f"Iteration {i}, Loss: {loss.item()}")
                    print(f"Iteration {i}, Loss: {loss.item()}")
                    print(f"Reconstruction loss: {torch.norm(V - reconstruction, 'fro')**2}")
        
        return D.detach(), S.detach()