import torch
from pathlib import Path
import logging
from tqdm import tqdm
from utils import plot_matrix_bases



class DictionaryLearner:
    def __init__(self):
        # Configure paths
        self.base_dir = Path('data')
        self.decomp_dir = self.base_dir / 'svd_decompositions'
        self.dict_dir = self.base_dir / 'dictionary_learning'
        self.vocab_sizes = [1000]
        
        # Dictionary learning parameters
        self.lambda_reg = 0.02  # Sparsity regularization parameter
        self.num_iterations = 5000  # Number of total iterations
        self.num_dict_atoms = 50  # Number of dictionary atoms (K)
        
        # Set up directories and logging
        self._setup_directories()
        self._setup_logging()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

    def update_S_ISTA(self, D, S, X, lambda_reg, num_iters=50, lr=0.1):
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
            S = torch.sign(S) * torch.clamp(torch.abs(S) - lr * lambda_reg, min=0.0)
        
        return S

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
            # Normalize columns of D
            with torch.no_grad():
                D.data = D.data / torch.norm(D.data, dim=0, keepdim=True)
        
        return D.detach()

    def learn_dictionary(self, V):
        """Perform dictionary learning on input matrix V."""
        N, M = V.shape  # N: number of words (1001), M: embedding dimension (15)
        K = self.num_dict_atoms  # Number of dictionary atoms (can be > M)
        V = V.t()  # Transpose V for easier matrix multiplication (M x N)
        
        # Initialize dictionary D (M x K) and sparse codes S (N x K)
        D = torch.randn(M, K, device=self.device)  # Each column is a dictionary atom of size M
        with torch.no_grad():
            D = D / torch.norm(D, dim=0, keepdim=True)
        
        S = torch.zeros(K, N, device=self.device)  # Each row represents sparse coefficients for a word
        
        # Main loop for alternating updates
        pbar = tqdm(range(self.num_iterations), desc="Dictionary Learning")
        for i in pbar:
            # Update S using ISTA
            S = self.update_S_ISTA(D, S, V, self.lambda_reg, num_iters=20, lr=0.01)
            # Update D
            D = self.update_D(D, S, V, num_iters=20, lr=0.01)
            
            if i % 10 == 0:
                with torch.no_grad():
                    reconstruction = torch.matmul(D, S) # S.t()
                    loss = torch.norm(V - reconstruction, 'fro')**2 + self.lambda_reg * torch.norm(S, 1)
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                    logging.info(f"Iteration {i}, Loss: {loss.item()}")
        
        return D.detach(), S.detach()


W = torch.load('tensors/W.pt')
V = torch.load('tensors/V.pt')
U = torch.cat([W, V], dim=0)

learner = DictionaryLearner()
D, S = learner.learn_dictionary(U)

illusrtate the decomposed matrices:
Fix one matrix and use it as a basis for the other


# Plot the original and transformed basis vectors
plot_matrix_bases(W, V.T, chars)





'''
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

class DictionaryLearner:
    def __init__(self):
        # Configure paths
        self.base_dir = Path('data')
        self.decomp_dir = self.base_dir / 'svd_decompositions'
        self.dict_dir = self.base_dir / 'dictionary_learning'
        self.vocab_sizes = [1000]
        
        # Dictionary learning parameters
        self.lambda_reg = 0.02  # Sparsity regularization parameter
        self.num_iterations = 5000  # Number of total iterations
        self.num_dict_atoms = 50  # Number of dictionary atoms (K)
        
        # Set up directories and logging
        self._setup_directories()
        self._setup_logging()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.dict_dir.mkdir(exist_ok=True)
        for subdir in ['dictionary', 'sparse_codes']:
            (self.dict_dir / subdir).mkdir(exist_ok=True)

    def _setup_logging(self):
        """Configure logging to both file and console."""
        log_file = self.base_dir / "logs" / f'dictionary_learning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def load_v_matrix(self, vocab_size):
        """Load reduced V matrix for given vocabulary size."""
        v_file = self.decomp_dir / 'V_reduced' / f"V_matrix_reduced_{vocab_size}.npy"
        if not v_file.exists():
            raise FileNotFoundError(f"Reduced V matrix not found for vocabulary size {vocab_size}")
        return torch.from_numpy(np.load(v_file)).float().to(self.device)

    def update_S_ISTA(self, D, S, X, lambda_reg, num_iters=50, lr=0.1):
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
            S = torch.sign(S) * torch.clamp(torch.abs(S) - lr * lambda_reg, min=0.0)
        
        return S

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
            # Normalize columns of D
            with torch.no_grad():
                D.data = D.data / torch.norm(D.data, dim=0, keepdim=True)
        
        return D.detach()

    def learn_dictionary(self, V):
        """Perform dictionary learning on input matrix V."""
        N, M = V.shape  # N: number of words (1001), M: embedding dimension (15)
        K = self.num_dict_atoms  # Number of dictionary atoms (can be > M)
        V = V.t()  # Transpose V for easier matrix multiplication (M x N)
        
        # Initialize dictionary D (M x K) and sparse codes S (N x K)
        D = torch.randn(M, K, device=self.device)  # Each column is a dictionary atom of size M
        with torch.no_grad():
            D = D / torch.norm(D, dim=0, keepdim=True)
        
        S = torch.zeros(K, N, device=self.device)  # Each row represents sparse coefficients for a word
        
        # Main loop for alternating updates
        pbar = tqdm(range(self.num_iterations), desc="Dictionary Learning")
        for i in pbar:
            # Update S using ISTA
            S = self.update_S_ISTA(D, S, V, self.lambda_reg, num_iters=20, lr=0.01)
            # Update D
            D = self.update_D(D, S, V, num_iters=20, lr=0.01)
            
            if i % 10 == 0:
                with torch.no_grad():
                    reconstruction = torch.matmul(D, S) # S.t()
                    loss = torch.norm(V - reconstruction, 'fro')**2 + self.lambda_reg * torch.norm(S, 1)
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                    logging.info(f"Iteration {i}, Loss: {loss.item()}")
        
        return D.detach(), S.detach()

    def save_results(self, D, S, vocab_size):
        """Save dictionary and sparse codes as both numpy arrays and pandas DataFrames."""
        # Ensure tensors are detached and moved to CPU before converting to numpy
        D_np = D.detach().cpu().numpy()
        S_np = S.detach().cpu().numpy()
        
        # Save as numpy arrays
        np.save(self.dict_dir / 'dictionary' / f"dictionary_{vocab_size}.npy", D_np)
        np.save(self.dict_dir / 'sparse_codes' / f"sparse_codes_{vocab_size}.npy", S_np)
        
        # Save as pandas DataFrames
        pd.DataFrame(D_np).to_csv(self.dict_dir / 'dictionary' / f"dictionary_{vocab_size}.csv", index=False)
        pd.DataFrame(S_np).to_csv(self.dict_dir / 'sparse_codes' / f"sparse_codes_{vocab_size}.csv", index=False)
        
        # Log sparsity statistics
        sparsity = (S_np == 0).sum() / S_np.size
        logging.info(f"Sparsity level for vocab size {vocab_size}: {sparsity:.2%}")

    def evaluate_reconstruction(self, D, S, V):
        """Evaluate reconstruction quality."""
        with torch.no_grad():
            reconstruction = torch.matmul(D, S)  # S.t()
            error = torch.norm(V.t() - reconstruction, 'fro')**2
            relative_error = error / torch.norm(V, 'fro')**2
            logging.info(f"Reconstruction error: {error.item():.4f}")
            logging.info(f"Relative reconstruction error: {relative_error.item():.4f}")
        return error.item(), relative_error.item()

    def process(self):
        """Main processing function."""
        results = []
        
        for vocab_size in self.vocab_sizes:
            logging.info(f"Processing vocabulary size: {vocab_size}")
            
            try:
                # Load reduced V matrix
                V = self.load_v_matrix(vocab_size)
                logging.info(f"Loaded V matrix of shape {V.shape}")
                
                # Perform dictionary learning
                D, S = self.learn_dictionary(V)
                logging.info(f"Dictionary shape: {D.shape}, Sparse codes shape: {S.shape}")
                
                # Evaluate reconstruction
                error, relative_error = self.evaluate_reconstruction(D, S, V)
                
                # Save results
                self.save_results(D, S, vocab_size)
                
                results.append({
                    'vocab_size': vocab_size,
                    'reconstruction_error': error,
                    'relative_error': relative_error,
                    'dictionary_shape': tuple(D.shape),
                    'sparse_codes_shape': tuple(S.shape)
                })
                
                logging.info(f"Completed processing for vocabulary size {vocab_size}")
                
            except Exception as e:
                logging.error(f"Error processing vocabulary size {vocab_size}: {str(e)}")
                continue
        
        # Save summary results
        pd.DataFrame(results).to_csv(self.dict_dir / 'summary_results.csv', index=False)

if __name__ == "__main__":
    learner = DictionaryLearner()
    learner.process()
'''