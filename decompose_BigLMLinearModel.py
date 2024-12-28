import torch
from utils.decompose import DictionaryLearner


# words:  ['nathan*', 'dunn*']
# ['*', 'a', 'd', 'h', 'n', 't', 'u']
# unique_chars:  ['a', 'd', 'h', 'n', 't', 'u']
W = torch.load('tensors/BigLMLinearModel/dim_3/W_words_2.npy')
V = torch.load('tensors/BigLMLinearModel/dim_3/V_words_2.npy')

# W = torch.load('tensors/BigLMLinearModel/dim_6/W_words_2.npy')
# V = torch.load('tensors/BigLMLinearModel/dim_6/V_words_2.npy')


# find class features
learner = DictionaryLearner()
W_recon, C = learner.learn_dictionary(V, lambda_reg=0.1, num_iterations=2000, fixed_dictionary=W.T)
print("logits matrix: \n", torch.matmul(V, W.T))
print("embedding matrix V: \n", V.T)
print("W: \n", W.T)
print("C: \n", C)

# V = V[1:, :] # remove entry for * 
chars = ['*', 'a', 'd', 'h', 'n', 't', 'u']
print("projection_matrix W: \n", W)
print("embedding matrix V: \n", V)
# plot_matrix_bases(W, V, chars)

COR = W @ W.T
L = COR @ C
print("Logits: \n", L)
print("Correlation matrix: \n", COR)

print("C[:,2]: ", C[:,2])
print("COR[-1:]: ", COR[-1:])
print("COR[-1,:]@C[:,2]: ", COR[-1,:]@C[:,2])

print("C[:,3]: ", C[:,3])
print("COR[-1:]: ", COR[-1:])
print("COR[-1,:]@C[:,3]: ", COR[-1,:]@C[:,3])




'''
# One linear soft max layer

# Math operations
torch.set_printoptions(sci_mode=False)

# convert V in W basis
L = W @ V.T
V.T ~= W.T@C
COR = W @ W.T
L ~= W @ W.T @ C
P = torch.softmax(L, dim=0)
P ~= torch.softmax(W @ W.T @ C, dim=0)
P ~= C

COR[-1,:]
C[:,2]
COR[-1,:]@C[:,2]


COR[-1,:]
tensor([-0.8079, -2.6716,  3.1223,  0.6771, -2.7933,  1.5520,  6.0865],
       grad_fn=<SliceBackward0>)
C[:,2]
tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.8112, 0.0000, 1.6119])
COR[-1,:]@C[:,2]
tensor(7.5453, grad_fn=<DotBackward0>)



# words:  ['nathan*', 'dunn*']
# C: class features
        *         a       d       h       n       t       u
 tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.8624, 0.0000, 0.0000], *
        [0.0481, 0.0000, 0.0000, 1.7929, 0.5251, 0.0000, 0.0000],  a
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  d
        [0.6032, 0.0000, 0.0000, 0.2564, 0.0000, 1.9296, 0.2969],  h
        [0.0000, 2.2848, 0.8112, 0.2723, 1.5382, 0.7428, 2.9361],  n
        [0.0000, 1.5621, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  t
        [0.0000, 0.1185, 1.6119, 0.0000, 0.4127, 0.0000, 0.5934]]) u


Correlation matrix: 
COR = W @ W.T
tensor([[ 2.7292,  2.5777,  0.8280, -1.4495,  0.1068, -1.0328, -0.8079],
        [ 2.5777,  5.1876,  1.9846,  0.0898, -0.6495, -1.2011, -2.6716],
        [ 0.8280,  1.9846,  8.6855,  1.7765, -3.4743,  3.6155,  3.1223],
        [-1.4495,  0.0898,  1.7765,  3.9013, -1.8747,  0.9160,  0.6771],
        [ 0.1068, -0.6495, -3.4743, -1.8747,  2.7599, -0.6263, -2.7933],
        [-1.0328, -1.2011,  3.6155,  0.9160, -0.6263,  3.9963,  1.5520],
        [-0.8079, -2.6716,  3.1223,  0.6771, -2.7933,  1.5520,  6.0865]],

P: probabilities = torch.softmax(L, dim=0)
tensor([[    0.0118,     0.0003,     0.0002,     0.0012,     0.4996,     0.0000,    0.0007],
        [    0.0853,     0.0000,     0.0000,     0.9983,     0.2498,     0.0006,    0.0000],
        [    0.0875,     0.0002,     0.0001,     0.0002,     0.0001,     0.0003,    0.0000],
        [    0.6656,     0.0001,     0.0002,     0.0002,     0.0002,     0.9969,    0.0001],
        [    0.0193,     0.4998,     0.0001,     0.0000,     0.2495,     0.0005,    0.9986],
        [    0.0856,     0.4994,     0.0002,     0.0000,     0.0004,     0.0008,    0.0006],
        [    0.0449,     0.0001,     0.9993,     0.0000,     0.0003,     0.0008,    0.0000]])


L: Logits = W @ V.T
tensor([[-1.5772, -2.3816, -1.1272,  2.5634,  3.6381, -3.6939, -1.2769],
        [ 0.4035, -4.1175, -4.7625,  9.2474,  2.9451, -1.1635, -4.7058],
        [ 0.4287, -2.6895, -1.9698,  0.8889, -4.9370, -1.7590, -7.7951],
        [ 2.4576, -3.1313, -0.7355,  0.7508, -4.0598,  6.2354, -3.8440],
        [-1.0838,  5.0964, -2.1638, -0.7937,  2.9436, -1.4674,  5.9890],
        [ 0.4061,  5.0957, -0.8893, -2.9150, -3.3871, -0.8707, -1.4960],
        [-0.2389, -3.1366,  7.6453, -6.2566, -3.7841, -0.8569, -4.2884]],
       grad_fn=<MmBackward0>)



'''



######################################################
######################################################
''' 
# Atom features
logits = W @ V.T
V_recon = D @ A
assert W @ V.T == W @ V_recon
'''


'''
((
# words:  ['nathan*', 'dunn*']
# unique_chars:  ['a', 'd', 'h', 'n', 't', 'u']
# analysis of atomic features
 tensor([[0.0000, 3.8518, 0.0000, 0.3315, 2.4077, 0.0000],
        [4.4555, 0.0746, 0.0000, 0.0000, 0.0000, 0.3682],
        [0.0000, 0.0000, 4.3154, 3.0664, 0.0000, 0.0000], a
        [0.0000, 0.0000, 0.0000, 2.2813, 0.0000, 3.9270], n
        [0.0000, 0.0000, 2.5356, 0.0000, 4.3748, 0.6876]])
))
'''


'''
# Two layer network relu and softmax
P = softmax(U @ relu(W @ V))

# Ansatz I
L1 = W @ V
V.T ~= W.T@Cv
R = relu(W @ V)
R ~= Cv
P ~ softmax(U @ Cv)

# Ansatz II
relu(W @ V) ~= U.T@C2

# Ansatz III:
P = softmax(U @ relu(W @ V))
W ~= U @ C
P = softmax(U @ relu(U @ C @ V))

# Ansatz IV:
L = W @ V
V = D @ C1
L = P @ C2

P @ C2 = W @ D @ C1
P = C2^-1 @ W @ D @ C1 
P = T1 @ C1
''' 

''' 
# find atoms (atom features) for embedding matrix V
learner = DictionaryLearner()
D, A = learner.learn_dictionary(V, num_dict_atoms=5, lambda_reg=0.1, num_iterations=1000)
print("embedding matrix V: \n", V.T)
print("D: \n", D)
print("A: \n", A)
'''




'''
learner = DictionaryLearner()
D, S = learner.learn_dictionary(L_red)
print("D: ", D)
print("S: ", S)


# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(L_red)
print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Convert eigenvalues to a diagonal matrix
Lambda = torch.diag(eigenvalues)

# Reconstruct L_red
L_red_reconstructed = eigenvectors @ Lambda @ torch.linalg.inv(eigenvectors)

print("\nReconstructed L_red:")
print(L_red_reconstructed)

dictionary = Lambda @ torch.linalg.inv(eigenvectors)
print("\ndictionary:")
print(dictionary)
first_colum = eigenvectors @ dictionary[:,:1]
print("\nfirst_colum:")
print(first_colum)
'''




''' 
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