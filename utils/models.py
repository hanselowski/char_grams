import torch
import torch.nn as nn

class BigLMLogitsModel(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super(BigLMLogitsModel, self).__init__()
        self.emb = nn.Embedding(num_chars, num_chars)

    def forward(self, indices):
        return self.emb(indices)

# linear output layer model
class BigLMLinearModel(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super(BigLMLinearModel, self).__init__()
        self.emb = nn.Embedding(num_chars, hidden_size)
        self.lin_out_layer = torch.nn.Parameter(torch.rand(num_chars, hidden_size))

    def forward(self, indices=None, features=None):
        if indices is not None:
            return torch.matmul(self.emb(indices), self.lin_out_layer.T)
        else:
            return torch.matmul(features, self.lin_out_layer.T)
    
    def activations(self, indices):
        return self.emb(indices)

# transition_matrix_lm: linear transformation model
class BigLMTranModel(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super(BigLMTranModel, self).__init__()
        self.emb = nn.Embedding(num_chars, hidden_size)
        self.tran_mat = torch.nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.lin_out_layer = self.emb.weight

    def forward(self, indices):
        X = self.emb(indices)
        Y = torch.matmul(self.tran_mat, X.T).T
        return torch.matmul(Y, self.lin_out_layer.T)
    
    def activations(self, indices):
        X = self.emb(indices)
        return torch.matmul(self.tran_mat, X.T).T


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, ngrm_num=2):
        super(TwoLayerMLP, self).__init__()
        self.V = nn.Embedding(input_dim, hidden_dim)
        self.W = torch.nn.Parameter(torch.rand(ngrm_num, output_dim, hidden_dim)) # Shape: [2, 8, 3]
        self.U = torch.nn.Parameter(torch.rand(output_dim, output_dim*2))
        self.relu = nn.ReLU()
        
    def forward(self, indices):
        X = self.V(indices)
        X = torch.einsum('bch,coh->bco', X, self.W)
        X = self.relu(X)
        X = X.reshape(X.shape[0], -1)
        logits = X @ self.U.T
        return logits


class NGramLinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, ngrm_num=2):
        super(NGramLinearModel, self).__init__()
        self.V = nn.Embedding(input_dim, hidden_dim)
        self.U = torch.nn.Parameter(torch.rand(ngrm_num, output_dim, hidden_dim)) 

    def forward(self, indices):
        X = self.V(indices)
        X = torch.einsum('bch,coh->bco', X, self.U)
        logits = X.sum(dim=1) 
        return logits
    

'''
        # Reshape X to combine batch and pair dimensions: [10, 2, 3] -> [20, 3]
        X_reshaped = X.reshape(-1, 3)

        # Create position indices to select the correct transformation from W
        position_indices = torch.arange(2).repeat(10)  # [0,1, 0,1, ..., 0,1]

        # Select the appropriate W for each position
        W_selected = self.W[position_indices]  # Shape: [20, 8, 3]

        # Perform the multiplication
        Y = torch.matmul(X_reshaped.unsqueeze(1), W_selected.transpose(1, 2))  # Shape: [20, 1, 8]
        Y = Y.squeeze(1)  # Shape: [20, 8]

        # Reshape back to original batch and pair dimensions
        Y = Y.reshape(10, 2, 8)  # Final shape: [10, 2, 8]



        W_reshaped = self.W.transpose(1, 2)  # Shape becomes [2, 3, 8]
        X = torch.matmul(X, W_reshaped)  # Result shape: [10, 2, 8]
'''