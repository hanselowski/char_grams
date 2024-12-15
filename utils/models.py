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

    def forward(self, indices):
        return torch.matmul(self.emb(indices), self.lin_out_layer.T)
    
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
