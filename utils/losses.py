import torch
import torch.nn as nn
from utils.models import BigLMLogitsModel

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, model, input_indices, target_indices, labels=None):
        logits = model(input_indices)
        return self.loss_fun(logits, target_indices)

class CorrectedCELoss(nn.Module):
    def __init__(self):
        super(CorrectedCELoss, self).__init__()
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, model, input_indices, target_indices):
        Y = model.activations(input_indices)  # shape: samples x hidden_size
        T = model.lin_out_layer[target_indices] # shape: samples x vocab_size x hidden_size

        # Reshape Y for batch matrix multiplication
        Y_batched = Y.unsqueeze(-1)  # shape: samples x hidden_size x 1

        # Batch matrix multiplication
        logits = torch.matmul(T, Y_batched).squeeze(-1)  # shape: samples x vocab_size

        labels = torch.zeros(len(Y), dtype=torch.long)
        return self.loss_fun(logits, labels)
