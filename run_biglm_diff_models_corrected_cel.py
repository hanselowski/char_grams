import torch
import torch.nn as nn
from utils import plot_matrix_bases, calculate_perplexity, sample_greedy, create_dataset


# parameters
num_words = 2
hidden_size = 3
learning_rate = 1e-3
nmb_epochs = 5000


# Model and loss functions ---------------------------------------------------------------

# transition_matrix_lm: linear transformation model
class BigLMTransClass(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super(BigLMTransClass, self).__init__()
        self.embedding = nn.Embedding(num_chars, hidden_size)
        self.projection_matrix = torch.nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.lin_out_layer = self.embedding.weight

    def forward(self, indices):
        X = self.embedding(indices)
        return torch.matmul(self.projection_matrix, X.T).T


# linear output layer model
class BigLMLinearClass(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super(BigLMLinearClass, self).__init__()
        self.embedding = nn.Embedding(num_chars, hidden_size)
        self.lin_out_layer = torch.nn.Parameter(torch.rand(num_chars, hidden_size))

    def forward(self, indices):
        return self.embedding(indices)


# loss functions ---------------------------------------------------------------

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, model, input_indices, target_indices, labels=None):
        Y = model(input_indices)
        logits = torch.matmul(model.lin_out_layer, Y.T)
        return self.loss_fun(logits.T, target_indices)


class CorrectedCELoss(nn.Module):
    def __init__(self):
        super(CorrectedCELoss, self).__init__()
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, model, input_indices, target_indices):
        Y = model(input_indices)  # shape: samples x hidden_size
        T = model.lin_out_layer[target_indices] # shape: samples x vocab_size x hidden_size

        # Reshape Y for batch matrix multiplication
        Y_batched = Y.unsqueeze(-1)  # shape: samples x hidden_size x 1

        # Batch matrix multiplication
        logits = torch.matmul(T, Y_batched).squeeze(-1)  # shape: samples x vocab_size

        labels = torch.zeros(len(Y), dtype=torch.long)
        return self.loss_fun(logits, labels)


# Train model ---------------------------------------------------------------

# create dataset
data, chars, target_indices_lst = create_dataset(num_words=num_words)


# select model
model = BigLMLinearClass(num_chars=len(chars), hidden_size=hidden_size)
# model = BigLMTransClass(num_chars=len(chars), hidden_size=hidden_size)


# select loss function
criterion = CELoss()
# criterion = CorrectedCELoss()


# optimizer
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# convert data to tensor
data_tens = torch.tensor(data)
if type(criterion) == CorrectedCELoss:
    input_indices = torch.tensor([t[0] for t in data])
    target_indices = torch.tensor(target_indices_lst)
else:
    input_indices = data_tens[:, 0]
    target_indices = data_tens[:, 1]


for epoch in range(nmb_epochs):
    optim.zero_grad()

    # model based loss: forward pass and loss calculation
    # loss = criterion(model, input_indices, target_indices)
    loss = criterion(model, input_indices, target_indices)

    # backward pass and optimization
    loss.backward()
    optim.step()

    if epoch%100 == 0:
        print(loss.item())


# Print the learned projection matrix W and embedding matrix V
V = model.embedding.weight
W = model.lin_out_layer

print("projection_matrix W: \n", W)
print("embedding matrix V: \n", V.T)

# Calculate perplexity
perplexity = calculate_perplexity(model, data_tens, custom_model=True)
print(f'Perplexity: {perplexity}')

# Sample words greedily
generated_word = sample_greedy(model, chars, custom_model=True)
print("Generated word:", generated_word)

# Plot the original and transformed basis vectors
plot_matrix_bases(W, V.T, chars)








