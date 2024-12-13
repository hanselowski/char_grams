import torch
import torch.nn as nn
from utils import plot_matrix_bases, calculate_perplexity, sample_greedy, create_dataset
from collections import defaultdict
import random


# parameters
num_words = 5
hidden_size = 3
learning_rate = 1e-3
nmb_epochs = 10000


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

# CEL loss [done]
# positives: Y = model(input_indices), logits = emb_mat @ Y (all with all)
# loss = CEL(logits, target_indices)
# criterion = nn.CrossEntropyLoss()
class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, model, input_indices, target_indices, labels=None):
        Y = model(input_indices)
        logits = torch.matmul(model.lin_out_layer, Y.T)
        return self.loss_fun(logits.T, target_indices)
    
class CustomCELoss(nn.Module):
    def __init__(self):
        super(CustomCELoss, self).__init__()
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




# transition_matrix_lm: multiple negatives ranking loss
# positives: Y = transition_matrix @ X, logits_pos = T @ Y (all with all)
# loss = CEL(logits_pos, torch.tensor(range(len(logits_pos))))
class MNRLossDefect(nn.Module):
    def __init__(self):
        super(MNRLoss, self).__init__()
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, model, input_indices, target_indices, labels=None):
        Y = model(input_indices)
        T = model.lin_out_layer[target_indices]
        logits = torch.matmul(Y, T.T)
        return self.loss_fun(logits, torch.tensor(range(len(Y))))


'''
# dot product loss (CEL) (triplet loss)
positives: Y = model(input_indices), logits_pos = T @ Y
negatives: logits_neg = T @ X
loss = CEL(torch.cat(logits_pos, logits_neg, dim=0), torch.zeros(len(logits_pos)))


# cosine sim loss (triplet loss)
positives: Y = model(input_indices), cos_sim(T,Y)
negatives: cos_sim(T,X)
loss = (1 - cos_sim(T,Y) + cos_sim(T,X)).mean()


# word2vec loss
v = self.embedding_v(center_words)
u = self.embedding_u(context_words)
scores = torch.mul(v, u).sum(dim=1)
log_probs = torch.logsigmoid(scores)
-log_probs.mean()
'''




# Train model ---------------------------------------------------------------

# create dataset
data, chars = create_dataset(num_words=num_words)


# postprocess data

inpind2outlst = defaultdict(list)
for t in data:
    inpind2outlst[t[0]].append(t[1])

output_indices_all = [i for i in range(len(chars))]

all_neg_indices_lst = []
for t in data:
    inp_indx = t[0]
    pot_pos = t[1]
    all_pos = inpind2outlst[inp_indx]
    all_neg = list(set(output_indices_all) - set(all_pos))
    assert t[1] not in all_neg
    all_neg_indices_lst.append(all_neg)

# padding
max_length = max(len(t) for t in all_neg_indices_lst)
all_neg_indices_lst_padded = [
    list(t) + random.choices(t, k=max_length - len(t)) if len(t) < max_length else list(t)
    for t in all_neg_indices_lst
]

# add positives
all_target_indices_lst_padded = []
for t, lst_neg in zip(data, all_neg_indices_lst_padded):
    all_target_indices_lst_padded.append([t[1]] + lst_neg)




'''
# construct clean batches
data = sorted(data, key=lambda x: x[0])
data_lsts = []
for t in data:
    if len(data_lsts) == 0 or data_lsts[-1][0] != t[0]:
        data_lsts.append([t])
    else:
        data_lsts[-1].append(t)

data_lsts = sorted(data_lsts, key=lambda x: len(x))
'''


# select model
# model = BigLMTransClass(num_chars=len(chars), hidden_size=hidden_size)
model = BigLMLinearClass(num_chars=len(chars), hidden_size=hidden_size)


# select loss function
# criterion = CELoss()
criterion = CustomCELoss()
# criterion = MNRLossDefect()


# optimizer
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# convert data to tensor
data_tens = torch.tensor(data)
if type(criterion) == CustomCELoss:
    input_indices = torch.tensor([t[0] for t in data])
    target_indices = torch.tensor(all_target_indices_lst_padded)
else:
    input_indices = data_tens[:, 0]
    target_indices = data_tens[:, 1]
# all_neg_indices = torch.tensor(all_neg_indices_lst_padded)


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








