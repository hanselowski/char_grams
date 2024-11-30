
import torch
import torch.nn as nn
from utils import plot_matrix_bases

# parameters
lambda_contr_loss = 2.0
num_words = 1000
hidden_size = 3
learning_rate = 1e-3
nmb_epochs = 20000
# ...
num_sents = 10000


# Model loss functions ---------------------------------------------------------------
class BigLM(nn.Module):
    def __init__(self, num_chars, hidden_size, relu_activation=False):
        super(BigLM, self).__init__()
        self.embedding = nn.Embedding(num_chars, hidden_size)
        self.transition_matrix = torch.nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.relu_activation = relu_activation

    def forward(self, indices):
        X = self.embedding(indices)
        Y = torch.matmul(self.transition_matrix, X.T)
        if self.relu_activation:
            Y = torch.relu(Y)
        return Y

class CosineSimilarityLossTrans(nn.Module):
    def __init__(
        self,
        loss_fct: nn.Module = nn.MSELoss(),
        cos_score_transformation: nn.Module = nn.Identity(),
    ) -> None:

        super().__init__()
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, Y, T, labels):
        output = self.cos_score_transformation(torch.cosine_similarity(Y, T, dim=1))
        return self.loss_fct(output, labels.float().view(-1))

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, Y, T):
        # Compute cosine similarity
        cos_sim = torch.cosine_similarity(Y, T, dim=0)
        # Compute the loss
        loss = torch.mean(1 - cos_sim)
        return loss

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, X, Y, T):
        cos_sim_pos = torch.cosine_similarity(Y, T, dim=0)
        cos_sim_neg = torch.cosine_similarity(Y, X, dim=0)
        loss = torch.mean(1 - cos_sim_pos + torch.abs(cos_sim_neg))
        return loss


# create dataset ---------------------------------------------------------------
# load words
words = []
with open(f"data/words_num_sents_{num_sents}.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        words.extend(line.strip().split())
        if len(words) >= num_words:
            words = words[:num_words]
            break
print('words: ', words)

# get unique characters in words
unique_chars = set()
for word in words:
    for char in word:
        unique_chars.add(char)
chars = sorted(list(unique_chars))
print('unique_chars: ', chars)
print('len(unique_chars): ', len(unique_chars))

# create mapping from character to index
char2idx = {char: idx for idx, char in enumerate(chars)}

# create bigram dataset
data = []
for word in words:
    for i in range(len(word) - 1):
        data.append((char2idx[word[i]], char2idx[word[i+1]]))



# Train model ---------------------------------------------------------------
# create model
model = BigLM(num_chars=len(chars), hidden_size=hidden_size)

# cosine similarity loss
# criterion = CosineSimilarityLoss()
# criterion = CosineSimilarityLossTrans()
criterion = TripletLoss()

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

# convert data to tensor
data_tens = torch.tensor(data)
input_indices = data_tens[:, 0]
target_indices = data_tens[:, 1]
indices_contr = torch.tensor(list(char2idx.values()))

for epoch in range(nmb_epochs):
    optim.zero_grad()

    # forward pass
    input = model.embedding(input_indices)  # X
    output = model(input_indices)  # Y
    target = model.embedding(target_indices)  # T

    ''' 
    # cosine similarity loss
    labels_cos_sim = torch.ones(len(output))
    cos_sim_loss = criterion(output, target.T, labels_cos_sim)

    # contrastive loss
    output = model(indices_contr)
    target = model.embedding(indices_contr)
    
    # joint loss
    labels_contr = torch.zeros(len(output))
    cos_sim_contr_loss = criterion(output, target.T, labels_contr)
    
    # contrastive loss V2
    cos_sim_contr = torch.cosine_similarity(output.T, target, dim=1)
    # cos_sim_contr_loss = torch.mean(cos_sim_contr)
    cos_sim_contr_loss = torch.mean(torch.abs(cos_sim_contr))
    
    # loss
    loss = cos_sim_loss + lambda_contr_loss * cos_sim_contr_loss
    '''

    # triplet loss
    loss = criterion(input.T, output, target.T)

    # backward pass and optimization
    loss.backward()
    optim.step()

    if epoch%100 == 0:
        print(loss.item())


'''
print("target: \n", T)
print("reconstruction: \n", Y)
print()
print("cos_sim: ", cos_sim)
print("cos_sim_contr: ", cos_sim_contr)
'''
W = model.transition_matrix
V = model.embedding.weight

print("transition matrix W: \n", W)
print("embedding matrix V: \n", V.T)

plot_matrix_bases(W, V.T, chars)


    

