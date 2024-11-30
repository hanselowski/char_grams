
import torch
import torch.nn as nn
from utils import plot_matrix_bases, calculate_perplexity, sample_greedy


# parameters
num_words = 2
hidden_size = 3
learning_rate = 1e-3
nmb_epochs = 1000
# ...
num_sents = 10000


# Model and loss functions ---------------------------------------------------------------

# transition matrix model full output
'''
class BigLMTransClass(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super(BigLMTransClass, self).__init__()
        self.embedding = nn.Embedding(num_chars, hidden_size)
        self.projection_matrix = torch.nn.Parameter(torch.rand(hidden_size, hidden_size))

    def forward(self, indices):
        X = self.embedding(indices)
        Y = torch.matmul(self.projection_matrix, X.T)
        logits = torch.matmul(self.embedding.weight, Y)
        return logits


# linear output layer model
class BigLMLinearClass(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super(BigLMLinearClass, self).__init__()
        self.embedding = nn.Embedding(num_chars, hidden_size)
        self.projection_matrix = torch.nn.Parameter(torch.rand(num_chars, hidden_size))

    def forward(self, indices):
        X = self.embedding(indices)
        logits = torch.matmul(self.projection_matrix, X.T)
        return logits
'''
class BigLMTransClass(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super(BigLMTransClass, self).__init__()
        self.embedding = nn.Embedding(num_chars, hidden_size)
        self.projection_matrix = torch.nn.Parameter(torch.rand(hidden_size, hidden_size))

    def forward(self, indices):
        X = self.embedding(indices)
        Y = torch.matmul(self.projection_matrix, X.T)
        logits = torch.matmul(self.embedding.weight, Y)
        return logits

# loss functions ---------------------------------------------------------------

# CEL loss [done]
# positives: Y = model(input_indices), logits = emb_mat @ Y (all with all)
# loss = CEL(logits, target_indices)
# criterion = nn.CrossEntropyLoss()


# transition_matrix_lm: multiple negatives ranking loss
# positives: Y = transition_matrix @ X, logits_pos = T @ Y (all with all)
# loss = CEL(logits_pos, torch.tensor(range(len(logits_pos))))
class MNRLoss(nn.Module):
    def __init__(self):
        super(MNRLoss, self).__init__()
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, model, input_indices, target_indices, labels=None):
        Y = model(input_indices)
        T = model.embedding(target_indices)
        logits = torch.matmul(Y, T)
        return self.loss_fun(logits, torch.tensor(range(len(Y))))


# transition_matrix_lm: dot product loss (CEL) (triplet loss)
# positives: Y = transition_matrix @ X, logits_pos = T @ Y
# negatives: logits_neg = T @ X
# loss = CEL(torch.cat(logits_pos, logits_neg, dim=0), torch.zeros(len(logits_pos)))

# transition_matrix_lm: cosine sim loss (triplet loss)
# positives: Y = transition_matrix @ X, cos_sim(T,Y)
# negatives: cos_sim(T,X)
# loss = (1 - cos_sim(T,Y) + cos_sim(T,X)).mean()


'''
# word2vec loss
v = self.embedding_v(center_words)
u = self.embedding_u(context_words)
scores = torch.mul(v, u).sum(dim=1)
log_probs = torch.logsigmoid(scores)
-log_probs.mean()
'''


# create dataset ---------------------------------------------------------------
# load words
words = []
with open(f"data/words_num_sents_{num_sents}.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # split to words, add end or word character to each word, skip the word if it contains a character not in the 26-letter alphabet, that is words containing characters with accents, umlauts, etc. are skipped

        words.extend([word + '*' for word in line.strip().split() if all(char in "abcdefghijklmnopqrstuvwxyz" for char in word)])
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
# model = BigLMTransClass(num_chars=len(chars), hidden_size=hidden_size)
model = BigLMLinearClass(num_chars=len(chars), hidden_size=hidden_size)

# loss function
# criterion = nn.CrossEntropyLoss()
criterion = MNRLoss()


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
    # logits = model(input_indices)
    # loss 
    # loss = criterion(logits.T, target_indices)

    # model based loss
    loss = criterion(model, input_indices)

    # backward pass and optimization
    loss.backward()
    optim.step()

    if epoch%100 == 0:
        print(loss.item())


# Print the learned projection matrix W and embedding matrix V
W = model.projection_matrix
V = model.embedding.weight
print("projection_matrix W: \n", W)
print("embedding matrix V: \n", V.T)

# Calculate perplexity
perplexity = calculate_perplexity(model, data_tens)
print(f'Perplexity: {perplexity}')


# Example usage
generated_word = sample_greedy(model, chars)
print("Generated word:", generated_word)


# Plot the original and transformed basis vectors
plot_matrix_bases(W, V.T, chars)








