
import torch
import torch.nn as nn
from utils import plot_matrix_bases, calculate_perplexity, sample_greedy


# parameters
num_words = 100
hidden_size = 3
learning_rate = 1e-3
nmb_epochs = 5000
# ...
num_sents = 10000


# Model and loss functions ---------------------------------------------------------------

# classification model
class BigLM(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super(BigLM, self).__init__()
        self.embedding = nn.Embedding(num_chars, hidden_size)
        self.projection_matrix = torch.nn.Parameter(torch.rand(num_chars, hidden_size))

    def forward(self, indices):
        X = self.embedding(indices)
        Y = torch.matmul(self.projection_matrix, X.T)
        return Y
criterion = nn.CrossEntropyLoss()



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
model = BigLM(num_chars=len(chars), hidden_size=hidden_size)


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
    logits = model(input_indices)
    loss = criterion(logits.T, target_indices)

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








