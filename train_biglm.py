import torch
from utils.utils import plot_matrix_bases, calculate_perplexity, sample_greedy, create_dataset
from utils.models import BigLMLogitsModel, BigLMLinearModel, BigLMTranModel
from utils.losses import CELoss, CorrectedCELoss

# parameters
num_words = 2
hidden_size = 3
learning_rate = 1e-3
nmb_epochs = 5000



# Train model ---------------------------------------------------------------

# create dataset
data, chars, target_indices_lst = create_dataset(num_words=num_words)

# select model
# model_class = BigLMLogitsModel
# model_class = BigLMLinearModel
model_class = BigLMTranModel
model = model_class(num_chars=len(chars), hidden_size=hidden_size)

# select loss function
# criterion = CELoss()
criterion = CorrectedCELoss()

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

# run training loop
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



# Print matrices ---------------------------------------------------------------
'''
if type(model) == BigLMLinearModel:
    custom_model = True
else:
    custom_model = False
'''
custom_model = False

# Sample words greedily
generated_word = sample_greedy(model, chars, custom_model=custom_model)
print("Generated word:", generated_word)

# Calculate perplexity
perplexity = calculate_perplexity(model, data_tens, custom_model=custom_model)
print(f'Perplexity: {perplexity}')

# Print the learned projection matrix W and embedding matrix V
V = model.emb.weight
W = model.lin_out_layer

print("projection_matrix W: \n", W)
print("embedding matrix V: \n", V.T)
plot_matrix_bases(W, V.T, chars)