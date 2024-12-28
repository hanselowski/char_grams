import torch
from utils.utils import plot_matrix_bases, calculate_perplexity, sample_greedy, create_dataset
from utils.models import BigLMLogitsModel, BigLMLinearModel, BigLMTranModel
from utils.losses import CELoss, CorrectedCELoss
import numpy as np

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
model_class = BigLMLinearModel
# model_class = BigLMTranModel
model = model_class(num_chars=len(chars), hidden_size=hidden_size)

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

# Sample words greedily
generated_word = sample_greedy(model, chars)
print("Generated word:", generated_word)

# Calculate perplexity
perplexity = calculate_perplexity(model, data_tens)
print(f'Perplexity: {perplexity}')


if type(model) == BigLMLogitsModel:
    L = model.emb.weight
    L_red = L[1:,1:].T
    print(L_red)

    # save them as torch tensors
    torch.save(L_red, f'tensors/BigLMLogitsModel/L_red_num_words_{num_words}.npy')

    # also as csv files
    np.savetxt(f'tensors/BigLMLogitsModel/L_red_words_{num_words}.csv', L_red.detach().numpy(), delimiter=',')


elif type(model) == BigLMLinearModel: 
    # Print the learned projection matrix W and embedding matrix V
    V = model.emb.weight
    W = model.lin_out_layer

    # save V
    torch.save(V, f'tensors/BigLMLinearModel/V_words_{num_words}.npy')
    np.savetxt(f'tensors/BigLMLinearModel/V_words_{num_words}.csv', V.detach().numpy(), delimiter=',')

    # save W
    torch.save(W, f'tensors/BigLMLinearModel/W_words_{num_words}.npy')
    np.savetxt(f'tensors/BigLMLinearModel/W_words_{num_words}.csv', W.detach().numpy(), delimiter=',')

    print("projection_matrix W: \n", W)
    print("embedding matrix V: \n", V)
    plot_matrix_bases(W, V, chars)