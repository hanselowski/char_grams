import torch
from utils.utils import plot_matrix_bases, calculate_perplexity, sample_greedy, create_dataset, create_ngram_dataset
from utils.models import TwoLayerMLP, NGramLinearModel, OneLoopNODE
from utils.losses import CELoss
import numpy as np

# parameters
num_words = 2  # 1048

hidden_dim = 3
ngrm_num = 2


learning_rate = 1e-3
nmb_epochs = 5000


# Train model ---------------------------------------------------------------

# create dataset
input_indices, target_indices, data, chars = create_ngram_dataset(num_words=num_words, ngrm_num=ngrm_num, first_word=False)

# select model
model_class = TwoLayerMLP
model = model_class(input_dim=len(chars), hidden_dim=hidden_dim, output_dim=len(chars), ngrm_num=ngrm_num)


# select loss function
criterion = CELoss()

# optimizer
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# run training loop
for epoch in range(nmb_epochs):
    optim.zero_grad()

    loss = criterion(model, input_indices=input_indices, target_indices=target_indices)
    loss.backward()
    optim.step()

    if epoch%100 == 0:
        print(loss.item())



# Tensor analysis ---------------------------------------------------------------

# Sample words greedily
generated_word = sample_greedy(model, input_indices, chars, ngrm_num=ngrm_num)
print("Generated word:", generated_word)

# Calculate perplexity
perplexity = calculate_perplexity(model, input_indices, target_indices)
print(f'Perplexity: {perplexity}')



# Get tensors
V = model.V.weight
W = model.W
U = model.U

model_name = model.__class__.__name__

# save V
torch.save(V, f'tensors/{model_name}/V_words_{num_words}.npy')
# np.savetxt(f'tensors/{model_name}/V_words_{num_words}.csv', V.detach().numpy(), delimiter=',')
print("embedding matrix V: \n", V)

# save U
torch.save(U, f'tensors/{model_name}/U_words_{num_words}.npy')
# np.savetxt(f'tensors/{model_name}/U_words_{num_words}.csv', U.detach().numpy(), delimiter=',')
print("output matrix U: \n", U)

if type(model) == TwoLayerMLP:
    W = model.W
    torch.save(W, f'tensors/{model_name}/W_words_{num_words}.npy')
    # np.savetxt(f'tensors/{model_name}/W_words_{num_words}.csv', W.detach().numpy(), delimiter=',')
    print("projection_matrix W: \n", W)
    plot_matrix_bases(matrices=[U, W[1], W[0], V], colors=['g','b','c','r'], legends=["U", "W[1]", "W[0]", "V"], chars=chars)

else:
    plot_matrix_bases(matrices=[U[1], U[0], V], colors=['b','c','r'], legends=["U[1]", "U[0]", "V"], chars=chars)


