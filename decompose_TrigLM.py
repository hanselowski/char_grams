import torch
from utils.decompose import DictionaryLearner, plot_matrix_bases

model_name = "NGramLinearModel"

# words:  ['nathan*', 'dunn*']
# ['*', 'a', 'd', 'h', 'n', 't', 'u']
# unique_chars:  ['a', 'd', 'h', 'n', 't', 'u']
V = torch.load(f'tensors/{model_name}/dim_3/V_words_2.npy')
# W = torch.load(f'tensors/{model_name}/dim_3/W_words_2.npy')
U = torch.load(f'tensors/{model_name}/dim_3/U_words_2.npy')


chars = ['#', '*', 'a', 'd', 'h', 'n', 't', 'u']
plot_matrix_bases(matrices=[U[1], U[0], V], colors=['b','c','r'], legends=["U[1]", "U[0]", "V"], chars=chars)


U_cat = torch.cat([U[0], U[1]], dim=0)



# find class features
learner = DictionaryLearner()
W_recon, C = learner.learn_dictionary(V, lambda_reg=0.1, num_iterations=2000, fixed_dictionary=U_cat.T)
X = torch.einsum('bch,coh->bco', V, U)
logits = X.sum(dim=1) 
print("logits matrix: \n", logits)
print("embedding matrix V: \n", V.T)
print("W: \n", W.T)
print("C: \n", C)


''' 
chars = ['#', '*', 'a', 'd', 'h', 'n', 't', 'u']
print("projection_matrix W: \n", W)
print("embedding matrix V: \n", V)
# plot_matrix_bases(W, V, chars)

COR = W @ W.T
L = COR @ C
print("Logits: \n", L)
print("Correlation matrix: \n", COR)

print("C[:,2]: ", C[:,2])
print("COR[-1:]: ", COR[-1:])
print("COR[-1,:]@C[:,2]: ", COR[-1,:]@C[:,2])

print("C[:,3]: ", C[:,3])
print("COR[-1:]: ", COR[-1:])
print("COR[-1,:]@C[:,3]: ", COR[-1,:]@C[:,3])
'''

