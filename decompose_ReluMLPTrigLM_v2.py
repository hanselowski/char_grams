import torch
from utils.decompose import DictionaryLearner
from utils.utils import plot_matrix_bases, create_ngram_dataset, create_heatmaps
torch.set_printoptions(precision=5, sci_mode=False)

model_name = "NGramLinearModel"

# words:  ['nathan*', 'dunn*']
# ['*', 'a', 'd', 'h', 'n', 't', 'u']
# unique_chars:  ['a', 'd', 'h', 'n', 't', 'u']
V = torch.load(f'tensors/{model_name}/dim_3/V_words_2.npy')
# W = torch.load(f'tensors/{model_name}/dim_3/W_words_2.npy')
U = torch.load(f'tensors/{model_name}/dim_3/U_words_2.npy')


chars = ['#', '*', 'a', 'd', 'h', 'n', 't', 'u']
# plot_matrix_bases(matrices=[U[1], U[0], V], colors=['b','c','r'], legends=["U[1]", "U[0]", "V"], chars=chars)


# find class features
U_cat = torch.cat([U[0], U[1]], dim=0)
learner = DictionaryLearner()
U_recon, C_cat = learner.learn_dictionary(V, lambda_reg=0.1, num_iterations=2000, fixed_dictionary=U_cat.T)
U_recon_0, C_0 = learner.learn_dictionary(V, lambda_reg=0.1, num_iterations=2000, fixed_dictionary=U[0].T)
U_recon_1, C_1 = learner.learn_dictionary(V, lambda_reg=0.1, num_iterations=2000, fixed_dictionary=U[1].T)
print("embedding matrix V: \n", V)
print("U[0]: \n", U[0])
print("C_0: \n", C_0)
C = torch.stack([C_0, C_1])

# compute logits
data, chars = create_ngram_dataset(num_words=2, ngrm_num=2)
data_tens = torch.tensor(data)
input_indices = data_tens[:, :2]
target_indices = data_tens[:, 2]

X = V[input_indices]
X = torch.einsum('bch,coh->bco', X, U)
L = X.sum(dim=1)
print("L: \n", L)
P = torch.softmax(L, dim=1)
print("P: \n", P)

## Activations
# create_heatmaps([C_0[:,input_indices[:,0]].T, C_1[:,input_indices[:,0]].T, P], titles=['C_0.T', 'C_1.T', 'P'])
# create_heatmaps([C_0[:,input_indices[:,0]].T + C_1[:,input_indices[:,0]].T, P], titles=['C_0.T + C_1.T', 'P'])
# create_heatmaps([U[0]@U[0].T, C_0[:,input_indices[:,0]], U[0]@U[0].T @ C_0[:,input_indices[:,0]], P.T], titles=['U[0] @ U[0].T', 'C_0', 'U[0] @ U[0].T @ C_0', 'P.T'])
# create_heatmaps([C_0[:,input_indices[:,0]], U[0]@U[0].T @ C_0[:,input_indices[:,0]], P.T], titles=['C_0', 'U[0] @ U[0].T @ C_0', 'P.T'])

# Feature extractors
# create_heatmaps([U[0], U[1], V], titles=['U[0]', 'U[1]', 'V'])
# create_heatmaps([U[0]@V.T, U[1]@V.T], titles=['U[0]@V.T', 'U[1]@V.T'])
# create_heatmaps([U[0]@V.T, U[0], V.T], titles=['U[0]@V.T', 'U[0]', 'V.T'])


plot_matrix_bases(matrices=[U[1], U[0], V], colors=['b','c','r'], legends=["U[1]", "U[0]", "V"], chars=chars)



''' 
torch.set_printoptions(precision=5, sci_mode=False)

# two independent Cs
L = V[input_indices[:,0]] @ U[0].T + V[input_indices[:,1]] @ U[1].T

V[input_indices[:,0]] ~= C_0[:,input_indices[:,0]].T @ U[0] 
V[input_indices[:,1]] ~= C_1[:,input_indices[:,1]].T @ U[1]

L = C_0[:,input_indices[:,0]].T @ U[0] @ U[0].T + C_1[:,input_indices[:,1]].T @ U[1] @ U[1].T

# joined C
L = V[input_indices[:,0]] @ U[0].T + V[input_indices[:,1]] @ U[1].T
P = torch.softmax(L, dim=1)

V[input_indices[:,0]] ~= C_cat[:,input_indices[:,0]].T @ U_cat 
V[input_indices[:,1]] ~= C_cat[:,input_indices[:,1]].T @ U_cat

L ~= C_cat[:,input_indices[:,0]].T @ U_cat @ U[0].T + C_cat[:,input_indices[:,1]].T @ U_cat @ U[1].T

Wrong: C_cat[:,input_indices[:,0]].T[:,:8] @ U[0] @ U[0].T + C_cat[:,input_indices[:,1]].T[:,8:] @ U[1] @ U[1].T

See what features are extracted
do some algebra to the above equations
Try to analyze what featrure are used in the first and what features are used in the second term

L = ( U[0], U[1] ) @ U_cat @ ( C_cat[:,input_indices[:,0]] + C_cat[:,input_indices[:,1]] )
'''








#########
'''
V.T ~= U_cat.T @ C_cat

L = [[U[0],    0],    
     [   0, U[1]]] @ V[input_indices]

     
L = [[U[0],    0],    
     [   0, U[1]]] @ [[U[0], U[1]]] @ [[C[input_indices[0]], C[input_indices[1]]]]

L = [[U[0]@U[0],    0],    
     [   0, U[1]@U[1]]] @ [[C[input_indices[0]], C[input_indices[1]]]]

'''





####


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

