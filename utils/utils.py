import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from collections import defaultdict
import random

def plot_matrix_bases(W, V, v_column_names, limits=(-3, 3)):
    """
    Plots the original basis vectors defined by V and the transformed basis vectors W,
    with labels for the columns of V.
    
    Args:
        W: Transformation matrix
        V: Original basis vectors
        v_column_names: List of names for the columns in V
    """
    # Detach tensors and convert to numpy arrays
    W = W.detach().cpu().numpy()
    V = V.detach().cpu().numpy()

    # Determine the dimensionality
    dim = V.shape[1]

    if dim == 2:
        # 2D plotting
        fig, ax = plt.subplots()
        
        # Create origin points for each vector
        num_vectors_V = V.shape[1]  # For V vectors
        num_vectors_W = W.shape[1]  # For W vectors (transformation matrix columns)
        
        origin_V = np.zeros((2, num_vectors_V))  # For V vectors
        origin_W = np.zeros((2, num_vectors_W))  # For W vectors

        # Plot original basis vectors with labels
        for i in range(num_vectors_V):
            q = ax.quiver(origin_V[0, i], origin_V[1, i], 
                         V[0, i], V[1, i], 
                         angles='xy', scale_units='xy', scale=1, 
                         color='b', label=f'V ({v_column_names[i]})')
            
            # Add text label at the tip of the vector
            # Calculate the position for the text (slightly offset from vector tip)
            tip_x = V[0, i]
            tip_y = V[1, i]
            # Add a small offset to avoid overlap with the arrow
            offset = 0.1
            ax.text(tip_x + offset, tip_y + offset, v_column_names[i], 
                   fontsize=10, color='b')

        # Plot transformed basis vectors (only the columns of W)
        for i in range(num_vectors_W):
            ax.quiver(origin_W[0, i], origin_W[1, i], 
                     W[0, i], W[1, i], 
                     angles='xy', scale_units='xy', scale=1, 
                     color='r', label='W' if i == 0 else "")

        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        plt.show()

    else:
        if dim > 3:
            pca = PCA(n_components=3)
            V = pca.fit_transform(V.T).T
            W = pca.fit_transform(W)


        # 3D plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        origin = np.zeros(3)  # Single origin point for 3D

        # Plot original basis vectors with labels
        for i in range(V.shape[0]):
            ax.quiver(*origin, V[i, 0], V[i, 1], V[i, 2], 
                     color='r', label=f'V ({v_column_names[i]})' if i == 0 else "")
            
            # Add text label at the tip of the vector
            tip_x = V[i, 0]
            tip_y = V[i, 1]
            tip_z = V[i, 2]
            # Add a small offset to avoid overlap with the arrow
            offset = 0.1
            ax.text(tip_x + offset, tip_y + offset, tip_z + offset, 
                   v_column_names[i], fontsize=10, color='r')

        # Plot transformed basis vectors
        for i in range(W.shape[0]):
            ax.quiver(*origin, W[i, 0], W[i, 1], W[i, 2], 
                     color='b', label=f'W ({v_column_names[i]})' if i == 0 else "")
            
            # Add text label at the tip of the vector
            tip_x = W[i, 0]
            tip_y = W[i, 1]
            tip_z = W[i, 2]
            # Add a small offset to avoid overlap with the arrow
            offset = 0.1
            ax.text(tip_x + offset, tip_y + offset, tip_z + offset, 
                   v_column_names[i], fontsize=10, color='b')


        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)
        ax.grid(True)
        ax.legend()
        plt.show()


def calculate_perplexity(model, data):
    model.eval()
    with torch.no_grad():
        input_indices = data[:, 0]
        target_indices = data[:, 1]
        logits = model(input_indices)
        loss = F.cross_entropy(logits, target_indices)
        perplexity = torch.exp(loss)
    return perplexity.item()


def sample_greedy(model, chars, max_length=10, custom_model=False):
    indices = [i for i in range(1, len(chars))]
    generated = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    input_seq = torch.tensor(indices, dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_seq)
            next_char = torch.argmax(output, dim=1)
            generated = torch.cat([generated, next_char.unsqueeze(0)], dim=0)
            input_seq = next_char

    words = []
    generated = generated.T.squeeze().tolist()
    for lst in generated:
        word = ''
        for idx in lst:
            if idx == 0:
                break
            word += chars[idx]
        words.append(word)
    return words
        

def create_dataset(dataset_file='data/words_num_sents_10000.txt', num_words=10):

    # load words
    words = []
    with open(dataset_file, "r", encoding="utf-8") as f:
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

    # postprocess data
    inpind2outlst = defaultdict(list)
    for t in data:
        inpind2outlst[t[0]].append(t[1])

    output_indices_all = [i for i in range(len(chars))]

    all_neg_indices_lst = []
    for t in data:
        inp_indx = t[0]
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

    return data, chars, all_target_indices_lst_padded