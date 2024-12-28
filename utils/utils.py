import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from collections import defaultdict
import random

def plot_matrix_bases(matrices, colors, legends, chars, limits=(-3, 3)):
    """
    Plots multiple sets of basis vectors with different colors and labels.
    
    Args:
        matrices: List of matrices to plot
        colors: List of colors for each matrix
        legends: List of legend labels for each matrix
        chars: List of names for the vectors (shared across matrices)
        limits: Tuple of (min, max) for axis limits
    """
    # Convert all matrices to numpy arrays
    matrices = [m.detach().cpu().numpy() if hasattr(m, 'detach') else m for m in matrices]
    
    # Determine the dimensionality from the first matrix
    dim = matrices[0].shape[1]

    if dim == 2:
        # 2D plotting
        fig, ax = plt.subplots()
        
        # Plot each matrix's vectors
        for matrix_idx, (matrix, color, legend) in enumerate(zip(matrices, colors, legends)):
            origin = np.zeros((2, matrix.shape[1]))  # Origin points for current matrix
            
            # Plot vectors
            for vec_idx in range(matrix.shape[1]):
                q = ax.quiver(origin[0, vec_idx], origin[1, vec_idx],
                            matrix[0, vec_idx], matrix[1, vec_idx],
                            angles='xy', scale_units='xy', scale=1,
                            color=color, label=legend if vec_idx == 0 else "")
                
                # Add text label at vector tip
                tip_x = matrix[0, vec_idx]
                tip_y = matrix[1, vec_idx]
                offset = 0.1
                # Only add character labels if they're provided and valid
                if chars and vec_idx < len(chars):
                    ax.text(tip_x + offset, tip_y + offset, chars[vec_idx],
                           fontsize=10, color=color)

        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        plt.show()

    else:
        # Handle higher dimensions
        if dim > 3:
            # Apply PCA to reduce to 3D
            pca = PCA(n_components=3)
            matrices = [pca.fit_transform(m.T).T for m in matrices]

        # 3D plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        origin = np.zeros(3)

        # Plot each matrix's vectors
        for matrix_idx, (matrix, color, legend) in enumerate(zip(matrices, colors, legends)):
            for vec_idx in range(matrix.shape[0]):
                ax.quiver(*origin, matrix[vec_idx, 0], matrix[vec_idx, 1], matrix[vec_idx, 2],
                         color=color, label=legend if vec_idx == 0 else "")
                
                # Add text label at vector tip
                tip_x = matrix[vec_idx, 0]
                tip_y = matrix[vec_idx, 1]
                tip_z = matrix[vec_idx, 2]
                offset = 0.1
                # Only add character labels if they're provided and valid
                if chars and vec_idx < len(chars):
                    ax.text(tip_x + offset, tip_y + offset, tip_z + offset,
                           chars[vec_idx], fontsize=10, color=color)

        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)
        ax.grid(True)
        ax.legend()
        plt.show()


def calculate_perplexity(model, input_indices, target_indices):
    model.eval()
    with torch.no_grad():
        logits = model(input_indices)
        loss = F.cross_entropy(logits, target_indices)
        perplexity = torch.exp(loss)
    return perplexity.item()


def sample_greedy(model, chars, max_length=10, custom_model=False, ngrm_num=1):
    indices = [[0, i] for i in range(2, len(chars))]

    generated = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    input_seq = torch.tensor(indices, dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_seq)
            next_char = torch.argmax(output, dim=1)
            generated = torch.cat([generated.squeeze(0), next_char.unsqueeze(1)], dim=1)
            input_seq = generated[:,-ngrm_num:]

    words = []
    generated = generated.tolist()
    for lst in generated:
        word = ''
        for idx in lst:
            if idx == 1:
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


def create_ngram_dataset(num_words=2, ngrm_num=2, dataset_file='data/words_num_sents_10000.txt'):

    # load words
    words = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # split to words, add end or word character to each word, skip the word if it contains a character not in the 26-letter alphabet, that is words containing characters with accents, umlauts, etc. are skipped

            words.extend(['#'+word + '*' for word in line.strip().split() if all(char in "abcdefghijklmnopqrstuvwxyz" for char in word)])
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
        indices = [char2idx[char] for char in word]
        for i in range(len(indices) - ngrm_num):
            data.append(indices[i:i+ngrm_num+1])

    return data, chars