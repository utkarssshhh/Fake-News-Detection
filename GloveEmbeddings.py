import torch
import torchtext
import numpy as np

def create_custom_glove_embedding(vocab, dim):
    # Load the pre-trained GloVe vectors (the default cache location will be used)
    glove = torchtext.vocab.GloVe(name='6B', dim=dim)

    # Initialize an embedding matrix filled with zeros
    embedding_matrix = np.zeros((len(vocab), dim))

    # Populate the embedding matrix with GloVe vectors for words in your vocabulary
    for word, idx in vocab.stoi.items():
        if word in glove.stoi:
            embedding_matrix[idx] = glove[word]

    # Create a torch tensor from the embedding matrix
    custom_embedding = torch.tensor(embedding_matrix, dtype=torch.float32)

    # Create an embedding layer in PyTorch using the custom embedding
    custom_embedding_layer = torch.nn.Embedding.from_pretrained(custom_embedding)

    return embedding_matrix
