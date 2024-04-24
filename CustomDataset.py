import torch
import torch.nn as nn
from torch.utils.data import Dataset
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence

class TextClassificationDataset(Dataset):
    def __init__(self, data, categories, training_vocab = None, max_length = 100, min_freq = 5):
        
        self.data = data
        self.max_length = max_length
        
        # Allow to import a vocabulary (validation and testing will use the training vocabulary)
        if training_vocab is not None:
            self.word2idx, self.idx2word = training_vocab
        else:
            # Build the vocabulary if none is imported
            self.word2idx, self.idx2word = self.build_vocab(self.data, min_freq)
        
        # We tokenize the articles
        tokenized_data = [word_tokenize(file.lower()) for file in self.data]
        # Transform words into lists of indexes
        indexed_data = [[self.word2idx.get(word, self.word2idx['UNK']) for word in file] for file in tokenized_data]
        # Transform into a list of Pytorch LongTensors
        tensor_data = [torch.LongTensor(file) for file in indexed_data]
        # Lables are passed into a FloatTensor
        tensor_y = torch.FloatTensor(categories)
        # Finally we cut too the determined maximum length
        cut_tensor_data = [tensor[:max_length] for tensor in tensor_data]
        # We pad the sequences to have the whole dataset containing sequences of the same length
        self.tensor_data = pad_sequence(cut_tensor_data, batch_first=True, padding_value=0)
        self.tensor_y = tensor_y
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.tensor_data[idx], self.tensor_y[idx] 
    
    def build_vocab(self, corpus, count_threshold):
        word_counts = {}
        for sent in corpus:
            for word in word_tokenize(sent.lower()):
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1   
        filtered_word_counts = {word: count for word, count in word_counts.items() if count >= count_threshold}        
        words = sorted(filtered_word_counts.keys(), key=word_counts.get, reverse=True) + ['UNK']
        word_index = {words[i] : (i+1) for i in range(len(words))}
        idx_word = {(i+1) : words[i] for i in range(len(words))}
        return word_index, idx_word
    
    def get_vocab(self):
        return self.word2idx, self.idx2word