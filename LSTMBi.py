import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class RNN_LSTM(nn.Module):
    '''
    Building an RNN_LSTM model 
    '''
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, output_dim, pre_trained_embed_weights=None):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        #Embedding 
        self.embedding_layer = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.embedding_layer.weight.data.copy_(pre_trained_embed_weights)

        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,batch_first = True,bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)

        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        embed = self.embedding_layer(x)
        # packed_embed = pack_padded_sequence(embed,lens.to('cpu'))

        lstm_out,(hidden,_) = self.lstm(embed)
        
        # out,out_len = pack_padded_sequence(packed_out)
        
        hidden_dash = torch.cat((hidden[-2],hidden[-1]),dim=1)

        out = torch.squeeze(self.fc(hidden_dash))
        
        return out
