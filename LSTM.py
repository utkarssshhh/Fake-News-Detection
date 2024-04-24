import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_LSTM(nn.Module):
    '''
    Building an RNN_LSTM model 
    '''
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, output_dim, pre_trained_embed_weights=False, pre_trained_embed_model = None):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        #Embedding 
        if pre_trained_embed_weights == False:
            self.embedding_layer = nn.Embedding(num_embeddings=input_dim+1, embedding_dim=embedding_dim)
        else:
            self.embedding_layer = nn.Embedding.from_pretrained(embeddings=torch.FloatTensor(pre_trained_embed_model),freeze= False)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        embed = self.embedding_layer(x)

        out,(hidden,_) = self.lstm(embed)

        flattened = hidden[-1]

        out = torch.squeeze(self.fc(flattened))
        
        return out
