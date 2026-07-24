import torch 
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    #max_len its the max length of tokens 1 fragment have TODO compute it first instead of 5000 here
    #  embed_dim is 128 which we created with each of 12 in vocab dictionary 
    # PositionalEncoding will have a table with (maxlen ,embed_dim ) that have precomputed sin/cosin 
    # table (pe) and then pull out the exact position that we need then add it to the corosponding vector 
    # from the embeding . 
    # Basically "Positional Encoding adds order to the dense vector that represents the token."
    def __init__(self, embed_dim, dropout =0.1 ,max_len =5000 ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # (max_len, embed_dim)
        pe = torch.zeros(max_len ,embed_dim)

        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embed_dim,
                2
            ).float()
            * (-math.log(10000.0) / embed_dim)
        )

        pe[:,0::2]= torch.sin(position *div_term) # every even column get sin 
        pe[:,1::2]= torch.cos(position*div_term) # every add column get cos 

        # (1, max_len, embed_dim)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe" ,pe)



    def forward(self, x):

        """
        x

        Shape:

        (batch, sequence_length, embed_dim) 
        Batch size (How many fragments are we processing at once?)
        Sequence length (How many tokens are in ONE fragment?)
        Embedding dimension (How many numbers represent each token?)
        """
        #x.size(1) is the number of tokens in your current fragment
        #self.pe is the  positional table 
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)