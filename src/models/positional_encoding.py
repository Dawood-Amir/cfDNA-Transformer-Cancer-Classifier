import torch 
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
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

        pe[:,0::2]= torch.sin(position *div_term)
        pe[:,1::2]= torch.cos(position*div_term)

        # (1, max_len, embed_dim)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe" ,pe)



    def forward(self, x):

        """
        x

        Shape:

        (batch, sequence_length, embed_dim)
        """

        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)