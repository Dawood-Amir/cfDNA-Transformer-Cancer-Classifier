import torch
import torch.nn as nn


class CFDNATokenEmbedding(nn.Module): # embed_dim here is our d_model
    def __init__(self, vocab_size, embed_dim =128 ,padding_idx=0 ):
        super().__init__()


        '''
         # Lookup table creted here. PyTorch creates one matrix of shape: (vocab_size, embedding_dim) 
         # During the embedding layer, the only trainable parameters are the embedding lookup table 
         # which gets updated in the backward pass.
        
        '''
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,embedding_dim=embed_dim ,padding_idx=padding_idx
        )







    
    '''
    The forward pass expects x of shape (batch, seq_len). 

    self.embedding(x) does a lookup: for each position in the batch, it takes the row index x[b, t] 
    from the embedding weight matrix of shape (vocab_size, embed_dim).

    Output shape: (batch, seq_len, embed_dim).
    '''

    def forward(self,x):
        """
        x:
        token ids

        shape:
        (batch, sequence_length)

        output:

        (batch, sequence_length, embed_dim) or batch, seq_len, d_model
        """
        return self.embedding(x)