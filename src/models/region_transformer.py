#regional transformer encoder 

import torch.nn as nn

from models.token_embedding import CFDNATokenEmbedding
from models.positional_encoding import PositionalEncoding

class RegionTransformerEncoder(nn.Module): #embed_dim is d_model
    def __init__(
            self,
            vocab_size,
            embed_dim =128,
            num_heads=4,
            num_layers=4,
            dropout =0.1,
            padding_idx=0
        ):
        
        super().__init__()


        # -------------------------
        # Token embedding
        # -------------------------

        self.embedding = CFDNATokenEmbedding(
            vocab_size=vocab_size ,
            embed_dim=embed_dim,
            padding_idx=padding_idx
            )
        
        # -------------------------
        # Positional encoding
        # -------------------------

        self.position_encoding = PositionalEncoding(
            embed_dim=embed_dim,
        dropout= dropout
        )#skiped tokens max_length here maybe change it later in here or in PositionalEncoding class


        # -------------------------
        # Transformer Encoder
        # -------------------------
        
        #dim_feedforward=embed_dim * 4 means  FFN takes the 128-number
        #token vectors, expands them to 512 numbers to process the context.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward= embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )


        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
             num_layers=num_layers
        )

    

    def forward(self, input_ids,padding_mask=None):
        """
        input_ids:

        Shape:
        (batch, sequence_length)


        padding_mask:

        Shape:
        (batch, sequence_length)

        True = ignore this token
        """

        # Token embedding
        x= self.embedding(input_ids)

        # Add positional encoding
        x = self.position_encoding(x)

        # Transformer
        x= self.transformer(x, src_key_padding_mask=padding_mask)

        # CLS token representation

        region_embedding = x[:,0,:]

        return region_embedding