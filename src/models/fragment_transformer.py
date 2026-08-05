import torch.nn as nn

from models.positional_encoding import PositionalEncoding



class FragmentTransformer(nn.Module):


    def __init__(
            self,
            vocab_size,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1
    ):

        super().__init__()


        #################################
        # Token embedding
        #################################

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )


        #################################
        # Position encoding
        #################################

        self.position = PositionalEncoding(
            embed_dim
        )


        #################################
        # Transformer
        #################################

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=num_heads,

            dim_feedforward=embed_dim*4,

            dropout=dropout,

            batch_first=True

        )


        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )


    def forward(
            self,
            input_ids,
            padding_mask=None
    ):


        """
        input_ids:

        (N_fragments, tokens)


        Example:

        (5540,334)

        """


        x = self.embedding(
            input_ids
        )


        x = self.position(
            x
        )


        x = self.transformer(

            x,

            src_key_padding_mask=padding_mask

        )


        #################################
        # Mean pooling
        #################################

        if padding_mask is not None:


            mask = (~padding_mask).unsqueeze(-1)


            x = x * mask


            x = x.sum(1) / mask.sum(1)



        else:

            x = x.mean(1)



        return x