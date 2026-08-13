import torch
import torch.nn as nn
import math



class PositionalEncoding(nn.Module):


    def __init__(
            self,
            embed_dim,
            max_len=512
    ):

        super().__init__()


        pe = torch.zeros(
            max_len,
            embed_dim
        )


        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)


        div_term = torch.exp(

            torch.arange(
                0,
                embed_dim,
                2
            )
            *
            (-math.log(10000.0) / embed_dim)

        )


        pe[:,0::2] = torch.sin(
            position * div_term
        )


        pe[:,1::2] = torch.cos(
            position * div_term
        )


        pe = pe.unsqueeze(0)


        self.register_buffer(
            "pe",
            pe
        )


    def forward(self,x):

        """
        x:

        [batch, regions, embed_dim]

        """

        return x + self.pe[:,:x.size(1)]


    

class PatientTransformer(nn.Module):


    def __init__(
            self,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.15
            
    ):

        super().__init__()



        ###################################
        # Region positional encoding
        ###################################

        self.position = PositionalEncoding(
            embed_dim,
            max_len=512
        )



        ###################################
        # Transformer Encoder
        ###################################

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=num_heads,

            dim_feedforward=embed_dim*4,

            dropout=dropout,

            batch_first=True

        )


        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )



        ###################################
        # Layer normalization
        ###################################

        self.norm = nn.LayerNorm(
            embed_dim
        )




    def forward(
            self,
            region_embeddings,
            region_padding_mask
    ):


        """
        region_embeddings:

            [batch, regions, 128]


        region_padding_mask:

            [batch, regions]

            True = padding

        """


        ###################################
        # Add region positions
        ###################################

        x = self.position(
            region_embeddings
        )



        ###################################
        # Transformer
        ###################################

        x = self.encoder(

            x,

            src_key_padding_mask=
                region_padding_mask

        )



        x = self.norm(x)



        ###################################
        # Patient pooling
        ###################################

        mask = (
            ~region_padding_mask
        ).unsqueeze(-1)



        x = x * mask



        patient_embedding = (

            x.sum(dim=1)

            /

            mask.sum(dim=1).clamp(min=1)

        )



        return patient_embedding