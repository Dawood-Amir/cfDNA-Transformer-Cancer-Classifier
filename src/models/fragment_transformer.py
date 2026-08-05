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

        pe = torch.zeros(max_len, embed_dim)

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

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


###############################################################
#
# Fragment Transformer
#
###############################################################

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

        self.embedding = nn.Embedding(

            vocab_size,

            embed_dim,

            padding_idx=0

        )

        self.position = PositionalEncoding(
            embed_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=num_heads,

            dim_feedforward=embed_dim * 4,

            dropout=dropout,

            batch_first=True

        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )

        self.embed_dim = embed_dim

    ###########################################################
    # Encode ONE chunk
    ###########################################################

    def encode_chunk(

            self,

            input_ids,

            token_padding_mask=None

    ):

        x = self.embedding(
            input_ids
        )

        x = self.position(
            x
        )

        x = self.transformer(

            x,

            src_key_padding_mask=token_padding_mask

        )

        if token_padding_mask is not None:

            mask = (~token_padding_mask).unsqueeze(-1)

            x = x * mask

            x = x.sum(1)

            x = x / mask.sum(1).clamp(min=1)

        else:

            x = x.mean(1)

        return x

    ###########################################################
    # Automatic chunking
    ###########################################################

    def forward(

            self,

            input_ids,

            token_padding_mask,

            fragment_padding_mask,

            chunk_size=256,

            show_progress=False

    ):

        """
        input_ids

        [B,R,F,T]

        token_padding_mask

        [B,R,F,T]

        fragment_padding_mask

        [B,R,F]

        """

        B, R, F, T = input_ids.shape

        #######################################################
        # Flatten
        #######################################################

        flat_ids = input_ids.view(-1, T)

        flat_token_mask = token_padding_mask.view(-1, T)

        flat_fragment_mask = fragment_padding_mask.view(-1)

        #######################################################
        # Keep only real fragments
        #######################################################

        real_ids = flat_ids[~flat_fragment_mask]

        real_masks = flat_token_mask[~flat_fragment_mask]

        device = input_ids.device

        real_ids = real_ids.to(device)

        real_masks = real_masks.to(device)

        #######################################################
        # Chunk loop
        #######################################################

        outputs = []

        iterator = range(
            0,
            real_ids.size(0),
            chunk_size
        )

        if show_progress:

            from tqdm import tqdm

            iterator = tqdm(

                iterator,

                desc="Fragment Encoder",

                leave=False

            )

        for start in iterator:

            end = start + chunk_size

            chunk_ids = real_ids[start:end]

            chunk_mask = real_masks[start:end]

            emb = self.encode_chunk(

                chunk_ids,

                chunk_mask

            )

            outputs.append(emb)

        fragment_embeddings = torch.cat(

            outputs,

            dim=0

        )

        return fragment_embeddings