import torch
import torch.nn as nn


class RegionPooling(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(
        self,
        fragment_embeddings,
        patient_ids,
        region_ids
    ):


        """
        fragment_embeddings:

        [num_fragments, embedding_dim]


        patient_ids:

        [num_fragments]


        region_ids:

        [num_fragments]

        """


        device = fragment_embeddings.device


        region_keys = torch.stack(
            [
                patient_ids,
                region_ids
            ],
            dim=1
        )


        unique_regions, inverse = torch.unique(
            region_keys,
            dim=0,
            return_inverse=True
        )


        num_regions = unique_regions.size(0)

        embed_dim = fragment_embeddings.size(1)


        region_embeddings = torch.zeros(
            num_regions,
            embed_dim,
            device=device
        )


        region_embeddings.index_add_(
            0,
            inverse,
            fragment_embeddings
        )


        counts = torch.bincount(
            inverse
        ).unsqueeze(1)


        region_embeddings = (
            region_embeddings / counts
        )


        return (
            region_embeddings,
            unique_regions
        )
#maybe just use the region attention pooling instead of this one, since it is more sophisticated 
# and can capture more information
import torch
import torch.nn as nn


class RegionAttentionPooling(nn.Module):

    """
    Pools fragment embeddings into one embedding per region.

    Input
    -----
    fragment_embeddings : [N_fragments, embed_dim]

    fragment_region_ids : [N_fragments]

    fragment_patient_ids : [N_fragments]

    Output
    ------
    region_embeddings : [N_regions, embed_dim]

    region_mapping : [N_regions, 2]
                     column 0 = patient id
                     column 1 = region id inside patient
    """

    def __init__(self, embed_dim):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(embed_dim, embed_dim),

            nn.Tanh(),

            nn.Linear(embed_dim, 1)

        )

    def forward(

        self,

        fragment_embeddings,

        fragment_region_ids,

        fragment_patient_ids

    ):

        device = fragment_embeddings.device

        unique_regions = torch.unique(fragment_region_ids)

        pooled_regions = []

        region_mapping = []

        for region in unique_regions:

            mask = fragment_region_ids == region

            fragments = fragment_embeddings[mask]

            ####################################
            # Attention weights
            ####################################

            scores = self.attention(fragments)

            weights = torch.softmax(scores, dim=0)

            ####################################
            # Weighted sum
            ####################################

            pooled = (weights * fragments).sum(dim=0)

            pooled_regions.append(pooled)

            ####################################
            # Mapping
            ####################################

            patient = fragment_patient_ids[mask][0]

            region_mapping.append(

                torch.tensor(

                    [

                        patient,

                        region

                    ],

                    device=device

                )

            )

        region_embeddings = torch.stack(

            pooled_regions,

            dim=0

        )

        region_mapping = torch.stack(

            region_mapping,

            dim=0

        )

        return region_embeddings, region_mapping