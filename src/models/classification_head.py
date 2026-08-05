import torch
import torch.nn as nn


class ClassificationHead(nn.Module):

    def __init__(
            self,
            embed_dim=128,
            num_classes=4,
            dropout=0.2
    ):

        super().__init__()


        self.classifier = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embed_dim,
                num_classes
            )

        )


    def forward(self, x):

        """
        x:

        Patient representation

        Shape:

        [batch, embed_dim]

        """

        logits = self.classifier(x)

        return logits