import torch


class EarlyStopping:


    def __init__(
        self,
        patience=10,
        min_delta=0.001
    ):

        self.patience = patience

        self.min_delta = min_delta

        self.counter = 0

        self.best_loss = None

        self.stop = False



    def __call__(
        self,
        val_loss
    ):


        if self.best_loss is None:

            self.best_loss = val_loss



        elif val_loss > self.best_loss - self.min_delta:

            self.counter += 1


            print(
                f"Early stopping counter: {self.counter}/{self.patience}"
            )


            if self.counter >= self.patience:

                self.stop=True



        else:

            self.best_loss = val_loss

            self.counter = 0



        return self.stop