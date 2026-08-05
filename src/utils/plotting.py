import matplotlib.pyplot as plt



def plot_training(
        train_losses,
        val_losses,
        save_path
):


    plt.figure(figsize=(8,5))


    plt.plot(
        train_losses,
        label="Train Loss"
    )


    plt.plot(
        val_losses,
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )


    plt.legend()


    plt.grid()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()