import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)



def calculate_metrics(
        labels,
        predictions,
        probabilities,
        num_classes=4
):

    labels = np.array(labels)

    predictions = np.array(predictions)


    results = {}


    results["accuracy"] = accuracy_score(
        labels,
        predictions
    )


    results["macro_f1"] = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )


    try:

        results["roc_auc"] = roc_auc_score(

            labels,

            probabilities,

            multi_class="ovr"

        )

    except:

        results["roc_auc"] = 0.0



    print("\n==============================")
    print("Classification Report")
    print("==============================")

    print(
        classification_report(
            labels,
            predictions,
            zero_division=0
        )
    )



    print("==============================")
    print("Confusion Matrix")
    print("==============================")


    cm = confusion_matrix(
        labels,
        predictions
    )


    print(cm)


    return results, cm