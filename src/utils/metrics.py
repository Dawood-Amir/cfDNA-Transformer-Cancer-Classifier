from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred, y_probs):

    num_classes = len(y_probs[0])

    y_true_bin = label_binarize(
        y_true,
        classes=list(range(num_classes))
    )

    auc = roc_auc_score(
        y_true_bin,
        y_probs,
        average="macro",
        multi_class="ovr"
    )

    return {

        "confusion_matrix":
            confusion_matrix(y_true, y_pred),

        "classification_report":
            classification_report(y_true, y_pred),

        "auc":
            auc
    }