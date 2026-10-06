# 🧬 cfDNA Hierarchical Transformer for Cancer Classification

A deep learning pipeline for classifying cancer subtypes from cell-free DNA (cfDNA) fragment data using a **hierarchical Transformer architecture**.

---

## 📖 Overview

Liquid biopsy analysis via cell-free DNA (cfDNA) is a non-invasive method for cancer detection and subtyping. This repository implements a **hierarchical Transformer** that processes cfDNA data at multiple levels:

1. **Fragment-level**: Token sequences of methylation patterns (`<m>`, `<um>`)
2. **Region-level**: Fragment embeddings aggregated within genomic regions
3. **Patient-level**: Region embeddings processed as a patient-level sequence

### Key Findings

- A compact hierarchical Transformer achieves **59.88% accuracy** and **0.7723 Macro ROC-AUC** on 3-class classification (Healthy / GBM / LGG).
- `WeightedRandomSampler` was used to address class imbalance, while **label smoothing (0.07)** was used in the classification loss.
- **GBM ↔ LGG confusion** remains the primary source of error.

> **Project Status:** Experimental / Research  
> This project is intended for research and educational purposes only and is **not intended for clinical diagnosis or medical decision-making**.

---

## 📊 Dataset

The model uses synthetic cfDNA fragment data originally provided through Kaggle.

| Property | Value |
| :--- | :--- |
| Source | Synthetic cfDNA fragment data (Kaggle) |
| Original Format | JSON files containing fragment-level methylation tokens |
| Processed Format | PyTorch `.pt` files containing patient-level data |
| Classes | Healthy (0), GBM (1), LGG (2) |
| Total Patients | 1,640 (after removing DMG_H3K27M) |
| Train / Validation Split | 1,311 / 329 (stratified) |

### Processed Dataset

The processed patient-level `.pt` files used by the final V4 model are available as a Kaggle dataset.

**Kaggle Dataset:**  
`dawoodaamar/cfdna-patient-tensors`

The processed dataset contains the patient-level files required to train the model.

> **Note:** The original dataset contained a fourth class (`DMG_H3K27M`) which was removed due to insufficient samples and poor model performance.

### Dataset Distribution

| Class | Train | Validation | Total |
| :--- | :--- | :--- | :--- |
| Healthy | 524 | 132 | 656 |
| GBM | 360 | 90 | 450 |
| LGG | 427 | 107 | 534 |

---

## 🏗️ Architecture

### Hierarchical Design

```text
Raw cfDNA Fragments
(tokens: <m>, <um>, ...)
        │
        ▼
┌─────────────────────────────┐
│  Tokenization               │
│  Convert tokens → IDs       │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Token Embedding +          │
│  Positional Encoding        │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Fragment Transformer       │
│  Token-level attention      │
│  2 Transformer layers       │
└─────────────────────────────┘
        │
        ▼
   Fragment embeddings
        │
        ▼
┌─────────────────────────────┐
│  Region Mean Pooling        │
│  Aggregate fragment         │
│  embeddings within regions  │
└─────────────────────────────┘
        │
        ▼
   Region embeddings
        │
        ▼
┌─────────────────────────────┐
│  Region Batch Builder       │
│  Regions → patient          │
│  sequences + masks          │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Patient Transformer        │
│  Region-level attention     │
│  2 Transformer layers       │
└─────────────────────────────┘
        │
        ▼
   Patient embedding
        │
        ▼
┌─────────────────────────────┐
│  Classification Head        │
│  Healthy / GBM / LGG        │
└─────────────────────────────┘
        │
        ▼
   3-Class Prediction
```

### Implementation Mapping

| Architecture Component | Implementation |
| :--- | :--- |
| Tokenization | `tokenizer.py` |
| Token Embedding | `token_embedding.py` |
| Positional Encoding | `positional_encoding.py` |
| Fragment Encoder | `fragment_transformer.py` |
| Region Mean Pooling | `region_pooling.py` |
| Region Batching | `region_batch_builder.py` |
| Patient Encoder | `patient_transformer.py` |
| Classification | `classification_head.py` |
| Full Model | `cfdna_transformer.py` |
| Training Pipeline | `train_patient_transformer.py` |
| Batch Collation | `patient_collate.py` |

### Model Configuration

| Component | Value |
| :--- | :--- |
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Fragment Transformer Layers | 2 |
| Patient Transformer Layers | 2 |
| Fragment Transformer Dropout | 0.15 |
| Patient Transformer Dropout | 0.15 |
| Classification Head Dropout | 0.20 |
| Region Pooling | Mean Pooling |
| Total Parameters | 811,779 |

---

## 🎯 Training Configuration

| Parameter | Value |
| :--- | :--- |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 0.01 |
| Learning Rate Scheduler | CosineAnnealingLR |
| Batch Size | 1 |
| Gradient Accumulation | 4 (effective batch = 4) |
| Epochs | 20 (early stopped at 14) |
| Early Stopping Patience | 7 |
| Loss Function | CrossEntropyLoss with Label Smoothing (0.07) |
| Class Imbalance | WeightedRandomSampler |
| Mixed Precision | AMP (CUDA) |
| Best Metric | Macro F1 |

### Class Imbalance Handling

`WeightedRandomSampler` was used to address class imbalance. No class weights were applied inside the cross-entropy loss, avoiding double correction.

The final V4 sampling weights were:

| Class | Sampling Weight |
| :--- | :--- |
| Healthy | 0.002424 (boosted ×1.27) |
| GBM | 0.002639 (reduced ×0.95) |
| LGG | 0.002342 |

Separately, **label smoothing (0.07)** was applied to the cross-entropy classification loss.

This sampler-only approach produced the most stable results among the tested class-imbalance strategies.

---

## 📈 Results

### Final Performance (Best Model - Epoch 7)

| Metric | Value |
| :--- | :--- |
| Accuracy | 59.88% |
| Balanced Accuracy | 59.00% |
| Macro F1 | 0.5931 |
| Weighted F1 | 0.6081 |
| Macro ROC-AUC | 0.7723 |
| Weighted ROC-AUC | 0.7819 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Healthy | 0.7311 | 0.6591 | 0.6932 | 132 |
| GBM | 0.3950 | 0.5222 | 0.4498 | 90 |
| LGG | 0.6923 | 0.5888 | 0.6364 | 107 |

### Confusion Matrix

```text
              Predicted
           Healthy  GBM   LGG
Actual  Healthy   87    34    11
        GBM       26    47    17
        LGG        6    38    63
```

### Prediction Distribution

| Class | Predicted | Actual |
| :--- | :--- | :--- |
| Healthy | 119 | 132 |
| GBM | 119 | 90 |
| LGG | 91 | 107 |

### Key Observations

- **Healthy detection** is strong (F1 = 0.69).
- **GBM precision (0.40)** is the main limiting factor, with the model over-predicting GBM.
- **LGG recall (0.59)** shows room for improvement, with 38 LGG patients misclassified as GBM.
- **GBM ↔ LGG confusion** accounts for a substantial portion of the classification errors.

---

## 📁 Repository Structure

```text
cfDNA-Transformer-Cancer-Classifier/
├── src/
│   ├── data/
│   │   ├── patient_dataset.py
│   │   └── patient_collate.py
│   ├── models/
│   │   ├── tokenizer.py
│   │   ├── token_embedding.py
│   │   ├── positional_encoding.py
│   │   ├── cfdna_transformer.py
│   │   ├── fragment_transformer.py
│   │   ├── patient_transformer.py
│   │   ├── region_pooling.py
│   │   ├── region_attention_pooling.py
│   │   ├── classification_head.py
│   │   └── region_batch_builder.py
│   ├── utils/
│   │   ├── utils.py
│   │   └── ...
│   └── train_patient_transformer.py
│
├── training_outputs_3classes_v4/
│   ├── checkpoints/
│   ├── metrics/
│   ├── logs/
│   └── tokenizer_vocab.json
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements

The main dependencies are:

- PyTorch >= 2.0
- NumPy >= 1.24
- Pandas >= 2.0
- scikit-learn >= 1.3
- tqdm >= 4.65
- Matplotlib >= 3.7
- Seaborn >= 0.12

### 2. Prepare Data

Download the processed patient-level dataset from the Kaggle dataset listed above.

After downloading the dataset, the processed data should be organized as:

```text
data/processed/patient_tensors_3classes/
├── manifest.csv
├── patient_0001.pt
├── patient_0002.pt
└── ...
```

Each `.pt` file contains patient-level data including genomic regions, fragments, methylation tokens, and the corresponding class label.

Example structure:

```python
{
    "regions": [
        {
            "fragments": [
                {
                    "tokens": ["<m>", "<um>", ...],
                    "length": 150
                },
                ...
            ]
        },
        ...
    ],
    "label": 0  # 0=Healthy, 1=GBM, 2=LGG
}
```

> **Note:** The processed `.pt` dataset is not included directly in this GitHub repository. It is provided separately through Kaggle.

### 3. Train the Model

```bash
python src/train_patient_transformer.py
```

### 4. Evaluate

The training script automatically:

- Saves the best model (`best_model.pth`)
- Evaluates on the validation set
- Exports metrics to JSON

---

## 🔄 Version History

Several training configurations were explored before settling on the final V4 model.

| Version | Key Changes | Accuracy | Macro F1 | Macro AUC |
| :--- | :--- | :--- | :--- | :--- |
| V1 | Baseline Transformer | 56.40% | 0.5667 | 0.7587 |
| V2 | + Class weights + Oversampling | 58.23% | 0.5739 | 0.7779 |
| V3 | + Sqrt weights + Stratified split | 58.66% | 0.5827 | 0.7757 |
| V4 | + Balanced sampler + Label smoothing | 59.88% | 0.5931 | 0.7723 |

**V4 is the final model** included in this repository.

---

## 🔬 Discussion

### Why the Hierarchical Approach?

Traditional feature-engineering approaches aggregate fragments into statistical features such as mean fragment length and methylation ratio. While useful, these approaches can discard:

- **Sequential information** in methylation token sequences
- **Relationships between fragments** within a genomic region
- **Relationships between genomic regions** within a patient

The hierarchical Transformer instead learns representations at multiple levels:

**tokens → fragments → regions → patients**

This allows attention to be applied at both the fragment and patient/region levels.

### Challenges

1. **GBM vs LGG**: These are both glioma subtypes and remain difficult to distinguish using the available cfDNA representations.
2. **Data scarcity**: With approximately 1,600 patients, the dataset is relatively small compared with datasets used to train large genomic foundation models.
3. **Class imbalance**: GBM is the minority class and requires careful sampling during training.

### Limitations

- The dataset is synthetic and may not capture the complexity and variability of real-world cfDNA samples.
- Evaluation is based on a stratified train/validation split rather than an independent external test cohort.
- The relatively small dataset limits the ability to reliably train substantially larger Transformer architectures.
- GBM and LGG remain difficult to distinguish, resulting in substantial cross-class confusion.
- The reported results should not be interpreted as clinical diagnostic performance.

### Future Directions

- **Binary classification**: Healthy vs Cancer (GBM + LGG)
- **Multi-modal fusion**: Combine token-level sequences with engineered fragmentomic features
- **Foundation model fine-tuning**: Explore DNABERT-2 or similar pre-trained genomic models
- **Multiple-Instance Learning (MIL)**: Treat fragments as instances and learn patient-level aggregation
- **Ensemble methods**: Combine Transformer, XGBoost, and FFN predictions

---

## 📄 License

This project is for research and educational purposes only. Not intended for clinical use.

---

## 🙏 Acknowledgments

- Dataset provided via Kaggle (synthetic cfDNA)
- Architecture inspired by hierarchical attention models in genomics
- Built with PyTorch

---

## 📧 Contact

For questions or collaboration, please open an issue or email @ dawoodamir115@gmail.com.
