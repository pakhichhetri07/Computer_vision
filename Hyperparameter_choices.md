# Hyperparameter Configuration

This file documents all the hyperparameter settings used in the pneumonia detection project using InceptionV3 and PneumoniaMNIST.

---

## Model Configuration

| Parameter        | Value                              |
|------------------|-------------------------------------|
| Model            | InceptionV3                        |
| Pretrained       | True                               |
| Aux Logits       | True                               |
| Input Size       | 299 x 299                          |
| Number of Classes| 2                                  |

---

## Training Settings

| Parameter         | Value             |
|-------------------|------------------|
| Optimizer         | Adam             |
| Learning Rate     | 1e-4             |
| Loss Function     | CrossEntropyLoss |
| Batch Size        | 32               |
| Epochs (Max)      | 20               |
| Early Stopping    | Yes              |
| Patience          | 5 epochs         |
| Evaluation Metric | Validation AUROC |

---

## Evaluation Metrics

| Metric      | Description                                          |
|-------------|------------------------------------------------------|
| Accuracy    | Overall correct predictions                         |
| F1-Score    | Harmonic mean of precision and recall (class balance)|
| AUROC       | Area under ROC curve (threshold-independent)        |

---

## Cross-Validation

| Parameter         | Value              |
|-------------------|-------------------|
| Type              | StratifiedKFold   |
| Folds             | 5                 |
| Shuffle           | True              |
| Random State      | 42                |

---

## Data Augmentation (Training)

- Resize to 299 × 299
- Grayscale → 3 Channels
- Random Horizontal Flip
- Random Rotation (±10°)
- ToTensor()

---

##  Class Imbalance Handling

| Method                 | Description                                      |
|------------------------|--------------------------------------------------|
| WeightedRandomSampler  | Oversamples minority class during training       |


---

## Best Model Selection Criteria

- Based on **highest validation AUROC**
- Best weights saved for each fold

---

