# Pneumonia Detection Using InceptionV3 on PneumoniaMNIST (Computer Vision)
This project fine-tunes a **Inception-V3** model to distinguish pneumonia to classify chest X-ray images as **normal** or **pneumonia**

Dataset Used: [PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data?select=pneumoniamnist.npz)

Download and place `pneumoniamnist.npz` in the root directory or `/content/` if using Colab.

---

##  Features

- **Transfer learning** with InceptionV3 from torchvision.models
- Image preprocessing and **augmentation** (resize, grayscale to 3-channel, random flip, rotation)
- **Class imbalance** handling using WeightedRandomSampler
- 5-Fold **Cross Validation** for robust evaluation
- **Early stopping** based on validation AUROC
- **Evaluation** using Accuracy, F1-Score, and AUROC
- **Confusion matrix** plotted for each fold
- **Modular training** with evaluate() function for validation and testing

---

##  Dependencies

Install all dependencies using:

```bash
pip install -r requirements.txt

## Clone the repository (or open in Colab)
git clone https://github.com/pakhichhetri07/Computer_vision.git


Download the Dataset via kaggle [PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data?select=pneumoniamnist.npz) dataset 


## Train the Model
python train.py


## Folder structure 
.
├── pneumoniamnist.npz
├── inceptionv3_pneumonia.ipynb
├── requirements.txt
├── README.md
└── train.py

#  Training Accuracy

| Fold | Epochs | Max Val Acc | Max F1 | Max AUROC |
| ---- | ------ | ----------- | ------ | --------- |
| 1    | 8      | 0.9683      | 0.9823 | 0.9151    |
| 2    | 9      | 0.9671      | 0.9811 | 0.9451    |
| 3    | 10     | 0.9637      | 0.9792 | 0.9515    |
| 4    | 10     | 0.9739      | 0.9852 | 0.9699    |
| 5    | 10     | 0.9762      | 0.9865 | 0.9675    |

#  Final Test score
Accuracy: 88.78%
F1 Score: 0.9157
AUROC: 0.8590







