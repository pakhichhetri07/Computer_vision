# Pneumonia Detection Using InceptionV3 on PneumoniaMNIST (Computer Vision)
This project fine-tunes a **Inception-V3** model to distinguish pneumonia to classify chest X-ray images as **normal** or **pneumonia**

Dataset Used: [PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data?select=pneumoniamnist.npz)

Download and place `pneumoniamnist.npz` in the root directory or `/content/` if using Colab.

---

## 🚀 Features

- Transfer learning with **InceptionV3** from `torchvision.models`
- **Data augmentation** (random flip, rotation, brightness/contrast)
- **Class imbalance handling** via `WeightedRandomSampler`
- Evaluation using **Accuracy**, **F1-Score**, and **AUROC**
- Model checkpointing and **best model selection**
- Modular code with `evaluate()` function

---

##  Dependencies

Install all dependencies using:

```bash
pip install -r requirements.txt

## Clone the repository (or open in Colab)
git clone https://github.com/yourusername/pneumonia-inceptionv3.git
cd pneumonia-inceptionv3


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



