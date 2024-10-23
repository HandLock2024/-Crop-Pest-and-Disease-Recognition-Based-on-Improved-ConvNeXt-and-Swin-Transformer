# Crop-Pest-and-Disease-Recognition-Based-on-Improved-ConvNeXt-and-Swin-Transformer
This repository implements the method proposed in the paper "Crop Pest and Disease Recognition Based on Improved ConvNeXt and Swin Transformer". The project combines ConvNeXt and Swin Transformer architectures with the CAFM fusion module to enhance the accuracy of pest and disease recognition. The Mixup technique is used for data augmentation, and Equalization Loss is applied to address class imbalance in the IP102 dataset.
## Project Overview
The model leverages ConvNeXt's ability to capture local features and Swin Transformer's strength in global feature extraction. The CAFM fusion module enhances the feature fusion process, leading to improved pest and disease classification performance.
## Key Features:
ConvNeXt for fine-grained local feature extraction.
Swin Transformer for capturing global features.
CAFM (Channel Attention and Fusion Module) for efficient feature fusion.
Mixup for better generalization and prevention of overfitting.
Equalization Loss to handle class imbalance issues.
## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/HandLock2024/ConvNeXt-and-Swin-Transformer-with-CAFM.git
   cd crop-pest-disease-recognition
2. Create a virtual environment (optional but recommended):   
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Make sure to download the necessary datasets
## Dataset
This project uses the IP102 dataset for pest recognition and the PlantVillage dataset for plant disease detection.
### IP102 Dataset:
You can download the IP102 dataset from [here](https://aistudio.baidu.com/datasetdetail/245442). After downloading, extract the dataset into the data/ directory as follows:
   ```bash
   data/
   ├── IP102/
     ├── train/
     ├── val/
     └── test/
  ```
### PlantVillage
The PlantVillage dataset can be downloaded from [this link](https://www.tensorflow.org/datasets/catalog/plant_village). After downloading, extract it into the data/ directory.
### Usage
## Train
To train the model on the IP102 dataset, use the following command:
  ```bash
  python train.py --dataset data/IP102 --epochs 50 --batch_size 32
  ```
## Test
To evaluate the model on the test dataset:
  ```bash
  python test.py --model checkpoints/best_model_weights.pth --dataset data/IP102/test
  ```






