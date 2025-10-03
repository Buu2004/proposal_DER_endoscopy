# Deep Endoscopy Depth Estimation with Evidential Learning

## Project Description

This project helps computers understand depth in surgical images. It uses a special AI model that can predict how far objects are in images. The model also knows when it is not sure about its predictions. This is very important for surgery where doctors need reliable information.

## Main Features

- **Depth Prediction**: Shows distance of objects in images
- **Uncertainty Measurement**: Knows when predictions are not certain
- **Safe for Medical Use**: Good for healthcare applications
- **Fast Training**: Uses efficient training methods

## Installation

1. **Clone the repository**:

2. **Install requirements**:
```bash
pip install -r requirements.txt
```

## Usage

### Prepare Your Data

Download the datasets from these sources:

1. **Dataset Path Files**:
   - Download from: https://www.kaggle.com/datasets/mcocoz/mde-dataset-path
   - Files needed: `endoslam_train_files_with_gt_Colon.txt` and `endoslam_test_files_with_gt_Colon.txt`

2. **EndoSLAM Dataset**:
   - Download from: https://www.kaggle.com/datasets/mcocoz/endoslam
   - This contains the actual images and depth maps

### Train the Model

```bash
python train.py
```

### Check Results

- View training progress with TensorBoard
- Find trained models in the 'checkpoints' folder

## Project Structure

```
proposal_DER_endoscopy/
├── models/          # AI model components
├── loss/           # Training loss functions
├── data/           # Data loading
├── metrics/          # Metric functions
└── train.py        # Main training script
```