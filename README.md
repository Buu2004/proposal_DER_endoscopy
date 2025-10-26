# Uncertainty-Aware Monocular Depth Estimation for Endoscopy

This repo provides code for training and testing uncertainty-aware monocular depth estimation models, designed for endoscopic surgical images.
It compares two main approaches for estimating uncertainty:

1. **Deep Evidential Regression (DER)** – The model predicts the parameters of a Normal-Inverse-Gamma (NIG) distribution. 
2. **Monte Carlo (MC) Dropout** – A depth model trained with dropout, which remains active during inference. The uncertainty is calculated as the variance of multiple predictions.

Both methods use a DINOv2 Vision Transformer (ViT) as the backbone, fine-tuned with Low-Rank Adaptation (LoRA) for efficient training.

---

## Main Features

* **Modern Backbone** – Uses the strong DINOv2 transformer model.
* **Efficient Fine-Tuning** – Trains only a small number of parameters using LoRA.
* **Two Uncertainty Estimation Methods** – Supports both Evidential Deep Learning and MC Dropout.
* **Comprehensive Evaluation** – Includes scripts for quantitative metrics (RMSE, AbsRel, $\delta_1$) and qualitative visualizations (depth maps, confidence maps, and error maps).
* **Config-Based Design** – All experiments, parameters, and paths are managed using YAML config files for easy reproducibility.

---

## Project Structure

```
├── configs/
│   ├── evidential.yaml        # Config for training DER
│   ├── mc_dropout.yaml        # Config for training MC Dropout
│   └── analysis.yaml          # Config for evaluation and plots
│
├── experiments/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Quantitative evaluation script
│   └── plot_qualitative.py    # Visualization script
│
├── src/
│   └── uncertainty_depth/
│       ├── __init__.py
│       ├── data.py            # Dataset loader (PyTorch)
│       ├── losses.py          # Evidential loss functions
│       ├── metrics.py         # Depth metrics
│       ├── models/            # Backbone and model definitions
│       └── utils.py           # Helper functions (loading, plotting)
│
├── requirements.txt
└── README.md
```

---

## Usage

The project is controlled by configuration files in the `configs/` folder and executed through scripts in `experiments/`.

---

### 1. Prepare the Dataset

This project uses the **EndoSLAM** dataset.

#### a. Dataset Path Files

* Download from [Kaggle](https://www.kaggle.com/datasets/mcocoz/mde-dataset-path)
* Place the `.txt` files (e.g., `endoslam_train_files_with_gt_Colon.txt`) in a known location.

#### b. EndoSLAM Dataset

* Download the full dataset from [Kaggle](https://www.kaggle.com/datasets/mcocoz/endoslam).
* It includes the endoscopic RGB images and corresponding depth maps.

#### c. Update Config Files

* Open `configs/evidential.yaml`, `configs/mc_dropout.yaml`, and `configs/analysis.yaml`.
* Under the `data:` section, update the paths to where you placed the `.txt` files.
* Make sure that the paths inside the `.txt` files are correct.
  If not, edit `src/uncertainty_depth/data.py` (the code assumes relative paths and removes the first character:
  `img_path.iloc[idx, 0].strip()[1:]`).

---

### 2. Train the Models

Run the training script with the config of your choice:

#### a. Train the Evidential (DER) Model

```bash
python experiments/train.py --config configs/evidential.yaml
```

#### b. Train the MC Dropout Model

```bash
python experiments/train.py --config configs/mc_dropout.yaml
```

Trained weights (`best_model.pth`, `final_model.pth`) will be saved in the `output_dir` defined in your config file
(e.g., `checkpoints/evidential_run_1/`).

---

### 3. Evaluate and Analyze (Quantitative)

After training, update `configs/analysis.yaml` with the path to your saved checkpoints.

Run the evaluation script:

```bash
python experiments/evaluate.py --config configs/analysis.yaml
```

This will:

* Load both trained models.
* Evaluate them on the test dataset.
* Print overall metrics (RMSE, AbsRel, $\delta_1$, etc.).
* Calculate the Spearman correlation between model confidence and RMSE.
* Save a scatter plot (`rmse_vs_confidence.png`) in the output directory.

---

### 4. Visualize Results (Qualitative)

To generate qualitative comparison plots, run:

```bash
python experiments/plot_qualitative.py --config configs/analysis.yaml
```

This will create a folder (e.g., `results/qualitative_plots/`) containing visualizations showing:

* Input Image
* Absolute Error Map
* Confidence Map (Evidence for DER, or 1/Variance for MC Dropout)

---
