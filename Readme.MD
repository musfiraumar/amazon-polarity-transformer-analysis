This is the Amazon polarity sentiment analysis using NLP.

## Project Overview

In this project, DistilBERT is fine-tuned to the Amazon Polarity dataset, a binary sentiment dataset.
Once it has been classified, it then goes through the process of analysis of the model through:

The ability to switch attention to different visualizations of layers (layer 0 & layer 5, multiple heads)
- 20 predictions, with explanations of the model's response to each factor using SHAP values.
Predictions that are limited to 20 were explained using lime.Lime was used for explanations of 20 predictions.
This paper is a comparative study of the faithfulness, stability and runtime of SHAP and LIME.This paper compares SHAP and LIME in terms of faithfulness, stability and runtime.
- Error analysis analysis of misclassified samples

---

## Hardware & Software Environment

| Component | Specification |
|-----------|--------------|
| CPU       | Any modern multi-core (Intel/AMD) |
GPU       | NVIDIA GPU (CUDA 11.8+): Recommended; CPU training is also possible |
| RAM       | ≥ 16 GB recommended |
| Python    | 3.10 or 3.11 |
| PyTorch   | 2.2+ |
| Transformers | 4.40+ |
| Datasets  | 2.19+ |
| SHAP      | 0.45+ |
| LIME      | 0.2.0.1 |

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/transformer-nlp-assignment.git
cd transformer-nlp-assignment
```

### 2. To create a Virtual Environment, use the following command:

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For **GPU (CUDA 11.8)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For **CPU only**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Next, fit the remaining:
```bash
pip install transformers datasets shap lime scikit-learn matplotlib seaborn bertviz
```

### 4. Dataset Preparation

The data is automatically downloaded from HuggingFace on first run:
```python
from datasets import load_dataset
raw = load_dataset("fancyzhx/amazon_polarity")
```

This will need approximately 1,400 Mb of disk space. The data will be saved in the `~/.cache/huggingface/`.

You don't need to download any manual.

---

## Running the Notebook

You can use Jupyter Lab / Notebook as an alternative option.Jupyter Lab / Notebook is an alternate option.

```bash
jupyter lab transformer_nlp_assignment.ipynb
# or
jupyter notebook transformer_nlp_assignment.ipynb
```

Run cells **top-to-bottom**. The notebook is organised into 8 sections:

| Section | Cells | Description |
|---------|-------|-------------|
| 1 | 1–3  | Data pipeline (load, clean, tokenise) |
| 2 | 4–7  | Model fine-tuning & evaluation |
| 3 | 8–11 | Attention extraction & heatmaps |
| 4 | 12–15| SHAP explainability |
| 5 | 16–19| LIME explainability |
| 6 | 20–22| SHAP vs LIME comparison |
| 7 | 23–25| Error analysis |
| 8 | 26–27| Flowchart & summary |

Choose the option to convert to script and run.Select the Convert to Script and Run option.

### Expected Runtime

| Step | CPU (no GPU) | GPU (RTX 3080) |
|------|-------------|----------------|
| Fine-tuning (3 epochs, 20K samples) | ~3-4 hours | ~15-20 min |
| Attention extraction | < 1 min | < 1 min |
| SHAP (20 samples) | ~15-30 min | ~5-8 min |
| LIME (20 samples) | ~10-20 min | ~4-7 min |

> Run the program with a small number of samples (LIME=100), and a small number of training points (TRAIN_SIZE=5000), for a quick test run.

---

## Reproducing Results

All random seed is set to 42. Key reproducibility settings:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

Model hyperparameters (cell 6):
- `learning_rate = 2e-5`
- `num_train_epochs = 3`
- `per_device_train_batch_size = 32`
- `weight_decay = 0.01`
- `warmup_ratio = 0.1`
- `MAX_LEN = 128`

---

## Output Files

The following files will be produced on completion of the entire notebook:

The test-set classification metrics are saved in the file `outputs/test_metrics.csv.
These files hold the SHAP results for the samples.These are SHAP results for 20 samples in the file: outputs/shap_explanations_20.csv.
A LIME explanation file.Contains the LIME results for 20 samples (outputs/lime_explanations_20.csv).
This table presents a comparison between the shapes.
This will contain a csv file called `outputs/error_analysis.csv` that contains details of the misclassified samples.
- `figures/*.png`: All plots and heat maps.
These files provide interactive explanations of LIME (open in browser): `- `outputs/lime_sample_*.html``

---

## References

2. The Amazon Polarity Dataset: https://huggingface.co/datasets/fancyzhx/amazon_polarity
2. SHAP Documentation: https://shap.readthedocs.io/en/latest/text_examples.html
3. LIME Documentation: https://lime-ml.readthedocs.io/en/latest/
4. DistilBERT: Sanh et al. (2019) — https://arxiv.org/abs/1910.01108
5. HuggingFace Transformers: https://huggingface.co/docs/transformers