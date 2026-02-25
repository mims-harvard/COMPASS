# COMPASS: Generalizable AI predicts immunotherapy outcomes across cancers and treatments

<!-- [![Documentation Status](https://readthedocs.org/projects/immuno-compass/badge/?version=latest&style=flat-square)](https://immuno-compass.readthedocs.io/en/latest/) -->
[![PyPI version](https://badge.fury.io/py/immuno-compass.svg)](https://badge.fury.io/py/immuno-compass)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/immuno-compass?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/immuno-compass)
---
[![COMPASS Paper](https://img.shields.io/badge/Paper-COMPASS-yellow)](https://www.medrxiv.org/content/10.1101/2025.05.01.25326820v1)
[![ProjectPage](https://img.shields.io/badge/ProjectPage-COMPASS-red)](https://www.immuno-compass.com/)
[![COMPASS Dataset & Model](https://img.shields.io/badge/Dataset&Model-Download-green)](https://www.immuno-compass.com/download/)
[![COMPASS Personalized Response Maps](https://img.shields.io/badge/PersonalizedResponseMaps-Maps-blue)](https://www.immuno-compass.com/explore/index.html)
[![Online COMPASS Predictor](https://img.shields.io/badge/OnlineCOMPASSPredictor-Predictor-blue)](https://www.immuno-compass.com/predict/)

---

An `example2run.ipynb` under the example folder is provided to run below experiments. Navigate to the example folder to run the code below:


## 1. Installing and Importing COMPASS

#### Installation

```bash
conda create -n compass_tc2 python=3.12
conda activate compass_tc2
pip install immuno-compass -U
```

> **⚡ GPU (CUDA) Support:**
> By default, `pip install immuno-compass` will install the CPU-only version of PyTorch.
> If you want GPU acceleration, **please manually install the CUDA-enabled version of torch** *before* installing other dependencies. For example, for CUDA 11.7:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```
>
> For other CUDA versions, see the [PyTorch official installation guide](https://pytorch.org/get-started/locally/).
>
> If you use `conda`, you can install GPU-enabled torch with:
>
> ```bash
> conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
> ```

#### Importing COMPASS

Now, you can import COMPASS and its key components:

```python
import compass
from compass import PreTrainer, FineTuner, loadcompass
```




## 2. Making Predictions with a Compass Model

You can download all available COMPASS fine-tuned models [here](https://www.immuno-compass.com/download/) for prediction.

The input `df_tpm` is gene expression tabular data. Please refer [here](https://www.immuno-compass.com/help/index.html#section1) for details on generating input data. The first column represents the cancer code, while the remaining 15,672 columns correspond to genes. Each row represents one patient. An example input file can be downloaded [here](https://www.immuno-compass.com/download/other/compass_gide_tpm.tsv).

The output `df_pred` contains two columns: `0` indicates non-response and `1` indicates response.

```python
df_tpm = pd.read_csv('./data/compass_gide_tpm.tsv', sep='\t', index_col=0)
# OR directly load the compass model from https://www.immuno-compass.com/download/model/LOCO/pft_leave_Gide.pt 
model = loadcompass('./model/pft_leave_Gide.pt', map_location = 'cpu')
# Use map_location = 'cpu' if you dont have a GPU card
_, df_pred = model.predict(df_tpm, batch_size=128)
```



## 3. Extracting Features with a COMPASS Model

COMPASS models can function as feature extractors. The extracted features-gene-level, geneset-level, or cell type/pathway-level-can be used for downstream tasks such as building a logistic regression model for response prediction or a Cox regression model for survival prediction.

```python
# Load COMPASS model and extract the features for a new cohort
model = loadcompass('https://www.immuno-compass.com/download/model/finetuner_pft_all.pt') 
dfgn, dfgs, dfct = model.extract(df_tpm, batch_size=128, with_gene_level=True)
```

The outputs `dfgn`, `dfgs`, and `dfct` correspond to gene-level (15,672), geneset-level (133), and concept-level (44) features, respectively. The extracted features are scalar scores. If you need vector features (dim=32), use the following method:

```python
dfgs, dfct = model.project(df_tpm, batch_size=128)
```



## 4. Fine-Tuning COMPASS on Your Own Data

If you have in-house data and would like to fine-tune a COMPASS model with your own data, you can use any COMPASS model for fine-tuning. You can either load the pre-trained COMPASS model or a publicly available fine-tuned COMPASS model.

**Important Note:** If you choose a fine-tuned model for further fine-tuning (multi-stage FT), ensure that the `load_decoder` parameter in `ft_args` is set to `True`:
```python
ft_args = {'load_decoder': True}
```
Select one of the fine-tuning modes: 'FFT', 'PFT', or 'LFT'. For small datasets (n<50), 'LFT' is recommended. 

### Example Fine-Tuning Process
```python
model = loadcompass('https://www.immuno-compass.com/download/model/LOCO/pft_leave_Gide.pt')  
ft_args = {'mode': 'PFT', 'lr': 1e-3, 'batch_size': 16, 'max_epochs': 100, 'load_decoder': True}

finetuner = FineTuner(model, **ft_args)

# Load the true labels
df_cln = pd.read_csv('./data/compass_gide_clinical.tsv', sep='\t', index_col=0)
dfy = pd.get_dummies(df_cln.response_label)

# Fine-tune the model
finetuner.tune(df_tpm, dfy)
finetuner.save('./model/my_finetuner.pt')
```
This process fine-tunes the COMPASS model on your data and saves the updated model for future use.




## 5. Pre-training COMPASS from Scratch
```python
# Load the example dataset for pretraining
# We provide sample datasets that include gene expression data for training and testing
# Ensure the data is preprocessed appropriately before use
tcga_train_sample = pd.read_csv('./data/tcga_example_train.tsv', sep='\t', index_col=0)
tcga_test_sample = pd.read_csv('./data/tcga_example_test.tsv', sep='\t', index_col=0)

# Define pre-training hyperparameters
pt_args = {'lr': 1e-3, 'batch_size': 128, 'epochs': 20}
pretrainer = PreTrainer(**pt_args)

# Train the model using the provided training and test datasets
# - dfcx_train: Training dataset
# - dfcx_test: Validation dataset to monitor performance
pretrainer.train(dfcx_train=tcga_train_sample,
                 dfcx_test=tcga_test_sample)

# Save the trained pretrainer model for future use
pretrainer.save('./model/my_pretrainer.pt')
```



## 6. Baseline Methods Usage Examples
You can also extract features using baseline immune score methods. These features can be used to build models for response prediction.
```python
# Import baseline immune score methods
import sys
sys.path.insert(0, '../')
from baseline.immune_score import immune_score_methods

# Extract features using baseline methods
# These features can be used to develop a logistic regression model for response prediction
res = []
for method_name, method_function in immune_score_methods.items():
    baseline_model = method_function(cancer_type='SKCM', drug_target='PD1')
    scores = baseline_model(df_tpm)
    res.append(scores)

# Combine results into a single DataFrame
pd.concat(res, axis=1).head()
```




## 7. Additional Information

This section provides detailed information to help you get started with the COMPASS project. We explain how to generate the necessary inputs from raw FASTQ data and introduce our online web server that supports both prediction and feature extraction using our pre-trained COMPASS models.

### Generating COMPASS Inputs from Raw FASTQ Data

Generating high-quality inputs is crucial for the optimal performance of the COMPASS models. Our comprehensive [Compass Data Pre-Processing Guide](https://www.immuno-compass.com/help/index.html) walks you through the entire workflow, ensuring that your raw FASTQ data is processed into a robust format ready for accurate predictions and feature extraction.

### Online Web Server for Prediction and Feature Extraction

To simplify the use of our models, we offer an online web server that enables you to interact directly with the COMPASS models without local installations. The web server provides two primary functionalities:

- **Prediction:** Submit your processed data to generate model predictions using our [online prediction tool](https://www.immuno-compass.com/predict).
- **Feature Extraction:** Extract key data attributes with our [feature extraction tool](https://www.immuno-compass.com/extract).

These user-friendly services are designed to streamline your workflow and integrate COMPASS into your analytical processes.

### Contributing Your Own COMPASS Models

We welcome contributions from the community. If you have developed a COMPASS model that can enhance our project, we encourage you to share it. By contributing your model, you help enrich the COMPASS ecosystem and promote collaborative innovation. For details on how to submit your model, please refer to our contribution guidelines. You can also [join our Slack channel](https://join.slack.com/t/immuno-compass/shared_invite/zt-2znjho738-YZOfLEGLNEH5eH_0W1TmQg) to discuss and collaborate with other users.

### Citing Our Work

If you use our resources, please cite our work as follows:

Wanxiang Shen, Thinh H. Nguyen, Michelle M. Li, Yepeng Huang, Intae Moon, Nitya Nair, Daniel Marbach‡, and Marinka Zitnik‡. *Generalizable AI predicts immunotherapy outcomes across cancers and treatments* [J]. [medRxiv](https://www.medrxiv.org/content/10.1101/2025.05.01.25326820).

---

We hope this information helps you make the most of the COMPASS project. If you have any questions or need further assistance, please do not hesitate to contact our support [team](https://www.immuno-compass.com/about/index.html).
