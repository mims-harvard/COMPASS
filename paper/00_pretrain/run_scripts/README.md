# COMPASS (Pretraining from Scratch & Downstream Fine-Tuning)
[![Repro pypi version](https://img.shields.io/badge/Repro%20version-2.0.5-green)](https://pypi.org/project/immuno-compass/2.0.5/)

This branch provides a **reproducible pipeline** for the COMPASS model on a subset of TCGA dataset, including **pretraining from scratch** and **fine-tuning on downstream immunotherapy response datasets**. 

---

## 🧩 Step 1: Download Pretraining Datasets

### **TCGA Dataset**

<a href="https://doi.org/10.6084/m9.figshare.30580055" target="_blank">
  <img src="https://img.shields.io/badge/Figshare-DOI-blue?style=flat-square&logo=figshare" alt="TCGA DOI">
</a> 

This dataset contains preprocessed TCGA transcriptomic profiles used for COMPASS pretraining.
To facilitate reproducibility and efficient execution, we provide an immune-focused subset of 2,475 genes, which is sufficient to run all pretraining scripts in this repository.

After downloading the dataset from Figshare, please organize the files under the data/ directory with the following structure:
```text
data/
└── TCGA/
    ├── GENE.TABLE
    ├── TCGA.PATIENT.PROCESSED.TABLE
    ├── TCGA.PATIENT.TABLE
    └── TCGA.TPM.TABLE
````

---

### **ITRP Dataset (Alternative / Downstream Fine-Tuning)**

<a href="https://doi.org/10.6084/m9.figshare.30580109" target="_blank">
  <img src="https://img.shields.io/badge/Figshare-DOI-blue?style=flat-square&logo=figshare" alt="ITRP DOI">
</a>  

The `ITRP.zip` archive contains two serialized pandas tables:

* `ITRP.TPM.TABLE` — gene-level RNA-seq TPM matrix
* `ITRP.PATIENT.TABLE` — patient metadata (cancer type, therapy, response labels)

This dataset integrates **1,133 patients** from **16 immunotherapy cohorts**, all standardized using the COMPASS preprocessing pipeline.

---

### **Reproducing Datasets from Raw Data (Optional)**

If you prefer to regenerate the datasets from raw sources, please refer to:

* **TCGA preprocessing pipeline**
  [https://github.com/mims-harvard/COMPASS-web/tree/main/TCGA_dataset_processing](https://github.com/mims-harvard/COMPASS-web/tree/main/TCGA_dataset_processing)

* **ITRP mRNA pipeline**
  [https://github.com/mims-harvard/COMPASS-web/tree/main/mRNA_pipeline](https://github.com/mims-harvard/COMPASS-web/tree/main/mRNA_pipeline)

---

## 🧠 Step 2: Install COMPASS

```bash
# for torch 2.x, please install:
conda create -n compass python=3.12
conda activate compass
pip install immuno-compass -U
```

```bash
# for torch 1.x, please install:
conda create -n compass python=3.8
conda activate compass
pip install immuno-compass==2.0.5
```

---

## ⚙️ Step 3: Run Pretraining from Scratch

Go to the `run_scripts` folder, Open and execute the following notebook:

```text
01_pretraining.ipynb
```

> **Note**
> The example notebook uses the **TCGA-2475** gene subset for faster execution and reduced GPU memory usage.


---

## 🔬 Step 4: Run Downstream Fine-Tuning

You can either run the notebooks interactively or execute them sequentially via scripts.

Below is an example using `nbconvert` (tested on **V100 GPU**):

```bash
jupyter nbconvert --to notebook --execute 01_loco_nft.ipynb --output 01_loco_nft.ipynb
jupyter nbconvert --to notebook --execute 02_loco_lft.ipynb --output 02_loco_lft.ipynb
jupyter nbconvert --to notebook --execute 03_loco_pft.ipynb --output 03_loco_pft.ipynb
jupyter nbconvert --to notebook --execute 04_loco_fft.ipynb --output 04_loco_fft.ipynb
jupyter nbconvert --to notebook --execute 05_loco_lgr.ipynb --output 05_loco_lgr.ipynb
jupyter nbconvert --to notebook --execute 06_analysis_loco.ipynb --output 06_analysis_loco.ipynb
```

---

## 📌 Notes

* This repository is intended for **methodological reproducibility**, not for matching a single reported checkpoint.
* For close reproducibility, use the same weight initialization and document the GPU and PyTorch versions.
* Minor numerical differences may occur due to hardware, CUDA versions, or random seeds.
* For best reproducibility, fix random seeds and document GPU / PyTorch versions.
* Minor numerical differences may occur due to variations in hardware setup, CUDA version and GPU configuration, or ML weight initialization.
