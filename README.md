# COMPASS Reproducibility (Pretraining from Scratch & Downstream Fine-Tuning)
[![Repro pypi version](https://img.shields.io/badge/Repro%20version-2.0.3-green)](https://pypi.org/project/immuno-compass/2.0.3/)

This branch provides a **fully reproducible pipeline** for the COMPASS model, including **pretraining from scratch** and **fine-tuning on downstream immunotherapy response datasets**. 

---

## 🧩 Step 1: Download Pretraining Datasets

### **TCGA Dataset**

<a href="https://doi.org/10.6084/m9.figshare.30580055" target="_blank">
  <img src="https://img.shields.io/badge/Figshare-DOI-blue?style=flat-square&logo=figshare" alt="TCGA DOI">
</a> 

This dataset contains **preprocessed TCGA transcriptomic data** used for COMPASS pretraining.  
Both **2,475-gene (immune-focused)** and **15,672-gene (whole-transcriptome)** versions are provided.

After downloading, place the files under the `data/` directory as follows:

```text
data/
└── TCGA/
    ├── 2475/
    │   ├── GENE.TABLE
    │   ├── TCGA.PATIENT.PROCESSED.TABLE
    │   ├── TCGA.PATIENT.TABLE
    │   └── TCGA.TPM.TABLE
    └── 15672/
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
# IMPORTANT:
# If you are pretraining COMPASS from scratch,
# you MUST use this specific version
pip install immuno-compass==2.0.1
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
* Minor numerical differences may occur due to hardware, CUDA versions, or random seeds.
* For best reproducibility, fix random seeds and document GPU / PyTorch versions.