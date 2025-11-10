## 🧩 Step 1: Download Pretraining Datasets

### **TCGA Dataset**

<a href="https://doi.org/10.6084/m9.figshare.30580055" target="_blank">
  <img src="https://img.shields.io/badge/Figshare-DOI-blue?style=flat-square&logo=figshare" alt="TCGA DOI">
</a> 

This dataset provides **preprocessed TCGA transcriptomic data** used for COMPASS model pretraining.
Both 1,065-gene (immune-focused) and 15,672-gene (whole transcriptome) versions are included. Download the dataset and put them in the data/ folder

```
data/
└── TCGA/
    ├── 1065/
    │   ├── GENE.TABLE
    │   ├── TCGA.PATIENT.PROCESSED.TABLE
    │   ├── TCGA.PATIENT.TABLE
    │   └── TCGA.TPM.TABLE
    └── 15672/
        ├── GENE.TABLE
        ├── TCGA.PATIENT.PROCESSED.TABLE
        ├── TCGA.PATIENT.TABLE
        └── TCGA.TPM.TABLE
```

---

### **ITRP Dataset (Alternative)**

<a href="https://doi.org/10.6084/m9.figshare.30580109" target="_blank">
  <img src="https://img.shields.io/badge/Figshare-DOI-blue?style=flat-square&logo=figshare" alt="ITRP DOI">
</a>  

The **ITRP.zip** file contains two pandas pickle tables:

* `ITRP.TPM.TABLE` — gene-level RNA-seq TPM matrix
* `ITRP.PATIENT.TABLE` — patient metadata (cancer type, therapy, response labels)

This dataset integrates **1,133 patients** across **16 immunotherapy cohorts**, standardized via the COMPASS preprocessing pipeline.

---

### **Reproduce from Raw Data**

You can also generate both datasets from scratch using:

* **TCGA dataset preprocessing** → [TCGA_dataset_processing](https://github.com/mims-harvard/COMPASS-web/tree/main/TCGA_dataset_processing)
* **ITRP dataset pipeline** → [mRNA_pipeline](https://github.com/mims-harvard/COMPASS-web/tree/main/mRNA_pipeline)

---

## 🧠 Step 2: Install COMPASS

```bash
pip install immuno-compass -U
```

---

## ⚙️ Step 3: Run Pretraining

Open and execute:

```
01_pretraining.ipynb
```

> The example notebook uses the **TCGA-1065** subset for faster training and lower GPU memory usage.
> Using the full 15,672-gene version requires high-memory GPUs.

---

## 🔬 Step 4: Run Fine-Tuning

Open and execute:

```
02_finetuning.ipynb
```
