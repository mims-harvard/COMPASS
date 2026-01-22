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