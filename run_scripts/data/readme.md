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