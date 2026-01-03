import pandas as pd
from itertools import chain
import numpy as np
import os, json
import torch

cwd = os.path.dirname(__file__)


with open(os.path.join(cwd, "cancer_code.json"), "r") as file:
    CANCER_CODE = json.load(file)

with open(os.path.join(cwd, "gene_tokens_long.json"), "r") as file:
    TOKENS_LONG = json.load(file)

with open(os.path.join(cwd, "gene_tokens_short.json"), "r") as file:
    TOKENS_SHORT = json.load(file)

CONCEPT = pd.read_csv(
    os.path.join(cwd, "conception_processed.tsv"), index_col=0, sep="\t"
)  # _processed


dfd = pd.read_excel(os.path.join(cwd, "concept_colors.xlsx"))
CONCEPT_palette = dfd.set_index("Concept_name").Color.to_dict()


RESPONSE_palette = {"R": "#008bfb", "NR": "#ff0051"}  # ,


