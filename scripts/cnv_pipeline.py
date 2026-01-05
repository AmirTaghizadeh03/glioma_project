# glioma_project
# cnv_pipeline.py
# Author: Amir Mahdi Taghizadeh


# imports

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# step 0: directories

base_dir = r"C:\Users\Asus\Desktop\projects\glioma_optimised"
raw_dir = os.path.join(base_dir, "data", "raw")
processed_dir = os.path.join(base_dir, "data", "processed")
results_dir = os.path.join(base_dir, "results")
figures_dir = os.path.join(base_dir, "figures")

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)


# step 1: loading data files

cnv_file = os.path.join(raw_dir, "Human__TCGA_GBMLGG__BI__SCNA__SNP_6.0__01_28_2016__BI__Gene__Firehose_GISTIC2.cct")
cnv_df = pd.read_csv(cnv_file, sep="\t", index_col=0, header=0)
cnv_df_cleaned = cnv_df.dropna().drop_duplicates()

clinical_file = os.path.join(raw_dir, "Human__TCGA_GBMLGG__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi")
clinical_df = pd.read_csv(clinical_file, sep='\t', index_col=0)

cnv_patients = cnv_df_cleaned.columns.str.upper()  
clinical_df.columns = clinical_df.columns.str.upper() 

common_patients_clinical = [p for p in clinical_df.columns if p in cnv_patients]
common_patients_cnv = [p for p in cnv_df_cleaned.columns if p in clinical_df.columns]
clinical_matched = clinical_df[common_patients_clinical]
cnv_matched = cnv_df_cleaned[common_patients_cnv]
cnv_matched = cnv_matched.apply(pd.to_numeric, errors='coerce')


clinical_matched_t = clinical_matched.transpose()
clinical_matched_t.index.name = 'patient_id'
clinical_matched_t = clinical_matched_t.reset_index()

cnv_matched_path = os.path.join(processed_dir, 'cnv_matched.csv')
cnv_matched.to_csv(cnv_matched_path, index=True)

clinical_matched_path = os.path.join(processed_dir, 'clinical_cnv_matched.csv')
clinical_matched_t.to_csv(clinical_matched_path, index=False)

print('Done with loading files. ')


# step 2: metadata

clinical_df_matched = pd.read_csv(clinical_matched_path, index_col=0)
hist_types = clinical_df_matched['histological_type']
gbm_keywords = [
    'untreatedprimary(denovo)gbm',
    'glioblastomamultiforme(gbm)',
    'primary(denovo)gbm'
]

disease_labels = [
    'GBM' if any(k in h.lower() for k in gbm_keywords)
    else 'LGG'
    for h in hist_types
]

print('Done with metadata. ')


# step 3: cnv analysis

cnv_matched.index = cnv_matched.index.str.split('|').str[0]

cnv_matched = cnv_matched[clinical_df_matched.index]
assert list(cnv_matched.columns) == list(clinical_df_matched.index)

cnv_matched = cnv_matched.groupby(cnv_matched.index).mean()

GAIN_TH = 0.3
LOSS_TH = -0.3

cnv_gain = cnv_matched > GAIN_TH
cnv_loss = cnv_matched < LOSS_TH
cnv_gain.index = cnv_gain.index.str.strip()  
assert cnv_gain.index.is_unique
cnv_loss.index = cnv_loss.index.str.strip()
assert cnv_loss.index.is_unique

clinical_df_matched['disease'] = disease_labels

gbm_samples = clinical_df_matched.query("disease == 'GBM'").index
lgg_samples = clinical_df_matched.query("disease == 'LGG'").index

gain_results = []

for gene in cnv_gain.index:
    gbm_gain = int(cnv_gain.loc[gene, gbm_samples].sum())
    gbm_no   = len(gbm_samples) - gbm_gain

    lgg_gain = int(cnv_gain.loc[gene, lgg_samples].sum())
    lgg_no   = len(lgg_samples) - lgg_gain

    if gbm_gain + lgg_gain == 0:
        continue

    table = [[gbm_gain, gbm_no],
             [lgg_gain, lgg_no]]

    _, pval = fisher_exact(table)
    gain_results.append((gene, gbm_gain, lgg_gain, pval))


loss_results = []

for gene in cnv_loss.index:
    gbm_loss = int(cnv_loss.loc[gene, gbm_samples].sum())
    gbm_no   = len(gbm_samples) - gbm_loss

    lgg_loss = int(cnv_loss.loc[gene, lgg_samples].sum())
    lgg_no   = len(lgg_samples) - lgg_loss

    if gbm_loss + lgg_loss == 0:
        continue

    table = [[gbm_loss, gbm_no],
             [lgg_loss, lgg_no]]

    _, pval = fisher_exact(table)
    loss_results.append((gene, gbm_loss, lgg_loss, pval))


gain_df = pd.DataFrame(
    gain_results,
    columns=["Gene", "GBM_gain", "LGG_gain", "pval"]
)

gain_df["FDR"] = multipletests(
    gain_df["pval"], method="fdr_bh"
)[1]

gain_path = os.path.join(results_dir, 'cnv_gain_all.csv')
gain_df.to_csv(gain_path, index=False)

loss_df = pd.DataFrame(
    loss_results,
    columns=["Gene", "GBM_loss", "LGG_loss", "pval"]
)

loss_df["FDR"] = multipletests(
    loss_df["pval"], method="fdr_bh"
)[1]

loss_path = os.path.join(results_dir, 'cnv_loss_all.csv')
loss_df.to_csv(loss_path, index=False)

gain_df["GBM_gain"] = pd.to_numeric(gain_df["GBM_gain"], errors="coerce")
gain_df["LGG_gain"] = pd.to_numeric(gain_df["LGG_gain"], errors="coerce")
loss_df["GBM_loss"] = pd.to_numeric(loss_df["GBM_loss"], errors="coerce")
loss_df["LGG_loss"] = pd.to_numeric(loss_df["LGG_loss"], errors="coerce")

sig_gain_genes = gain_df.loc[
    gain_df["FDR"] < 0.05
].copy()


sig_loss_genes = loss_df.loc[
    loss_df["FDR"] < 0.05
].copy()


sig_gain_genes["freq_diff"] = (
    sig_gain_genes["GBM_gain"] / len(gbm_samples)
    - sig_gain_genes["LGG_gain"] / len(lgg_samples)
).abs()

sig_loss_genes["freq_diff"] = (
    sig_loss_genes["GBM_loss"] / len(gbm_samples)
    - sig_loss_genes["LGG_loss"] / len(lgg_samples)
).abs()

sig_gain_path = os.path.join(results_dir, 'sig_cnv_gain.csv')
sig_gain_genes.to_csv(sig_gain_path, index=False)

sig_loss_path = os.path.join(results_dir, 'sig_cnv_loss.csv')
sig_loss_genes.to_csv(sig_loss_path, index=False)

sig_gain_10 = sig_gain_genes.loc[
    sig_gain_genes["freq_diff"] >= 0.10
].copy()

sig_gain_10_path = os.path.join(results_dir, "sig_cnv_gain_freq10.csv")
sig_gain_10.to_csv(sig_gain_10_path, index=False)


sig_loss_10 = sig_loss_genes.loc[
    sig_loss_genes["freq_diff"] >= 0.10
].copy()

sig_loss_10_path = os.path.join(results_dir, "sig_cnv_loss_freq10.csv")
sig_loss_10.to_csv(sig_loss_10_path, index=False)


final_cnv_genes = set(
    pd.concat([
        sig_gain_genes.loc[
            (sig_gain_genes["FDR"] < 0.05) & (sig_gain_genes["freq_diff"] > 0.10)
            
        ],
        sig_loss_genes.loc[
            (sig_loss_genes["FDR"] < 0.05) & (sig_loss_genes["freq_diff"] > 0.10)
            
        ]
    ])
)

print('Done with cnv Analysis. ')


# step 4: visualisation

gain_df["Total_gain"] = gain_df["GBM_gain"] + gain_df["LGG_gain"]

top_gain_genes = (
    gain_df
    .sort_values("Total_gain", ascending=False)
    .head(20)["Gene"]
)
top_gain_genes = [g for g in top_gain_genes if g in cnv_gain.index]

top_gain_df = (
    gain_df
    .assign(Total_gain=gain_df["GBM_gain"] + gain_df["LGG_gain"])
    .sort_values("Total_gain", ascending=False)
    .loc[lambda x: x["Gene"].isin(cnv_gain.index)]
    .head(20)
    .reset_index(drop=True)
)

top_gain_df.insert(0, "Rank", top_gain_df.index + 1)

top_gain_df.to_csv(
    os.path.join(results_dir, "Top20_cnv_gain_genes.csv"),
    index=False
)

gain_freq = pd.DataFrame({
    "Gene": top_gain_genes,
    "GBM": cnv_gain.loc[top_gain_genes, gbm_samples].mean(axis=1) * 100,
    "LGG": cnv_gain.loc[top_gain_genes, lgg_samples].mean(axis=1) * 100
}).set_index("Gene")


gain_freq.plot(kind="bar", figsize=(12,6), width=0.8)
plt.ylabel("Percentage of patients with gain")
plt.title("Top 20 CNV Gains in GBM vs LGG")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.ylim(0, 100)
plt.savefig(os.path.join(figures_dir, 'gain_cnv_barplot.png'), dpi=300, bbox_inches='tight')


loss_df["Total_loss"] = loss_df["GBM_loss"] + loss_df["LGG_loss"]

top_loss_genes = (
    loss_df
    .sort_values("Total_loss", ascending=False)
    .head(20)["Gene"]
)
top_loss_genes = [g for g in top_loss_genes if g in cnv_loss.index]
top_loss_df = (
    loss_df
    .assign(Total_loss=loss_df["GBM_loss"] + loss_df["LGG_loss"])
    .sort_values("Total_loss", ascending=False)
    .loc[lambda x: x["Gene"].isin(cnv_loss.index)]
    .head(20)
    .reset_index(drop=True)
)

top_loss_df.insert(0, "Rank", top_loss_df.index + 1)

top_loss_df.to_csv(
    os.path.join(results_dir, "Top20_cnv_loss_genes.csv"),
    index=False
)
loss_freq = pd.DataFrame({
    "Gene": top_loss_genes,
    "GBM": cnv_loss.loc[top_loss_genes, gbm_samples].mean(axis=1) * 100,
    "LGG": cnv_loss.loc[top_loss_genes, lgg_samples].mean(axis=1) * 100
}).set_index("Gene")


loss_freq.plot(kind="bar", figsize=(12,6), width=0.8)
plt.ylabel("Percentage of patients with loss")
plt.title("Top 20 CNV Losses in GBM vs LGG")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.ylim(0, 100)
plt.savefig(os.path.join(figures_dir, 'loss_cnv_barplot.png'), dpi=300, bbox_inches='tight')

print('Done with cnv visualisation. ')
