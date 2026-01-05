# glioma_project
# mutation_pipeline.py
# Author: Amir Mahdi Taghizadeh


# imports

import os
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns


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

mutation_file = os.path.join(raw_dir, "Human__TCGA_GBMLGG__WUSM__Mutation__GAIIx__01_28_2016__BI__Gene__Firehose_MutSig2CV(1).cbt")
mutation_df = pd.read_csv(mutation_file, sep="\t", index_col=0, header=0)
mutation_df_cleaned = mutation_df.dropna().drop_duplicates()

clinical_file = os.path.join(raw_dir, "Human__TCGA_GBMLGG__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi")
clinical_df = pd.read_csv(clinical_file, sep='\t', index_col=0)


mutation_patients = mutation_df_cleaned.columns.str.upper()  
clinical_df.columns = clinical_df.columns.str.upper() 


common_patients_clinical = [p for p in clinical_df.columns if p in mutation_patients]
common_patients_mutation = [p for p in mutation_df_cleaned.columns if p in clinical_df.columns]
clinical_matched = clinical_df[common_patients_clinical]
mutation_matched = mutation_df_cleaned[common_patients_mutation]
mutation_matched = mutation_matched.apply(pd.to_numeric, errors='coerce')


clinical_matched_t = clinical_matched.transpose()
clinical_matched_t.index.name = 'patient_id'
clinical_matched_t = clinical_matched_t.reset_index()

mutation_matched_path = os.path.join(processed_dir, 'mutation_matched.csv')
mutation_matched.to_csv(mutation_matched_path, index=True)

clinical_matched_path = os.path.join(processed_dir, 'clinical_mutation_matched.csv')
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


# step 3: mutation analysis

mutation_matched = mutation_matched[clinical_df_matched.index]
assert list(mutation_matched.columns) == list(clinical_df_matched.index)

MUT_TH = 0
mutation_binary = (mutation_matched > MUT_TH).astype(int)

clinical_df_matched['disease'] = disease_labels
gbm_samples = clinical_df_matched.query("disease == 'GBM'").index
lgg_samples = clinical_df_matched.query("disease == 'LGG'").index

mut_results = []

for gene in mutation_binary.index:
    gbm_mut = int(mutation_binary.loc[gene, gbm_samples].sum())
    gbm_no  = len(gbm_samples) - gbm_mut

    lgg_mut = int(mutation_binary.loc[gene, lgg_samples].sum())
    lgg_no  = len(lgg_samples) - lgg_mut

    if gbm_mut + lgg_mut == 0:
        continue

    table = [[gbm_mut, gbm_no],
             [lgg_mut, lgg_no]]

    _, pval = fisher_exact(table)
    mut_results.append((gene, gbm_mut, lgg_mut, pval))


mut_df = pd.DataFrame(
    mut_results,
    columns=["Gene", "GBM_mut", "LGG_mut", "pval"]
)

mut_df["FDR"] = multipletests(
    mut_df["pval"], method="fdr_bh"
)[1]
mutation_path = os.path.join(results_dir, 'mutation_all.csv')
mut_df.to_csv(mutation_path, index=False)

sig_mut_df = mut_df.loc[mut_df["FDR"] < 0.05].copy()
sig_mut_df.head(13)
sig_mutation_path = os.path.join(results_dir, 'sig_mutation.csv')
sig_mut_df.to_csv(sig_mutation_path, index=False)

print('Done with mutation Analysis. ')


# step 4: visualisation

mut_df["Total_mut"] = mut_df["GBM_mut"] + mut_df["LGG_mut"]

top_mut_genes = (
    mut_df
    .sort_values("Total_mut", ascending=False)
    .head(20)["Gene"]
)

top_mut_genes = [g for g in top_mut_genes if g in mutation_binary.index]

mut_freq = pd.DataFrame({
    "GBM": mutation_binary.loc[top_mut_genes, gbm_samples].mean(axis=1) * 100,
    "LGG": mutation_binary.loc[top_mut_genes, lgg_samples].mean(axis=1) * 100
})

mut_freq.plot(
    kind="bar",
    figsize=(13,6),
    width=0.85
)

plt.ylabel("Patients with mutation (%)")
plt.title("Top Mutated Genes (MutSig-derived) - GBM vs LGG")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
mut_visualisaton_path = os.path.join(figures_dir, 'mutation_barplot.png')

plt.savefig(mut_visualisaton_path, dpi=300, bbox_inches='tight')



top_heatmap_genes = (
    mut_df
    .sort_values("GBM_mut", ascending=False)
    .head(30)["Gene"]
)

heatmap_data = mutation_binary.loc[
    top_heatmap_genes,
    list(gbm_samples) + list(lgg_samples)
]

plt.figure(figsize=(14,6))
sns.heatmap(
    heatmap_data,
    cmap="Greys",
    cbar=False,
    xticklabels=False
)

plt.title("Somatic Mutation Landscape (GBM vs LGG)")
plt.ylabel("Genes")
plt.xlabel("Samples")
plt.tight_layout()
heatmap_path = os.path.join(figures_dir, 'mutation_heatmap.png')

plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')

print('Done with mutation visualisation. ')