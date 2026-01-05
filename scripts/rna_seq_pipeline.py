# glioma_project
# rna_seq_pipeline.py
# Author: Amir Mahdi Taghizadeh


# imports

import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np


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

rna_file = os.path.join(raw_dir, "Human__TCGA_GBMLGG__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct")
rna_df = pd.read_csv(rna_file, sep="\t", index_col=0, header=0)
rna_df_cleaned = rna_df.dropna().drop_duplicates()

clinical_file = os.path.join(raw_dir, "Human__TCGA_GBMLGG__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi")
clinical_df = pd.read_csv(clinical_file, sep='\t', index_col=0)

rna_patients = rna_df_cleaned.columns.str.upper()  
clinical_df.columns = clinical_df.columns.str.upper() 

common_patients_clinical = [p for p in clinical_df.columns if p in rna_patients]
common_patients_rna = [p for p in rna_df_cleaned.columns if p in clinical_df.columns]
clinical_matched = clinical_df[common_patients_clinical]
rna_matched = rna_df_cleaned[common_patients_rna]

clinical_matched_t = clinical_matched.transpose()
clinical_matched_t.index.name = 'patient_id'
clinical_matched_t = clinical_matched_t.reset_index()

rna_matched_path = os.path.join(processed_dir, 'rna_matched.csv')
rna_matched.to_csv(rna_matched_path, index=True)

clinical_matched_path = os.path.join(processed_dir, 'clinical_rna_matched.csv')
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


# step 3: AnnData

rna_df_matched = pd.read_csv(rna_matched_path, index_col=0)
adata = sc.AnnData(rna_df_matched.T)
adata.var_names_make_unique()
adata.obs['histological_type'] = disease_labels
sc.pp.calculate_qc_metrics(adata, inplace=True)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata.write(os.path.join(processed_dir, 'rna_seq_anndata.h5ad'))
adata.raw = adata.copy()

print('Done with AnnData object. ')


# step 4: PCA & UMAP

adata.obs['histological_type'] = adata.obs['histological_type'].astype('category')
adata.uns['histological_type_colors'] = ['red', 'blue'] 
sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=30, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata, log=True)

plt.figure(figsize=(6,5))
sc.pl.pca(adata, color='total_counts', show=False)
pca_path = os.path.join(figures_dir, 'pca_rna_seq.png')
plt.savefig(pca_path, dpi=300, bbox_inches='tight')

sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
sc.tl.umap(adata)
plt.figure(figsize=(6,5))
sc.pl.umap(adata, color='histological_type', show=False)
umap_path = os.path.join(figures_dir, 'umap_rna_seq.png')
plt.savefig(umap_path, dpi=300, bbox_inches='tight')

print('Done with PCA-UMAP. ')


# step 5: deg

sc.tl.rank_genes_groups(adata, groupby='histological_type', method='t-test', use_raw=True)
sc.pl.rank_genes_groups(adata, n_genes=25)
deg = sc.get.rank_genes_groups_df(adata, group='GBM')  

deg_path = os.path.join(results_dir, 'deg_all.csv')
deg.to_csv(deg_path, index=False)
deg_sig = deg[(abs(deg['logfoldchanges']) > 1) & (deg['pvals_adj'] < 0.05)]

deg_sig_path = os.path.join(results_dir, 'sig_deg.csv')
deg_sig.to_csv(deg_sig_path, index=False)

print('Done with DEG.')


# step 6: volcano plot

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=deg,
    x='logfoldchanges',
    y=-np.log10(deg['pvals']),
    hue=deg['pvals_adj'] < 0.05,
    palette={True: 'red', False: 'gray'},
    alpha=0.7
    )

plt.axvline(0, color="black", lw=1)
plt.xlabel("log2 Fold Change (GBM vs LGG)")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot: GBM vs LGG")
volcano_path = os.path.join(figures_dir, 'Volcano_CD_deg.png')
plt.savefig(volcano_path, dpi=300, bbox_inches='tight')

print("Done with DEG volcano plot. ")