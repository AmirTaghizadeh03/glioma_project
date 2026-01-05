# glioma_project
# combined_pipeline.py
# Author: Amir Mahdi Taghizadeh


# imports

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import numpy as np
from matplotlib_venn import venn2
from lifelines import CoxPHFitter
import gseapy as gp

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

sig_mutation_path = os.path.join(results_dir, 'sig_mutation.csv')
sig_mutation_df = pd.read_csv(sig_mutation_path)

sig_cnv_gain_path = os.path.join(results_dir, 'sig_cnv_gain_freq10.csv')
sig_cnv_gain_df = pd.read_csv(sig_cnv_gain_path)

sig_cnv_loss_path = os.path.join(results_dir, 'sig_cnv_loss_freq10.csv')
sig_cnv_loss_df = pd.read_csv(sig_cnv_loss_path)

sig_deg_path = os.path.join(results_dir, 'sig_deg.csv')
sig_deg_df = pd.read_csv(sig_deg_path)

print('Done with loading files. ')

# step 2: cleaning and combining

sig_mutation_df["Gene"] = sig_mutation_df["Gene"].astype(str)
sig_cnv_gain_df["Gene"] = sig_cnv_gain_df["Gene"].astype(str)
sig_cnv_loss_df["Gene"] = sig_cnv_loss_df["Gene"].astype(str)

sig_deg_df["Gene"] = sig_deg_df["names"].astype(str)

deg_genes = set(sig_deg_df["Gene"])
mutation_genes = set(sig_mutation_df["Gene"])
cnv_gain_genes = set(sig_cnv_gain_df["Gene"])
cnv_loss_genes = set(sig_cnv_loss_df["Gene"])

cnv_all_genes = cnv_gain_genes.union(cnv_loss_genes)

final_3omics = deg_genes & cnv_all_genes & mutation_genes

## no overlaped genes in 3omics so continue with 2omics shared for ML 

final_2omics = (
    (deg_genes & cnv_all_genes) |
    (deg_genes & mutation_genes) |
    (cnv_all_genes & mutation_genes)
)

final_2omics_df = pd.DataFrame({"Gene": sorted(final_2omics)})
final_2omics_df.to_csv(
    os.path.join(results_dir, "final_genes_2omics.csv"),
    index=False
)

final_genes = pd.read_csv(
    os.path.join(results_dir, "final_genes_2omics.csv")
)["Gene"].unique().tolist()

print('Done with cleaning and combining. ')


# step 3: feature selection for ML

rna_expr = pd.read_csv(
    os.path.join(processed_dir, "rna_matched.csv"),
    index_col=0
)

rna_feat = (
    rna_expr
    .loc[rna_expr.index.intersection(final_genes)]
    .T
)

rna_feat.columns = [f"RNA_{g}" for g in rna_feat.columns]

cnv_matched = pd.read_csv(
    os.path.join(processed_dir, "cnv_matched.csv"),
    index_col=0
)

cnv_matched.index = cnv_matched.index.str.split("|").str[0]
cnv_matched = cnv_matched.groupby(cnv_matched.index).mean()

cnv_feat = (
    cnv_matched
    .loc[cnv_matched.index.intersection(final_genes)]
    .T
)

cnv_feat.columns = [f"CNV_{g}" for g in cnv_feat.columns]

mutation_matched = pd.read_csv(
    os.path.join(processed_dir, "mutation_matched.csv"),
    index_col=0
)

mutation_binary = (mutation_matched > 0).astype(int)

mut_feat = (
    mutation_binary
    .loc[mutation_binary.index.intersection(final_genes)]
    .T
)

mut_feat.columns = [f"MUT_{g}" for g in mut_feat.columns]

## metadata

clinical_df_matched = pd.read_csv(
    os.path.join(processed_dir, "clinical_cnv_matched.csv"),
    index_col=0
)

hist_types = clinical_df_matched['histological_type']

gbm_keywords = [
    'untreatedprimary(denovo)gbm',
    'glioblastomamultiforme(gbm)',
    'primary(denovo)gbm'
]

disease_labels = [
    'GBM' if any(k in h.lower() for k in gbm_keywords) else 'LGG'
    for h in hist_types
]

y = pd.Series(disease_labels, index=clinical_df_matched.index, name='diagnosis')


common_patients = (
    rna_feat.index
    .intersection(cnv_feat.index)
    .intersection(mut_feat.index)
    .intersection(y.index)
)


rna_feat = rna_feat.loc[common_patients]
cnv_feat = cnv_feat.loc[common_patients]
mut_feat = mut_feat.loc[common_patients]
y = y.loc[common_patients]

assert rna_feat.shape[0] == cnv_feat.shape[0] == mut_feat.shape[0] == y.shape[0]

print('Done with feature selection for ML. ')


# step 4: ML

X = pd.concat([rna_feat, cnv_feat, mut_feat], axis=1)

y_encoded = y.map({'LGG': 0, 'GBM': 1})
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled.loc[:, rna_feat.columns.tolist() + cnv_feat.columns.tolist()] = scaler.fit_transform(
    X_scaled.loc[:, rna_feat.columns.tolist() + cnv_feat.columns.tolist()]
)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)


rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

auc_rf = roc_auc_score(y_test, y_prob_rf)

class_report_rf = classification_report(y_test, y_pred_rf, target_names=['LGG', 'GBM'])

cm_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['LGG','GBM'], yticklabels=['LGG','GBM'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
confision_heatmap_rf_path = os.path.join(figures_dir, 'confision_heatmap_rf.png')
plt.savefig(confision_heatmap_rf_path, dpi=300, bbox_inches='tight')


cv_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc_rf = cross_val_score(rf, X_scaled, y_encoded, cv=cv_rf, scoring='roc_auc')
mean_cv_rf = cv_auc_rf.mean()

models_performance_path = os.path.join(results_dir, 'models_performance.txt')
with open(models_performance_path, 'a') as f:
    f.write("\n" + "="*60 + "\n")
    f.write("Random Forest\n")
    f.write("="*60 + "\n")
    f.write(f"Test ROC-AUC: {auc_rf:.3f}\n\n")
    f.write(f'Classification Report: {class_report_rf}\n\n')
    f.write(f'Confusion Matrix: {cm_rf}\n\n')
    f.write(f"5-fold CV ROC-AUC: {cv_auc_rf}\n\n")
    f.write(f"Mean CV ROC-AUC: {mean_cv_rf}\n\n")

rf_importance = pd.Series(rf.feature_importances_, index=X_scaled.columns)
rf_importance_df = (
    rf_importance
    .rename("importance")
    .reset_index()
    .rename(columns={"index": "Feature"})
    .sort_values("importance", ascending=False)
)

rf_importance_path = os.path.join(results_dir, "random_forest_feature_importance.csv")
rf_importance_df.to_csv(rf_importance_path, index=False)

top_features_rf = rf_importance.sort_values(ascending=False).head(20)

plt.figure(figsize=(10,6))
sns.barplot(x=top_features_rf.values, y=top_features_rf.index)
plt.title("Top 20 Feature Importances")
top_features_barplot_rf_path = os.path.join(figures_dir, 'top_features_barplot_rf.png')
plt.savefig(top_features_barplot_rf_path, dpi=300, bbox_inches='tight')


lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:,1]

auc_lr = roc_auc_score(y_test, y_prob_lr)
class_report_lr = classification_report(y_test, y_pred_lr, target_names=['LGG','GBM'])

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['LGG','GBM'], yticklabels=['LGG','GBM'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
confision_heatmap_lr_path = os.path.join(figures_dir, 'confision_heatmap_lr.png')
plt.savefig(confision_heatmap_lr_path, dpi=300, bbox_inches='tight')


cv_lr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc_lr = cross_val_score(lr, X_scaled, y_encoded, cv=cv_lr, scoring='roc_auc')
mean_cv_lr = cv_auc_lr.mean()
models_performance_path = os.path.join(results_dir, 'models_performance.txt')
with open(models_performance_path, 'a') as f:
    f.write("\n" + "="*60 + "\n")
    f.write("Logistic Regression\n")
    f.write("="*60 + "\n")
    f.write(f"Test ROC-AUC: {auc_lr:.3f}\n\n")
    f.write(f'Classification Report: {class_report_lr}\n\n')
    f.write(f'Confusion Matrix: {cm_lr}\n\n')
    f.write(f"5-fold CV ROC-AUC: {cv_auc_lr}\n\n")
    f.write(f"Mean CV ROC-AUC: {mean_cv_lr}\n\n")

feature_names = X_scaled.columns

coef = lr.coef_[0]

lr_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coef,
    'Abs_Coeff': np.abs(coef)
})

lr_importance = lr_importance.sort_values('Abs_Coeff', ascending=False)
lr_importance_path = os.path.join(results_dir, "logistic_regression_feature_importance.csv")
lr_importance.to_csv(lr_importance_path, index=False)

top_n_lr_features = 20

plt.figure(figsize=(12,6))
sns.barplot(
    data=lr_importance.head(top_n_lr_features),
    x='Abs_Coeff',
    y='Feature'
)
plt.xlabel('Absolute Coefficient (Feature Importance)')
plt.ylabel('Feature')
plt.title(f'Top {top_n_lr_features} Important Features in Logistic Regression')
plt.tight_layout()
top_features_barplot_lr_path = os.path.join(figures_dir, 'top_features_barplot_lr.png')
plt.savefig(top_features_barplot_lr_path, dpi=300, bbox_inches='tight')

top_n = 20

lr_top_features = lr_importance.sort_values("Abs_Coeff", ascending=False).head(top_n)["Feature"].tolist()
rf_top_features = rf_importance.sort_values(ascending=False).head(top_n).index.tolist()

overlap_features = list(set(lr_top_features) & set(rf_top_features))

plt.figure(figsize=(6,6))
venn2([set(lr_top_features), set(rf_top_features)],
      set_labels=("Logistic Regression", "Random Forest"))
plt.figtext(0.5, -0.1, 'Overlaping features: RNA_RB1, RNA_RETN, CNV_LBX1, CNV_CEACAM8, CNV_PTEN, RNA_TP53'
            ,wrap=True, ha='center',va='center', fontsize=10)
plt.title("Top Feature Overlap")
overlap_venn_path = os.path.join(figures_dir, 'models_overlap_venn.png')
plt.savefig(overlap_venn_path, dpi=300, bbox_inches='tight')

overlap_df_models = pd.DataFrame({"Feature": overlap_features})

overlap_df_models["LR_abs_coeff"] = overlap_df_models["Feature"].map(
    lr_importance.set_index("Feature")["Abs_Coeff"]
)
overlap_df_models["RF_importance"] = overlap_df_models["Feature"].map(rf_importance)
overlap_df_models["Omics"] = overlap_df_models["Feature"].str.split("_").str[0]
overlap_df_models_path = os.path.join(results_dir, 'overlap_features_models.csv')
overlap_df_models.to_csv(overlap_df_models_path, index=False)


overlap_df_models["LR_rank"] = overlap_df_models["LR_abs_coeff"].rank(ascending=False)
overlap_df_models["RF_rank"] = overlap_df_models["RF_importance"].rank(ascending=False)

x = range(len(overlap_df_models))
plt.figure(figsize=(8,6))
plt.bar(x, overlap_df_models["LR_rank"], width=0.4, label="LR", align='center')
plt.bar([i+0.4 for i in x], overlap_df_models["RF_rank"], width=0.4, label="RF", align='center')
plt.xticks([i+0.2 for i in x], overlap_df_models["Feature"], rotation=45, ha='right')
plt.ylabel("Rank (1 = most important)")
plt.title("Top 6 Overlapping Features: Rank Comparison LR vs RF")
plt.legend()
plt.tight_layout()
rank_features_barplot_path = os.path.join(figures_dir, 'rank_features_barplot.png')
plt.savefig(rank_features_barplot_path, dpi=300, bbox_inches='tight')

# ROC curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(6,6))
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {auc_rf:.3f})")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Random Forest")
plt.legend(loc="lower right")
plt.tight_layout()

roc_rf_path = os.path.join(figures_dir, "roc_curve_rf.png")
plt.savefig(roc_rf_path, dpi=300, bbox_inches='tight')

# ROC curve for Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

plt.figure(figsize=(6,6))
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC = {auc_lr:.3f})")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()

roc_lr_path = os.path.join(figures_dir, "roc_curve_lr.png")
plt.savefig(roc_lr_path, dpi=300, bbox_inches='tight')

# ROC curve for LR vs RF
plt.figure(figsize=(6,6))
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {auc_rf:.3f})")
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC = {auc_lr:.3f})")
plt.plot([0,1], [0,1], linestyle='--', color='gray')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: RF vs LR")
plt.legend(loc="lower right")
plt.tight_layout()

roc_compare_path = os.path.join(figures_dir, "roc_curve_rf_vs_lr.png")
plt.savefig(roc_compare_path, dpi=300, bbox_inches='tight')


print('Done with ML. ')


# step 5: COX

selected_RNA_genes = ['RNA_RB1', 'RNA_RETN', 'RNA_TP53']
selected_CNV_genes = ['CNV_LBX1', 'CNV_CEACAM8', 'CNV_PTEN']

rna_selected = rna_feat[selected_RNA_genes] if any(g.startswith('RNA_') for g in selected_RNA_genes) else pd.DataFrame()
cnv_selected = cnv_feat[selected_CNV_genes] if any(g.startswith('CNV_') for g in selected_CNV_genes) else pd.DataFrame()

X_selected = pd.concat([rna_selected, cnv_selected], axis=1, join='inner')

surv_df = clinical_df_matched.loc[X_selected.index, ['overall_survival', 'status']]
df_cox = pd.concat([X_selected, surv_df], axis=1)
df_cox_clean = df_cox.dropna(subset=['overall_survival', 'status'])

cox = CoxPHFitter()
cox.fit(df_cox_clean, duration_col='overall_survival', event_col='status')

cox_summary = cox.summary.rename(columns={
    "coef": "Coefficient",
    "exp(coef)": "Hazard_Ratio",
    "se(coef)": "SE",
    "p": "P_value",
    "z": "Z_score",
    "coef lower 95%": "Coef_95CI_lower",
    "coef upper 95%": "Coef_95CI_upper",
    "exp(coef) lower 95%": "HR_95CI_lower",
    "exp(coef) upper 95%": "HR_95CI_upper"
})

cox_out_path = os.path.join(results_dir, "cox_multivariable_results.csv")
cox_summary.to_csv(cox_out_path)
plt.figure(figsize=(7, 4))

y_pos = np.arange(len(cox_summary))
plt.errorbar(
    cox_summary["Hazard_Ratio"],
    y_pos,
    xerr=[
        cox_summary["Hazard_Ratio"] - cox_summary["HR_95CI_lower"],
        cox_summary["HR_95CI_upper"] - cox_summary["Hazard_Ratio"]
    ],
    fmt="o"
)

plt.axvline(1, linestyle="--")
plt.yticks(y_pos, cox_summary.index)
plt.xlabel("Hazard Ratio (95% CI)")
plt.title("Multivariable Cox Regression (GBM/LGG)")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "cox_forest_plot.png"), dpi=300, bbox_inches='tight')

cox_stats = pd.DataFrame({
    "Concordance_index": [cox.concordance_index_],
    "Log_likelihood": [cox.log_likelihood_],
    "Partial_AIC": [cox.AIC_partial_],
    "N_samples": [cox._n_examples],
    "N_events": [cox.event_observed.sum()]
})

cox_stats.to_csv(
    os.path.join(results_dir, "cox_model_stats.csv"),
    index=False
)

df_cox_clean = df_cox.dropna(
    subset=['overall_survival', 'status']
).copy()
df_cox_clean["risk_score"] = cox.predict_partial_hazard(df_cox_clean)

df_cox_clean[["risk_score"]].to_csv(
    os.path.join(results_dir, "cox_patient_risk_scores.csv")
)

print('Done with COX Analysis. ')


# step 6: final analysis

final_genes = ['RB1', 'RETN', 'TP53', 'LBX1', 'CEACAM8', 'PTEN']
deg_all = pd.read_csv(os.path.join(results_dir, "deg_all.csv"))
cnv_gain_all = pd.read_csv(os.path.join(results_dir, "cnv_gain_all.csv"))
cnv_loss_all = pd.read_csv(os.path.join(results_dir, "cnv_loss_all.csv"))
mutation_all = pd.read_csv(os.path.join(results_dir, "mutation_all.csv"))

deg_all = deg_all.rename(columns={"names": "Gene"})

for df in [deg_all, cnv_gain_all, cnv_loss_all, mutation_all]:
    df["Gene"] = df["Gene"].astype(str)


final_df = pd.DataFrame(index=final_genes)
deg_sub = deg_all.set_index("Gene")

final_df["RNA_logFC"] = deg_sub["logfoldchanges"]
final_df["RNA_adj_pval"] = deg_sub["pvals_adj"]

cnv_gain_sub = cnv_gain_all.set_index("Gene")

final_df["CNV_gain_GBM"] = cnv_gain_sub["GBM_gain"]
final_df["CNV_gain_LGG"] = cnv_gain_sub["LGG_gain"]
final_df["CNV_gain_FDR"] = cnv_gain_sub["FDR"]

cnv_loss_sub = cnv_loss_all.set_index("Gene")

final_df["CNV_loss_GBM"] = cnv_loss_sub["GBM_loss"]
final_df["CNV_loss_LGG"] = cnv_loss_sub["LGG_loss"]
final_df["CNV_loss_FDR"] = cnv_loss_sub["FDR"]

mutation_sub = mutation_all.set_index("Gene")

final_df["Mutation_GBM"] = mutation_sub["GBM_mut"]
final_df["Mutation_LGG"] = mutation_sub["LGG_mut"]
final_df["Mutation_FDR"] = mutation_sub["FDR"]

final_df = final_df.sort_values("Mutation_FDR")
final_path = os.path.join(results_dir, "final_6genes_multiomics_table.csv")
final_df.to_csv(final_path)

df = pd.read_csv(final_path, index_col=0)

## Fill NaNs with 0 for visualisation
df_fill = df.fillna(0)


df_annot = df_fill.copy()
for col in df_annot.columns:
    if "FDR" in col or "adj_pval" in col:
        df_annot[col] = df_annot[col].apply(lambda x: f"{x:.2e}")
    elif "logFC" in col:
        df_annot[col] = df_annot[col].apply(lambda x: f"{x:.2f}")
plt.figure(figsize=(12,6))
sns.heatmap(
    df_fill,              
    annot=df_annot,       
    fmt="",               
    cmap="vlag",
    linewidths=0.5,
    cbar=False,
    linecolor="lightgray"
)
plt.title("Multi-omics Overview of 6 Selected Genes")
plt.ylabel("Gene")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'final_omics_heatmap.png'), dpi=300, bbox_inches='tight')

print('Done with final Analysis. ')


# step 7: pahtaway enrichment of final genes

path_directory = os.path.join(results_dir, 'final_pathaway_enrichment')
os.makedirs(path_directory, exist_ok=True)

enrichr_libs = [
    "GO_Biological_Process_2021",
    "KEGG_2021_Human",
    'Reactome_2022'
]
    
for l in enrichr_libs:
    gp.enrichr(
        gene_list=final_genes,
        gene_sets=l,
        outdir=path_directory,
        no_plot=True,
        cutoff=0.05
    )

for l in enrichr_libs:
    file_name = f'{l}.human.enrichr.reports.txt'
    file_path = os.path.join(path_directory, file_name)

    path_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    keep_cols = [c for c in ['Term', 'Adjusted P-value', 'Overlap', 'P-value', 'Combined Score'] if c in path_df.columns]
    path_df = path_df[keep_cols].sort_values(by=keep_cols[1]).head(50)

    out_file = os.path.join(path_directory, f"top50_{l}_enrichment.csv")
    path_df.to_csv(out_file, index=False)
    

libraries = [
    "GO_Biological_Process_2021",
    "KEGG_2021_Human",
    'Reactome_2022'
]
for li in libraries:
    top_enr_path = os.path.join(path_directory, f"top50_{li}_enrichment.csv")
    enrich = pd.read_csv(top_enr_path)
    score_col = 'Adjusted P-value' if 'Adjusted P-value' in enrich.columns else 'P-value'
    enrich = enrich.sort_values(by=score_col).head(15)
    enrich['Term'] = (
    enrich['Term']
    .astype(str)
    .str.replace(r'\s*\(.*?\)', '', regex=True)
    .str.replace(r'\s*R-HSA.*$', '', regex=True)
    .str.strip()
)
    
    plt.figure(figsize=(8, 6))
    plt.barh(enrich['Term'], -enrich[score_col].apply(lambda x: np.log10(x)))
    plt.xlim(0, enrich[score_col].apply(lambda x: -np.log10(x)).max() * 1.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('-log10(Adjusted P-value)', fontsize=20)
    plt.ylabel('Enriched Term', fontsize=20)
    plt.title(f'Top Enriched Terms: {li}')
    plt.title(f'Top Enriched Terms: {li}')
    plt.gca().invert_yaxis()
    
    
    plot_file = os.path.join(figures_dir, f"{li}_Final_barplot.png")
    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')

print("Done with enrichment. ")
