## **Glioma Multi-Omics Project**



***Author: Amir Mahdi Taghizadeh***

***Date: 2025***





**This repository presents a comprehensive multi-omics analysis of glioma (GBM vs LGG) integrating RNA-seq, mutation, and CNV data from TCGA. The project includes differential analysis, feature selection, machine learning classification, Cox regression, and pathway enrichment.**





⚠️ Note: Raw data, processed patient data, and scripts are not included in this repository to ensure privacy and sensitivity. Only figures and result tables are provided.





###### **Project Highlights**



* RNA-seq Analysis: normalization, PCA \& UMAP, DEG analysis, volcano plots



* Mutation Analysis: mutation frequency comparison, FDR-based significance, heatmaps and barplots



* CNV Analysis: identification of significant gains/losses in GBM vs LGG



* Multi-Omics Integration: overlapping genes across RNA, CNV, mutation data, feature selection



* Machine Learning: Random Forest \& Logistic Regression classifiers, feature importance, ROC curves, Venn diagrams



* Survival Analysis: multivariable Cox regression, hazard ratios, forest plots, patient risk scores



* Pathway Enrichment: GO, KEGG, Reactome enrichment for top multi-omics genes





###### 

###### **Repository Structure**



**glioma\_project/**

**│**

**├─ figures/**                  # Generated figures from analyses

**├─ results/**                  # Output tables and performance metrics

**├─ data/**                     # NOT INCLUDED due to sensitivity

**├─ scripts/**                  # NOT INCLUDED due to privacy reasons

**└─ README.md**                 # This file







###### **Figures \& Results**



Included files:



* Heatmaps: mutation, CNV, RNA-seq, multi-omics overview



* Barplots: top features, pathway enrichment



* ROC curves: Random Forest and Logistic Regression models



* Cox regression forest plots \& risk scores



* Venn diagrams for overlapping ML features



* CSV tables for DEGs, CNVs, mutations, ML feature importance, and Cox statistics







###### **Dependencies**



*pandas, numpy, matplotlib, seaborn, scanpy, scipy, statsmodels, scikit-learn, lifelines, gseapy, matplotlib-venn*





Install via pip:



*pip install pandas numpy matplotlib seaborn scanpy scipy statsmodels scikit-learn lifelines gseapy matplotlib-venn*







###### **Usage**



*Note: The scripts themselves are not included. This section is for reference if raw data were available.*



Place raw TCGA data in data/raw/

Run pipelines in order (if scripts were included):



1. *python cnv\_pipeline.py*
2. *python mutation\_pipeline.py*
3. *python rna\_seq\_pipeline.py*
4. *python combined\_pipeline.py*







###### **Contact**



*Author: Amir Mahdi Taghizadeh*

*GitHub: https://github.com/AmirTaghizadeh03*



*Email: \[amir.taghizadeh03@outlook.com]*







