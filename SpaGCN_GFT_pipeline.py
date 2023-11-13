import os, csv, re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings
import cv2
import time
from sklearn import metrics
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# Select two datasets for grid tuning
dataset_list = ['151507', '151509', '151510',
                '151669', '151671', '151672',
                '151673', '151674', '151675',
                '151676']
perf_df = pd.DataFrame(0, index=dataset_list, columns=['ARI'])
for dataset in dataset_list:
    # load data
    adata = sc.read_visium("/data/bqliu_data/jxliu_data/projects/SpaGFT/SpaGFT_SpaGCN/data/DLPFC/" + dataset)
    img = cv2.imread(
        "/data/bqliu_data/jxliu_data/projects/SpaGFT/SpaGFT_SpaGCN/data/DLPFC/image/" + dataset + '_full_image.tif')
    adata.obs['x_array'] = adata.obs['array_row']
    adata.obs['y_array'] = adata.obs['array_col']
    adata.obs['x_pixel'] = adata.obsm['spatial'][:, 1]
    adata.obs['y_pixel'] = adata.obsm['spatial'][:, 0]
    x_array = adata.obs["x_array"].tolist()
    y_array = adata.obs["y_array"].tolist()
    x_pixel = adata.obs["x_pixel"].tolist()
    y_pixel = adata.obs["y_pixel"].tolist()

    # Run SpaGCN
    n_clusters = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7
    adata.obs["pred"] = spg.detect_spatial_domains_ez_mode_gft(adata, img, x_array, y_array, x_pixel, y_pixel,
                                                               n_clusters=n_clusters,
                                                               histology=True,
                                                               r_seed=100,
                                                               t_seed=100,
                                                               n_seed=100,
                                                               num_fcs=1000)

    adata.obs["pred"] = adata.obs["pred"].astype('category')
    adata.obs["refined_pred"] = spg.spatial_domains_refinement_ez_mode(sample_id=adata.obs.index.tolist(),
                                                                       pred=adata.obs["pred"].tolist(),
                                                                       x_array=x_array,
                                                                       y_array=y_array, shape="hexagon")
    adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')

    # Load meta data
    df_meta = pd.read_csv('/data/bqliu_data/jxliu_data/projects/SpaGFT/SpaGFT_SpaGCN/data/DLPFC/ground_truth/' +
                          dataset + '_annotation.csv',
                          index_col=0)
    df_meta_layer = df_meta
    valid_spots = df_meta.dropna().index.tolist()
    adata.obs['ground_truth'] = df_meta_layer.iloc[:, 0].values

    # Calculate ARI
    gd_list = adata.obs.loc[valid_spots, 'ground_truth'].values.tolist()
    pred_list = adata.obs.loc[valid_spots, "refined_pred"].values.astype(str).tolist()
    ari = metrics.adjusted_rand_score(gd_list, pred_list)
    print(dataset, ari)
    perf_df.loc[dataset, 'ARI'] = ari

    # Save results
    results = adata.obs['ground_truth'].copy()
    results = pd.DataFrame(results)
    results.to_csv(
        f"/data/bqliu_data/jxliu_data/projects/SpaGFT/SpaGFT_SpaGCN/results/DLPFC/SpaGCN+SpaGFT_{dataset}_results.csv")
perf_df.to_csv(
    "/data/bqliu_data/jxliu_data/projects/SpaGFT/SpaGFT_SpaGCN/results//SpaGCN+SpaGFT_independent_results.csv")
