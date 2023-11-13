SpaGCN can ntegrating gene expression and histology to identify spatial domains and spatially variable genes using graph convolutional networks. SpaGCN_SpaGFT, is an improved version for orignal SpaGCN by extending orignal gene expresssion features with Fourier features of spots. Then, the new feature matrix will be used to predict spatial domain cluster labels.

## How to install SpaGCN_SpaGFT?
SpaGCN_SpaGFT is based on SpaGCN.  To install  it, make sure you have [PyTorch](https://pytorch.org/) and [scanpy](https://scanpy.readthedocs.io/en/stable/) installed.

Create a conda environment
```
conda create -n spagcn_spagft_env python==3.8
```
and activate the environment
```
conda activate spagcn_spagft_env
```

Next, install SpaGCN_SpaGFT
```
python setup.py install
```

## Run SpaGCN_SpaGFT.
To begin with, import following modules in python environment.
```
import scanpy as sc
import SpaGCN_SpaGFT as spg
import cv2
```
Next, load your spatial transcriptomics data and corresponding image.
```
adata = sc.read_visium(PATH_TO_YOUR_DATASET)
img = cv2.imread(PATH_TO_TIFF_IMAGE)
# adjust spatial information
adata.obs['x_array'] = adata.obs['array_row']
adata.obs['y_array'] = adata.obs['array_col']
adata.obs['x_pixel'] = adata.obsm['spatial'][:, 1]
adata.obs['y_pixel'] = adata.obsm['spatial'][:, 0]
x_array = adata.obs["x_array"].tolist()
y_array = adata.obs["y_array"].tolist()
x_pixel = adata.obs["x_pixel"].tolist()
y_pixel = adata.obs["y_pixel"].tolist()
```

Run detect_spatial_domains_ez_mode_gft function
```
n_clusters = NUMBER_OF_CLUSTERS
adata.obs["pred"] = spg.detect_spatial_domains_ez_mode_gft(adata, img,x_array, y_array, x_pixel, y_pixel, n_clusters=n_clusters, histology=True,r_seed=100, t_seed=100, n_seed=100, num_fcs=1000)
```

Fianally, obtain the refinement results by
```
adata.obs["pred"] = adata.obs["pred"].astype('category')
adata.obs["refined_pred"] = spg.spatial_domains_refinement_ez_mode(sample_id=adata.obs.index.tolist(),pred=adata.obs["pred"].tolist(), x_array=x_array, y_array=y_array, shape="hexagon")
```
