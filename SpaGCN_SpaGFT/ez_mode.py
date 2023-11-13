import numpy as np
import sklearn.preprocessing

from . SpaGCN import *
from . util import *
from . calculate_moran_I import *
from . calculate_adj import *
import pandas as pd
import sklearn
from sklearn.preprocessing import scale
import numpy as np
import os
import scanpy as sc
from scipy.sparse import csr_matrix
import scipy.sparse as ss
import anndata as ad
import random
import warnings
from sklearn.neighbors import kneighbors_graph
import networkx as nx


def create_degree_mtx(diag):
    diag = np.array(diag)
    diag = diag.flatten()
    row_index = list(range(diag.size))
    col_index = row_index
    sparse_mtx = ss.coo_matrix((diag, (row_index, col_index)),
                               shape=(diag.size, diag.size))

    return sparse_mtx

def get_laplacian_mtx(adata,
                      num_neighbors=6,
                      spatial_key=['array_row', 'array_col'],
                      normalization=False):
    if spatial_key in adata.obsm_keys():
        adj_mtx = kneighbors_graph(adata.obsm[spatial_key],
                                   n_neighbors=num_neighbors)
    elif set(spatial_key) <= set(adata.obs_keys()):
        adj_mtx = kneighbors_graph(adata.obs[spatial_key],
                                   n_neighbors=num_neighbors)
    else:
        raise KeyError("%s is not avaliable in adata.obsm_keys" % \
                       spatial_key + " or adata.obs_keys")

    adj_mtx = nx.adjacency_matrix(nx.Graph(adj_mtx))
    # Degree matrix
    deg_mtx = adj_mtx.sum(axis=1)
    deg_mtx = create_degree_mtx(deg_mtx)
    # Laplacian matrix
    # Whether need to normalize laplcian matrix
    if not normalization:
        lap_mtx = deg_mtx - adj_mtx
    else:
        deg_mtx = np.array(adj_mtx.sum(axis=1)) ** (-0.5)
        deg_mtx = create_degree_mtx(deg_mtx)
        lap_mtx = ss.identity(deg_mtx.shape[0]) - deg_mtx @ adj_mtx @ deg_mtx

    return lap_mtx
def obtain_freq_spots(adata,
				lap_mtx,
                n_fcs=256,
				c=0):
	# select highly varibale genes
	tmp_adata = adata.copy()
	if ss.issparse(tmp_adata.X):
		X = tmp_adata.X.A
	else:
		X = tmp_adata.X
	X = sklearn.preprocessing.normalize(X, norm='max', axis=0)
	X = np.matmul(X, X.transpose())
	X = sklearn.preprocessing.normalize(X)
	n_fcs = min(n_fcs, adata.shape[0] - 1)
    # Obtain Fourier modes and corresponding eigenvalues
	v0 = lap_mtx.shape[0] * [1 / np.sqrt(lap_mtx.shape[0])]
	eigvals, eigvecs = ss.linalg.eigsh(lap_mtx,
                                       k=n_fcs,
                                       which='SM',
                                       v0=v0)
	power = [1 / (1 + c * eigv) for eigv in eigvals]
	eigvecs = np.matmul(eigvecs, np.diag(power))
	eigvecs_T = eigvecs.transpose()
	freq_mtx = np.matmul(eigvecs_T, X)
	freq_mtx = freq_mtx[1:, :]
	freq_mtx = sklearn.preprocessing.normalize(freq_mtx,
											   norm='l2',
											   axis=0)
	freq_mtx = freq_mtx.transpose()
	return freq_mtx, eigvecs_T

def detect_spatial_domains_ez_mode(adata, img, x_array, y_array, x_pixel, y_pixel, n_clusters,
								   histology=True, s=1, b=49, p=0.5, r_seed=100, t_seed=100, n_seed=100):
	adj=calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, histology=histology)
	prefilter_genes(adata,
					min_cells=3) # avoiding all genes are zeros
	prefilter_specialgenes(adata)
	sc.pp.normalize_per_cell(adata)
	sc.pp.log1p(adata)
	l=search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
	res=search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05,
				   max_epochs=20,
				   r_seed=r_seed,
				   t_seed=t_seed,
				   n_seed=n_seed)
	clf=SpaGCN()
	clf.set_l(l)
	random.seed(r_seed)
	torch.manual_seed(t_seed)
	np.random.seed(n_seed)
	clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
	y_pred, prob=clf.predict()
	return y_pred


def detect_spatial_domains_ez_mode_gft(adata,
									   img,
									   x_array,
									   y_array,
									   x_pixel,
									   y_pixel,
									   n_clusters,
									   histology=True,
									   s=1, b=49, p=0.5, r_seed=100, t_seed=100, n_seed=100,
									   num_fcs=500, c=1):
	adj=calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, histology=histology)
	loc_df = pd.DataFrame(0,
						   index=adata.obs_names,
					       columns=['x', 'y'])
	loc_df.loc[:, 'x'] = x_pixel
	loc_df.loc[:, 'y'] = y_pixel
	# fourier transform
	adata.obs.loc[:, ['x', 'y']] = loc_df
	prefilter_genes(adata, min_cells=3)# avoiding all genes are zeros
	prefilter_specialgenes(adata)
	sc.pp.normalize_per_cell(adata)
	sc.pp.log1p(adata)

	lap_mtx = get_laplacian_mtx(adata, spatial_key=['x', 'y'], num_neighbors=6, normalization=False)
	freq_spots, ut = obtain_freq_spots(adata=adata, lap_mtx=lap_mtx, n_fcs=num_fcs, c=c)
	freq_spots = freq_spots * (adata.X.max() / freq_spots.max())
	l=search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

	res = search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=1e-5, lr=0.01,
					 max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed, emb2=freq_spots)
	clf=SpaGCN()
	clf.set_l(l)
	random.seed(r_seed)
	torch.manual_seed(t_seed)
	np.random.seed(n_seed)

	clf.train(adata=adata,
			  adj=adj,
			  emb2=freq_spots,
			  init_spa=True,init="louvain",res=res, tol=1e-5, lr=0.01, max_epochs=200,
			  )
	y_pred, prob=clf.predict()
	return y_pred

def spatial_domains_refinement_ez_mode(sample_id, pred, x_array, y_array, shape="hexagon"):
	adj_2d=calculate_adj_matrix(x=x_array,y=y_array, histology=False)
	refined_pred=refine(sample_id=sample_id, pred=pred, dis=adj_2d, shape=shape)
	return refined_pred

def plot_spatial_domains_ez_mode(adata, domain_name, x_name, y_name, plot_color,size, show=False, save=True,save_dir="./domains.png"):
	num_celltype=len(adata.obs[domain_name].unique())
	adata.uns[domain_name+"_colors"]=list(plot_color[:num_celltype])
	ax=sc.pl.scatter(adata,alpha=1,x=x_name,y=y_name,color=domain_name,show=False,size=size)
	ax.set_aspect('equal', 'box')
	ax.axes.invert_yaxis()
	if save:
		plt.savefig(save_dir, dpi=600)
	plt.close()

def detect_SVGs_ez_mode(adata, target, x_name, y_name, domain_name, min_in_group_fraction, min_in_out_group_ratio, min_fold_change):
	adj_2d=calculate_adj_matrix(x=adata.obs[x_name].tolist(), y=adata.obs[y_name].tolist(), histology=False)
	start, end= np.quantile(adj_2d[adj_2d!=0],q=0.001), np.quantile(adj_2d[adj_2d!=0],q=0.1)
	r=search_radius(target_cluster=target, cell_id=adata.obs.index.tolist(), x=adata.obs[x_name].tolist(), y=adata.obs[y_name].tolist(), pred=adata.obs[domain_name].tolist(), start=start, end=end, num_min=10, num_max=14,  max_run=100)
	nbr_domians=find_neighbor_clusters(target_cluster=target,
								   cell_id=adata.obs.index.tolist(), 
								   x=adata.obs[x_name].tolist(), 
								   y=adata.obs[y_name].tolist(), 
								   pred=adata.obs[domain_name].tolist(), 
								   radius=r,
								   ratio=1/2)
	nbr_domians=nbr_domians[0:3]
	de_genes_info=rank_genes_groups(input_adata=adata,
								target_cluster=target,
								nbr_list=nbr_domians, 
								label_col=domain_name, 
								adj_nbr=True, 
								log=True)
	de_genes_info=de_genes_info[(de_genes_info["pvals_adj"]<0.05)]
	filtered_info=de_genes_info
	filtered_info=filtered_info[(filtered_info["pvals_adj"]<0.05) &
							(filtered_info["in_out_group_ratio"]>min_in_out_group_ratio) &
							(filtered_info["in_group_fraction"]>min_in_group_fraction) &
							(filtered_info["fold_change"]>min_fold_change)]
	filtered_info=filtered_info.sort_values(by="in_group_fraction", ascending=False)
	filtered_info["target_dmain"]=target
	filtered_info["neighbors"]=str(nbr_domians)
	print("SVGs for domain ", str(target),":", filtered_info["genes"].tolist())
	return filtered_info

def plot_SVGs_ez_mode(adata, gene_list, x_name, y_name, plot_color, size, show=False, save=True, save_dir="./"):
	for g in gene_list:
		adata.obs["exp"]=adata.X[:,adata.var.index==g]
		ax=sc.pl.scatter(adata,alpha=1,x=x_name,y=y_name,color="exp",title=g,color_map=plot_color,show=show,size=size)
		ax.set_aspect('equal', 'box')
		ax.axes.invert_yaxis()
		if save:
			plt.savefig(save_dir+g+".png", dpi=600)
		plt.close()

def detect_meta_genes_ez_mode(adata, target, x_name, y_name, domain_name,start_gene, use_raw=False):
    meta_name, meta_exp=find_meta_gene(input_adata=adata,
                    pred=adata.obs[domain_name].tolist(),
                    target_domain=target,
                    start_gene=start_gene,
                    mean_diff=0,
                    early_stop=True,
                    max_iter=3,
                    use_raw=use_raw)
    print("Meta gene:", meta_name)
    return meta_exp

def plot_meta_genes_ez_mode(adata, x_name, y_name, meta_name, plot_color, size, show=False, save=True, save_dir="./"):
	ax=sc.pl.scatter(adata,alpha=1,x=x_name,y=y_name,color=meta_name,title="Meta gene",color_map=plot_color,show=show,size=size)
	ax.set_aspect('equal', 'box')
	ax.axes.invert_yaxis()
	if save:
		plt.savefig(save_dir+"meta_gene.png", dpi=600)
	plt.close()

