# AFQ
Geothermal Heat Flow model of Africa using random forest regression

The code reproduces the resuls published in Fronieters in Science

The provided Jupyter notebook reproduces the model and and figures in the paper, including supplimentary material. The notebook also generates some additional tests and figures, not included in the paper.


# Instructions:

Please install the following enviroment and packages :

conda create -n AFQ python=3.7 anaconda

activate AFQ 

conda config --set channel_priority flexible

conda update --all



conda install -c defaults -c conda-forge numpy  xarray  dask pyproj pandas scipy  imageio


conda install -c defaults -c conda-forge  geopandas fiona rasterio matplotlib  cartopy basemap

conda install -c defaults -c conda-forge  gstools  pykrige affine scikit-learn

conda install -c defaults -c conda-forge  affine
conda install -c defaults -c conda-forge  xarray netcdf4 packaging gmt
conda install -c defaults -c conda-forge   pygmt==0.5.0

conda install -c conda-forge scikit-optimize

pip install agrid

conda install jupyter notebook

for windows users :

conda install scikit-learn-intelex

# Citing

If used in publication, please consider to cite :

'''
@article{Test123,
abstract = {We generate a geothermal heat flow model over Africa using random forest regression based
4 on sixteen different geophysical and geological quantities (among them are Moho depth, Curie
5 temperature depth, gravity anomalies, topography, and seismic wave velocities). The training of the
6 random forest is based on direct heat flow measurements collected in the compilation of Lucazeau
7 (2019). The final model reveals structures that are consistent with existing regional geothermal heat
8 flow information. It is interpreted with respect to the tectonic setup of Africa, and the influence of
9 the selection of training data and target observables is illustrated in the supplementary material.},
author = {M. Al-Aghbary, M. Sobh and C. Gerhards},
doi = {.......},
journal = {Frontiers in science},
publisher = {Ubiquity Press, Ltd.},
title = {{A geothermal heat flow model of Africa based on Random Forest Regression}},
year = {2022}
}
'''
