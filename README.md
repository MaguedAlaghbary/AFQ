# AFQ
Geothermal Heat Flow model of Africa using random forest regression

The code reproduces the resuls published in Fronieters in Science

The provided Jupyter notebook reproduces the model and and figures in the paper, including supplimentary material. The notebook also generates some additional tests and figures, not included in the paper.


# Instructions:

Please install the following enviroment and packages

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
