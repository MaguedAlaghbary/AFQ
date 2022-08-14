# Magued Al-Aghbary 2022
# mauged.alaghbary@outlook.com


# https://doi.org/
#

# MIT License#

# Copyright (c) 2022 Magued Al-Aghbary #

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions: #

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.#

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


If anaconda distro is not yet installed please install the latest veriosn from https://docs.anaconda.com/anaconda/install/
If GMT is not yet installed please install the latest veriosn from  https://www.generic-mapping-tools.org/download/


## steps to configure the anaconda enviroemnt##


conda create -n AFQ python=3.9 anaconda

activate AFQ 

conda config --set channel_priority flexible

conda update --all



conda install --force-reinstall -y -c defaults -c conda-forge xarray  dask pyproj pandas scipy 


conda install --force-reinstall -y -c defaults -c conda-forge  geopandas fiona rasterio matplotlib affine

conda install --force-reinstall -y -c defaults -c conda-forge  xarray netcdf4 packaging gmt pygmt

conda install --force-reinstall -y -c conda-forge scikit-optimize pooch jupyter notebook


conda install --force-reinstall -y -c anaconda seaborn


conda update --force-reinstall -y --all