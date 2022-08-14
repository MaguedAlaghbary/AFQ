# !/usr/bin/env python
"""Utility script with functions to be used with the results of GridSearchCV.

**plot_grid_search** plots as many graphs as parameters are in the grid search results.

**table_grid_search** shows tables with the grid search results.

"""

# Helper functions

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pprint
from IPython.display import display
import numpy as np
from scipy.spatial import cKDTree as KDTree
import pyproj as proj
from pyproj import Transformer
import rasterio
from rasterio.warp import Resampling
from scipy import stats, interpolate

milli= 0.001

def plot_grid_search(clf, catgeories):

    # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
    # As it is frequent to have more than one combination with the same max score,
    # the one with the least mean fit time SHALL appear first.
    cv_results = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score', 'mean_fit_time'])

    # Get parameters
    parameters=cv_results['params'][0].keys()

    # Calculate the number of rows and columns necessary
    #rows = -(-len(parameters) // 2)
    #columns = min(len(parameters), 2)

    rows = len(parameters)
    columns = 1
    # Create the subplot
    fig = make_subplots(rows=rows, cols=columns)
    # Initialize row and column indexes
    #row = 1
    column = 1

    # For each of the parameters
    for parameter, category, row in zip(parameters, catgeories, range(rows)):

        # As all the graphs have the same traces, and by default all traces are shown in the legend,
        # the description appears multiple times. Then, only show legend of the first graph.
        #if row == 1 and column == 1:
         #   show_legend = True
        #else:
         #   show_legend = False
        mean_test_score = cv_results[cv_results['rank_test_score'] != 1].round(1)
        rank_1 = cv_results[cv_results['rank_test_score'] == 1].round(1)


        show_legend = True

        fig.update_xaxes(title_text=parameter, row=row, col=column)
        fig.update_yaxes(title_text='Score', row=row, col=column)

        # Check the linearity of the series
        # Only for numeric series
        if pd.to_numeric(cv_results['param_' + parameter], errors='coerce').notnull().all():
            x_values = cv_results['param_' + parameter].sort_values().unique().tolist()
            r = stats.linregress(x_values, range(0, len(x_values))).rvalue
            # If not so linear, then represent the data as logarithmic
            if r < 0.86:
                fig.update_xaxes(type='log', row=row, col=column)

        # Increment the row and column indexes

        row += 1

            # Show first the best estimators
        fig.update_layout(legend=dict(traceorder='reversed'),
                          width=columns * 500 + 100,
                          height=rows * 500,
                          title='Best score: {:.6f} with {}'.format(cv_results['mean_test_score'].iloc[0],
                                                                    str(cv_results['params'].iloc[0]).replace('{',
                                                                                                              '').replace(
                                                                        '}', '')).replace('regressor__', '').replace('regressor__',
                                                                        '').replace('regressor__', ''),
                          hovermode='closest',
                          template='none')
        if category:
             # Mean test score


            fig.add_trace(go.Box(
                name='Mean test score',
                x=mean_test_score['param_' + parameter],
                y=mean_test_score['mean_test_score']*-1,
                #color=mean_test_score['param_' + parameter],
                boxpoints='all',
                notched=True, # used notched shape
                #boxmean=True,
                boxmean='sd',
                jitter=0.5,
                whiskerwidth=0.2,

                line=dict(width=1),
                text=mean_test_score['params'].apply(
                   lambda x: pprint.pformat(x, width=-1).replace('{', '').replace('}', '').replace('\n', '<br />')),
                showlegend=show_legend,
                ),
                row=row,
                col=column)

            fig.add_trace(go.Scatter(
                name='Best estimators',
                x=rank_1['param_' + parameter],
                y=rank_1['mean_test_score']*-1,
                mode='markers',
                marker=dict(size=rank_1['mean_fit_time'],
                            color='Crimson',
                            sizeref=2. * cv_results['mean_fit_time'].max() / (40. ** 2),
                            sizemin=4,
                            sizemode='area'),
                text=rank_1['params'].apply(str),
                showlegend=show_legend),
                row=row,
                col=column)


        else:
            # Mean test score
            fig.add_trace(go.Scatter(
                name='Mean test score',
                x=mean_test_score['param_' + parameter],
                y=mean_test_score['mean_test_score']*-1,
                mode='markers',
                marker=dict(size=mean_test_score['mean_fit_time'],
                            color='SteelBlue',
                            sizeref=2. * cv_results['mean_fit_time'].max() / (40. ** 2),
                            sizemin=4,
                            sizemode='area'),
                text=mean_test_score['params'].apply(
                    lambda x: pprint.pformat(x, width=-1).replace('{', '').replace('}', '').replace('\n', '<br />')),
                legendgroup=str(row),
                showlegend=show_legend),
                row=row,
                col=column)

            # Best estimators


            fig.add_trace(go.Scatter(
                name='Best estimators',
                x=rank_1['param_' + parameter],
                y=rank_1['mean_test_score']*-1,
                mode='markers',
                marker=dict(size=rank_1['mean_fit_time'],
                            color='Crimson',
                            sizeref=2. * cv_results['mean_fit_time'].max() / (40. ** 2),
                            sizemin=4,
                            sizemode='area'),
                text=rank_1['params'].apply(str),
                legendgroup=str(row),
                showlegend=show_legend),
                row=row,
                col=column)


    fig.show()


def table_grid_search(clf, all_columns=False, all_ranks=False, save=True):

    # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
    # As it is frequent to have more than one combination with the same max score,
    # the one with the least mean fit time SHALL appear first.
    cv_results = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score', 'mean_fit_time'])

    # Reorder
    columns = cv_results.columns.tolist()
    # rank_test_score first, mean_test_score second and std_test_score third
    columns = columns[-1:] + columns[-3:-1] + columns[:-3]
    cv_results = cv_results[columns]

    if save:
        cv_results.to_csv('--'.join(cv_results['params'][0].keys()) + '.csv', index=True, index_label='Id')

    # Unless all_columns are True, drop not wanted columns: params, std_* split*
    if not all_columns:
        cv_results.drop('params', axis='columns', inplace=True)
        cv_results.drop(list(cv_results.filter(regex='^std_.*')), axis='columns', inplace=True)
        cv_results.drop(list(cv_results.filter(regex='^split.*')), axis='columns', inplace=True)

    # Unless all_ranks are True, filter out those rows which have rank equal to one
    if not all_ranks:
        cv_results = cv_results[cv_results['rank_test_score'] == 1]
        cv_results.drop('rank_test_score', axis = 'columns', inplace = True)
        cv_results = cv_results.style.hide_index()

    display(cv_results)


# Haversine arc distance
def distance(lat1, lon1, lat2, lon2):
    '''
    Haversine formula returns distance between pairs of coordinates.
    coordinates as numpy arrays, lists or real
    The haversine formula determines the great-circle distance between
    two points on a sphere given their longitudes and latitudes
    '''
    p = 0.017453292519943295 # pi/180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p)) / 2
    return 12742.0176 * np.arcsin(np.sqrt(a)) # returns in km



#function to make inverse distance weight interpolation

N = 10000
Ndim = 2
Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
Nnear = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
leafsize = 10
eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
p = 1  # weights ~ 1 / distance**p
cycle = .25
seed = 1

class Invdisttree:

    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]


# adapted from a script agrid https://github.com/TobbeTripitaka/agrid
# paper can be accssed from https://www.doi.org/10.5334/JORS.287


verbose = True
def _check_if_in(xx, yy, margin=2):
    '''Generate an array of the condition that coordinates
    are within the model or not.
    xx = list or array of x values
    yy = list or array of y values
    margin = extra cells added to mitigate extrapolation of
    interpolated values along the frame
    returns boolean array True for points within the frame.
    '''
    res = [xx.max()- xx.min(), yy.max(), yy.min() ]
    x_min = xx.min() - margin * res[0]
    x_max = xx.max() + margin * res[0]
    y_min = yy.min() - margin * res[1]
    y_max = yy.max() + margin * res[1]
    return (xx > x_min) & (xx < x_max) & (yy > y_min) & (yy < y_max)

def _set_meridian( x_array, center_at_0=True):
    '''
    Sloppy function to change longitude values from [0..360] to [-180..180]
    x_array :   Numpy array with longitude values (X)
    center_at_0 : Bool select direction of conversion.
    lon=(((lon + 180) % 360) - 180)
    '''
    if center_at_0:
        x_array[x_array > 180] = x_array[x_array > 180] - 360
    else:
        x_array[x_array < 0] = x_array[x_array < 0] + 360
    return x_array




def read_raster(    f_name,
                    crs_from=None,
                    crs_to = None,
                    ds= None,
                    source_extra=500,
                    resampling=None,
                    sub_sampling=None,
                    sub_window=None,
                    num_threads=4,
                    no_data=None,
                    rgb_convert=True,
                    bit_norm=255,
                    **kwargs):
        '''Imports raster in geotiff format to grid.

        Using gdal/rasterio warp to transform raster to right crs_toand extent.

        sub_sampling  -- integer to decrease size of input raster and speed up warp

        Resampling -- Interpolation method
                Options for resampling:
                    Resampling.nearest,
                    Resampling.bilinear,
                    Resampling.cubic,
                    Resampling.cubic_spline,
                    Resampling.lanczos,
                    Resampling.average

        A window is a view onto a rectangular subset of a raster
        dataset and is described in rasterio by column and row offsets
        and width and height in pixels. These may be ints or floats.
        Window(col_off, row_off, width, height)

        Returns numpy array.
        '''
        in_raster = rasterio.open(f_name)



        if crs_from is None:
            crs_from = in_raster.crs
            if verbose:
                print(crs_from)

        if resampling is None:
            resampling = Resampling.nearest

        if verbose:
            print('Raster bounds:', in_raster.bounds, in_raster.shape)

        dst_crs_to= crs_to

        if sub_sampling in (None, 0, 1):
            sub_sampling = 1

        raster_shape = (in_raster.count, in_raster.height //
                        sub_sampling, in_raster.width // sub_sampling)
        # window=Window.from_slices(sub_window)
        source = in_raster.read(out_shape=raster_shape)

        if sub_window is None:
            pass
        else:
            print('Window not implimented yet.')

        # Numpy arrays are indexed rows and columns (y,x)
        ny = len(ds.Y)
        nx = len(ds.X)
        nn = (ny, nx)

        src_transform = rasterio.transform.from_bounds(*in_raster.bounds, raster_shape[2],
                                                       raster_shape[1])

        dst_array = np.zeros((in_raster.count, *nn))

        left = ds.X.values.min()
        right =   ds.X.values.max()
        down =  ds.Y.values.min()
        up =   ds.Y.values.max()

        transform = rasterio.transform.from_bounds(
                    left, up, right, down, nx, ny)

        rasterio.warp.reproject(
            source,
            dst_array,
            src_transform=src_transform,
            src_crs=crs_from,
            dst_transform=transform,
            dst_crs=dst_crs_to,
            resampling=resampling,
            source_extra=source_extra,
            num_threads=num_threads,
            **kwargs)

        if (rgb_convert and in_raster.count > 2):
            dst_array = reshape_as_image(dst_array / bit_norm).astype(float)

        if in_raster.count == 1:
            dst_array = dst_array[0, :, :]

        if no_data is not None:
            dst_array[dst_array == no_data] = np.nan



        return dst_array


def read_grid(
                  f_name,
                  xyz=('x', 'y', 'z'),
                  ds=None,
                  interpol='linear',
                  crs_from=None,
                  crs_to=None,
                  use_dask=None,
                  dask_chunks=None,
                  read_dask_dict=None,
                  bulk=False,
                  extension='.nc',
                  ep_max=10,
                  pad_around=False,
                  sort=True,
                  only_frame=True,
                  deep_copy=False,
                  set_center=False,
                  regex_index=None,
                  def_depths=None,
                  verbose=False,
                  return_list=False,
                  names_to_numbers=True,
                  depth_factor=1,
                  name_i=-1,
                  **kwargs):
        '''Read irregular (or regular) grid. Resampling and interpolating.

        Keyword arguments:
        f_name : string path to dir or file. Ii list, it is read as list of paths to files.
        xyz --- Sequence with x, y and data labels
        interpol --- Interpolation method, e.g cubic, nearest
        only_frame --- Speeds up interpolation by only
                regard points within the grid extent (+ margins)

        Returns numpy array'''

        if crs_from is None:
            crs_from = crs_from

        if crs_to is None:
            crs_to= crs_to

        ny = len(ds.Y)
        nx = len(ds.X)
        nn = (ny, nx)

        if use_dask is None:
            use_dask = use_dask

        if bulk:
            if isinstance(f_name, str):
                assert os.path.isdir(
                    f_name), 'Please provide path to directory containing files.'
                f_names = glob.glob(f_name + '*' + extension)
            elif isinstance(f_names, list):
                for f_name in f_names:
                    assert os.path.isfile(f_name), '%s Is not a file.' % f_name
            else:
                f_names = []

            if sort:
                f_names.sort(key=str.lower)
        else:
            if isinstance(f_name, str):
                assert os.path.isfile(
                    f_name), 'Please provide path to a file, not directory. Or set bulk=True'
            f_names = [f_name]

        if names_to_numbers:
            try:
                f_names_float = [re.findall(r"[-+]?\d*\.\d+|\d+", _)
                                 for _ in f_names]
                f_names_float = [float(_[name_i]) *
                                 depth_factor for _ in f_names_float]
            except:
                names_to_numbers = False
                f_names_float = None

        i_grid = np.empty(nn + (len(f_names),))
        for i, f in enumerate(f_names):
            if verbose:
                print('%s/%s' % (i + 1, len(f_names)), f)

            if isinstance(f_name, str):
                array = xr.open_dataset(f, chunks=read_dask_dict).copy(
                    deep=deep_copy)
            else:
                array = f_name.copy(deep=deep_copy)

            x = array[xyz[0]].values
            y = array[xyz[1]].values

            # Set longitude, case from 0 to -360 insetad of -180 to 180
            if set_center:
                x = _set_meridian(x)

            xx, yy = np.meshgrid(x, y)  # x, y
            #xv, yv = proj.transform(proj.Proj(crs_from),
             #                       proj.Proj(crs), xx, yy)

            transformer = Transformer.from_crs(crs_from, crs_to)
            xv, yv =   transformer.transform(xx, yy)

            zv = array[xyz[2]].values
            n = zv.size

            zi = np.reshape(zv, (n))
            xi = np.reshape(xv, (n))
            yi = np.reshape(yv, (n))

            # Check and interpolate only elements in the frame
            if only_frame:
                is_in = _check_if_in(xi, yi)
                xi = xi[is_in]
                yi = yi[is_in]
                zi = zi[is_in]



            if interpol == "IDW":
                X = np.array([xv, yv]).T
                coords = np.stack([  ds.coords['XV'].values.flatten() ,
                                   ds.coords['YV'].values.flatten() ], axis=1)
                invdisttree = Invdisttree( X,
                                          zi.reshape(-1,1), leafsize=leafsize, stat=1 )
                arr = invdisttree( coords,
                                  nnear=Nnear, eps=eps, p=p )
                arr = arr.ravel().reshape(nn)


            else:

                arr = interpolate.griddata((xi, yi),
                                           zi,
                                           (ds.coords['XV'],
                                            ds.coords['YV']),
                                           method=interpol,
                                           **kwargs)

            if pad_around:
                for i in range(ep_max)[::-1]:
                    arr[:, i][np.isnan(arr[:, i])] = arr[
                        :, i + 1][np.isnan(arr[:, i])]
                    arr[:, -i][np.isnan(arr[:, -i])] = arr[:, -
                                                           i - 1][np.isnan(arr[:, -i])]
                    arr[i, :][np.isnan(arr[i, :])] = arr[
                        i + 1, :][np.isnan(arr[i, :])]
                    arr[-i, :][np.isnan(arr[-i, :])] = arr[-i -
                                                           1, :][np.isnan(arr[-i, :])]

            i_grid[..., i] = arr

            if len(f_names) is 1:
                i_grid = np.squeeze(i_grid, axis=2)

        if dask_chunks is None:
            if use_dask:
                i_grid = da.from_array(i_grid, chunks=dask_chunks)

            dask_chunks = (nx // chunk_n,) * i_grid.ndim



        if return_list:
            if names_to_numbers:
                f_names = f_names_float
            return i_grid, f_names
        else:
            return i_grid






def read_numpy(   i = 0,
                  j = 1,
                  k = 2,
                  data = None,
                  ds=None,
                  interpol='linear',
                  crs_from=None,
                  crs_to=None,
                  use_dask=None,
                  dask_chunks=None,
                  pad_around=False,
                  only_frame=True,
                  set_center=False,
                  verbose=False,
                  z_factor=1,
                  **kwargs):
        '''Read numpy array and interpolate to grid.

        Keyword arguments:
        x,y,z numpy arrays of same size, eg, A[0,:], A[1,:], A[2,:]
        Returns numpy array


        kwargs to interpolation
        '''


        if data is not None:
            x = data[:,i]
            y = data[:,j]
            z = data[:,k]

        assert(np.shape(x)==np.shape(y)==np.shape(z)), 'x, y, and z must have the same shape.'





        if crs_from is None:
            crs_from = crs_from

        if crs_to is None:
            crs_to = crs_to


        if verbose:
            print('Shape:', np.shape(x))

        if z_factor is not 1:
            z *= z_factor


        # Set longitude, case from 0 to -360 insetad of -180 to 180
        if set_center:
            x = _set_meridian(x)

        transformer = Transformer.from_crs(crs_from, crs_to)
        xv, yv =   transformer.transform(x, y)

        #xv, yv = proj.transform(proj.Proj(crs_src),
        #                        proj.Proj(crs), x, y)


        n = z.size
        zi = np.reshape(z, (n))
        xi = np.reshape(xv, (n))
        yi = np.reshape(yv, (n))

        # Check and interpolate only elements in the frame
        if only_frame:
            is_in = _check_if_in(xi, yi)
            xi = xi[is_in]
            yi = yi[is_in]
            zi = zi[is_in]

        ny = len(ds.Y)
        nx = len(ds.X)
        nn = (ny, nx)

        if interpol == "IDW":
            X = np.array([xv, yv]).T
            coords = np.stack([  ds.coords['XV'].values.flatten() ,
                               ds.coords['YV'].values.flatten() ], axis=1)
            invdisttree = Invdisttree( X,
                                      z.reshape(-1,1), leafsize=leafsize, stat=1 )
            arr = invdisttree( coords,
                              nnear=Nnear, eps=eps, p=p )
            arr = arr.ravel().reshape(nn)


        else:
            arr = interpolate.griddata((xi, yi),
                                       zi,
                                       (ds.coords['XV'],
                                        ds.coords['YV']),
                                       method=interpol,
                                       **kwargs)

        if pad_around:
            for i in range(ep_max)[::-1]:
                arr[:, i][np.isnan(arr[:, i])] = arr[
                    :, i + 1][np.isnan(arr[:, i])]
                arr[:, -i][np.isnan(arr[:, -i])] = arr[:, -
                                                       i - 1][np.isnan(arr[:, -i])]
                arr[i, :][np.isnan(arr[i, :])] = arr[
                    i + 1, :][np.isnan(arr[i, :])]
                arr[-i, :][np.isnan(arr[-i, :])] = arr[-i -
                                                       1, :][np.isnan(arr[-i, :])]


        if use_dask:
            if dask_chunks is None:
                dask_chunks = (nx // chunk_n,) * arr.ndim
            arr = da.from_array(arr, chunks=dask_chunks)

        return arr


milli= 0.001
# print total max min mean meian of heat flow
def print_latex_tab_line(d = None, c= 'Comment', factor=milli):
    '''
    Copy and paste to tex-file.
    '''
    # remove factor effect
    d = d/factor
    print(f'Total : {len(d)} & {c} & Min : {d.min():0.1f} & Max : {d.max():0.1f} & mean : {d.mean():0.1f} & median : {d.median():0.1f}' )
    return







# takes series and col name bin 151 range 0 300 first ax0
def hq_hist(hf, col, ax=None, hf_range = (0, 300), bins = 151, label=None, factor=milli):
    if not label:
        label = col
    ax.set_xlim(hf_range)
        # remove factor effect
    return (hf[col]/factor).hist(bins=bins, ax=ax, range = hf_range, label=label)
