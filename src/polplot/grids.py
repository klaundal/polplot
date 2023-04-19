import numpy as np
d2r = np.pi / 180

def equal_area_grid(dr = 2, K = 0, M0 = 8, N = 20):
    """ Function for calculating an equal area grid in polar coordinates

    Parameters
    ----------
    dr : float (optional)
        The latitudinal resolution of the grid. Default 2 degrees
    K : int (optional)
        This number determines the colatitude of the inner radius of the
        post poleward ring of grid cells (the pole is not inlcluded!).
        It relates to this colatitude (r0) as r0/dr = (2K + 1)/2 => K = (2r0/dr - 1)/2.
        Default value is 0
    M0 : int (optional)
        The number of sectors in the most poleward ring. Default is 8
    N : int (optional)
        The number of rings to be included. This determiend how far
        equatorward the grid extends. Typically if dr is changed from 2 to 1,
        N should be doubled to reach the same latitude. Default is 20, which means
        that the equatorward edge of the grid is 89 - 20*2 = 49 degrees
        (the most poleward latitude is 89 with default values)
        
    Returns
    -------
    grid : array
        Edge coordinates of the mlat-mlt grid
        The first row contains mlat of the grid
        The second row contains mlt of the grid
    mltres : array
        width, in magnetic local time, of the cells with lower left corners described
        by mlat and mlt. Notice that this width changes with latitude, while the
        latitudinal width is fixed, determined by the `dr` parameter

    2022/05/05 (AO): Changed output to a combined grid with (mlat,mlt), similar to sdarngrid
    """

    r0 = dr * (2*K + 1)/2.

    assert M0 % (K + 1) == 0 # this must be fulfilled

    grid = {}

    M = M0
    grid[90 - r0 - dr] = np.linspace(0, 24 - 24./M, M) # these are the lower limits in MLT

    for i in range(N - 1):

        M = M *  (1 + 1./(K + i + 1.)) # this is the partion for i + 1

        grid[90 - (r0 + (i + 1)*dr) - dr] = np.linspace(0, 24 - 24./M, int(M)) # these are the lower limits in MLT
        # jreistad added int() to the M argument in np.linspace call in the above line.
        # I think line 43 forces this to always be the case. (2020-06-18)

    mlats = []
    mlts = []
    mltres = []
    for key in sorted(grid.keys()):
        mltres_ = sorted(grid[key])[1] - sorted(grid[key])[0]
        for mlt in sorted(grid[key]):
            mlats.append(key)
            mlts.append(mlt)
            mltres.append(mltres_)

    grid = np.vstack((np.array(mlats), np.array(mlts)))
    
    return grid, np.array(mltres)
    
    
def sdarngrid(dlat = 2, dlon = 2, latmin = 58):
    """ Generate same grid as used by Matthias Forster in EDI study.
        
        The grid is almost equal area but not exactly.
        Translated from MF's IDL code

    Parameters
    ----------
    dlat : float (optional)
        The latitudinal resolution of the grid. Default 2 degrees.
    dlon : float (optional)
        The longitudinal resolution of the grid. Default 2 degrees.
    latmin : float (optional)
        Minimum mlat of the grid. Default 58 degrees.
        
    Returns
    -------
    grid : 2 x N array
        Edge coordinates of the mlat-mlt grid
        The first row contains mlat of the grid
        The second row contains mlt of the grid
    mltres : array
        width, in magnetic local time, of the cells with lower left corners described
        by mlat and mlt. Notice that this width changes with latitude, while the
        latitudinal width is fixed, determined by the `dr` parameter

    2020.03.06 (SMH): Added return_mltres keyword to make output of this function's output parallel to that of equal_area_grid
    2022/05/05 (AO): Removed DEBUG and changed mlon to mlt
    """

    nlats = int((90. - latmin)/dlat)
    mlats = latmin + np.arange(nlats, dtype = np.float32)*dlat # lower latitude corner

    npixel = 0

    for i in range(nlats):
        circle = 360. * np.cos(mlats[i] * d2r + dlat/2*d2r)
        nlongs = int(np.round( circle / dlon ))
        dellon = 360. / np.float32(nlongs )
        npixel = npixel + nlongs
        mlat   = np.zeros(nlongs) + mlats[i]
        mlon   = np.arange(nlongs, dtype = np.float32) * dellon
        if i == 0:
            mlat_arr = np.array([mlat]).flatten()
            mlon_arr = np.array([mlon]).flatten()
        else:
            mlat_arr = np.hstack((mlat_arr, np.array(mlat)))
            mlon_arr = np.hstack((mlon_arr, np.array(mlon)))

    MLT_arr = mlon_arr/15. % 24. 
    grid = np.vstack((mlat_arr, MLT_arr)) # mlat/mlt grid pattern

    # calc mltres
    mltzeros = np.where(np.isclose(MLT_arr,0))[0]
    if mltzeros[-1] == (MLT_arr.size-1):
        mltzeros = mltzeros[:-1]
    stopmlt = np.append(mltzeros[1:],MLT_arr.shape[0])
    mltres = np.zeros(MLT_arr.shape)

    for istart in range(len(mltzeros)):
        mltres[mltzeros[istart]:stopmlt[istart]] = MLT_arr[mltzeros[istart]+1]-MLT_arr[mltzeros[istart]]

    return grid, mltres
    


def bin_number(grid, mlat, mlt):
    """
    Ultra-fast routine to determine the bin number of mlat and mlt
    
    Parameters
    ----------
    grid : 2 x N array
        Array containing the edges of the mlat-mlt bins.
        First row is mlat and second row mlt.
        grid can be constructed using either equal_area_grid or sdarngrid
    mlat : array
        Array with the mlats to bin
    mlt : array
        Array with the mlts to bin
        
    Returns
    -------
    bin_n : array
        Array with the bin number for each mlat-mlt pair.
        Locations outside the defined grid are set to -1 in the returned array.
    

    SMH 2021/04/15
    Modified by JPR 2021/10/20
    2022-05-02: JPR added nan handling
    2022-05-05: AO added better handling of values outside grid (including nans).
    """
    
    llat = np.unique(grid[0]) # latitude circles
    assert np.allclose(np.sort(llat) - llat, 0) # should be in sorted order automatically. If not, the algorithm will not work
    dlat = np.diff(llat)[0] # latitude step
    latbins = np.hstack(( llat, llat[-1] + dlat )) # make latitude bin edges
    
    bin_n = -np.ones_like(mlat).astype(int) # initiate binnumber
    ii = (np.isfinite(mlat))&(np.isfinite(mlt))&(mlat>=latbins[0])&(mlat<=latbins[-1]) # index of real values inside the grid
    mlat,mlt = mlat[ii],mlt[ii] # Reduce to only real values inside grid
    
    latbin_n = np.digitize(mlat, latbins) - 1 # find the latitude index for each data point

    # number of longitude bins in each latitude ring:
    nlons = np.array([len(np.unique(grid[1][grid[0] == lat])) for lat in llat])

    # normalize all longitude bins to the equatorward ring:
    _mlt = mlt * nlons[latbin_n] / nlons[0]

    # make longitude bin edges for the equatorward ring:
    llon = np.unique(grid[1][grid[0] == llat[0]])
    dlon = np.diff(llon)[0]
    lonbins = np.hstack((llon, llon[-1] + dlon)) # make longitude bin edges
    lonbin_n = np.digitize(_mlt, lonbins) - 1 # find the longitude bin

    # map from 2D bin numbers to 1D by adding the number of bins in each row equatorward:
    bin_n[ii] = lonbin_n + np.cumsum(np.hstack((0, nlons)))[latbin_n]

    return bin_n

def cell_area(mlat, r, mlatres):
    '''
    Calculates the cell area of grid cells on a sphere

    Parameters
    ----------
    mlat    : Array of floats
        Magentic latitude of observations.
    r       : Integer or float
        Radius of sphere
    mlatres : Integer or float
        Latitudinal resolution of cells

    Returns
    -------
    cell_area : Array of floats
        Area of the cells in unit of r

    '''

    if not (isinstance(r, int) or isinstance(r, float)):
        raise TypeError('r has to be interger or float')

    if not (isinstance(mlatres, int) or isinstance(mlatres, float)):
        raise TypeError('mlatres has to be interger or float')

    if not (isinstance(mlat, list) or isinstance(mlat, np.ndarray)):
        raise TypeError('mlat has to be an 1D list of array')

    if isinstance(mlat, list):
        mlat = np.array(mlat)

    # Number of cells per latitude band
    p_per_lat = np.zeros(len(mlat))
    unique_lats, counts = np.unique(mlat, return_counts=True)
    for k, (lat_u, count) in enumerate(zip(unique_lats, counts)):
        p_per_lat[mlat == lat_u] = count

    # Height of spherical segment
    h = r*(np.sin((mlat+mlatres/2)/180*np.pi) - np.sin((mlat-mlatres/2)/180*np.pi))

    # Area per cell (surface area of spherical segment divided by number of cells)
    cell_area = 2*np.pi*r*h/p_per_lat

    return cell_area
