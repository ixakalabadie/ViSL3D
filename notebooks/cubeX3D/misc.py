#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:57:23 2024

@author: ixakalabadie
"""
from skimage import measure
from astropy.coordinates import SkyCoord
from astropy import wcs
from astroquery.ipac.ned import Ned
from astroquery.skyview import SkyView
import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt gives error in dachs
from matplotlib import cm

from . import np
from . import u

#Some miscellaneoues functions

def marching_cubes(cube, level, shift=(0,0,0), step_size=1):
    """
    Implementation of marching cubes algorithm to a datacube.

    Parameters
    ----------
    cube : 3D array
        Datacube.
    level : float
        Value of the isosurface.
    delta : tuple
        Spacing in each dimension.
    mins : tuple
        Minimum values of each dimension.
    shift : tuple, optional
        Shift in RA, DEC and V in same units as delta and mins. The default is (0,0,0).
    step_size : int, optional
        Step size for the marching_cubes algorithm. Set the resolution. Default is 1. 
    
    Returns
    --------
    Tuple with (1) Array with the coordinates of the vertices of the created triangular faces
    and (2) the indices for those faces and (3) normal vectors for each face.
    """
    nx, ny, nz = cube.shape
    trans = (2000/nx, 2000/ny, 2000/nz)
    verts, faces, normals, _ = measure.marching_cubes(cube, level = level,
                            allow_degenerate=False,
                            step_size=step_size)
    return (np.array([verts[:,0]*trans[0]-1000+shift[0], verts[:,1]*trans[1]-1000+shift[1],
                     verts[:,2]*trans[2]-1000+shift[2]]).T, faces,
                     np.array([normals[:,0], normals[:,1], normals[:,2]]).T)

def get_galaxies(galaxies, cubecoords, cubeunits, obj, delta, trans):
    """
    galdict['coord'] is in the same units as the cube
    """
    if galaxies == ['query']:
        sc = SkyCoord(cubecoords[0][0]*u.Unit(cubeunits[1]),
                        cubecoords[1][0]*u.Unit(cubeunits[2]))
        sepa = SkyCoord(
            np.mean(cubecoords[0])*u.Unit(cubeunits[1]),
            np.mean(cubecoords[1])*u.Unit(cubeunits[2])).separation(sc)
        result = Ned.query_region(
            obj, radius=sepa)['Object Name', 'Type', 'RA', 'DEC',
                                    'Velocity']
        if result['RA'].unit == 'degrees':
            result['RA'].unit = u.deg
        if result['DEC'].unit == 'degrees':
            result['DEC'].unit = u.deg
        result = objquery(result, [
            cubecoords[0]*u.Unit(cubeunits[1]),
            cubecoords[1]*u.Unit(cubeunits[2]),
            cubecoords[2]*u.Unit(cubeunits[3])], otype='G')
        galdict = {}
        for gal in result:
            galra = float(gal['RA'])*result['RA'].unit
            galdec = float(gal['DEC'])*result['DEC'].unit
            galv = float(gal['Velocity'])*result['Velocity'].unit
            galra = (galra - np.mean(cubecoords[0])*u.Unit(cubeunits[1])) \
                * np.cos(cubecoords[1][0]*u.Unit(cubeunits[2]).to('rad'))
            galdec = galdec - np.mean(cubecoords[1])*u.Unit(cubeunits[2])
            galv = galv - np.mean(cubecoords[2])*u.Unit(cubeunits[3])
            galra = galra/np.abs(delta[0])*trans[0]
            galdec = galdec/np.abs(delta[1])*trans[1]
            galv = galv/np.abs(delta[2])*trans[2]
            galdict[gal['Object Name']] = {
                    'coord': np.array([galra.to_value(), galdec.to_value(), galv.to_value()]),
                    'col': '0 0 1'}
    elif galaxies is not None:
        galdict = {}
        for gal in galaxies:
            result = Ned.query_object(gal)
            if result['RA'].unit == 'degrees':
                result['RA'].unit = u.deg
            if result['DEC'].unit == 'degrees':
                result['DEC'].unit = u.deg
            galra = float(result['RA'])*result['RA'].unit
            galdec = float(result['DEC'])*result['DEC'].unit
            galv = float(result['Velocity'])*result['Velocity'].unit
            galra = (galra - np.mean(cubecoords[0])*u.Unit(cubeunits[1])) \
                * np.cos(cubecoords[1][0]*u.Unit(cubeunits[2]).to('rad'))
            galdec = galdec - np.mean(cubecoords[1])*u.Unit(cubeunits[2])
            galv = galv - np.mean(cubecoords[2])*u.Unit(cubeunits[3])
            galra = galra/np.abs(delta[0])*trans[0]
            galdec = galdec/np.abs(delta[1])*trans[1]
            galv = galv/np.abs(delta[2])*trans[2]
            galdict[gal] = {
                    'coord': np.array([galra.to_value(), galdec.to_value(), galv.to_value()]),
                    'col': '0 0 1'}
    return galdict

def create_colormap(colormap, isolevels, start=0, end=255, lightdark=False):
    """
    Function to create a colormap for the iso-surfaces.

    Parameters
    ----------
    colormap : string
        Name of a matplotlib colormap.
    isolevels : list
        List of values of the iso-surfaces.
    start : int, optional
        Starting element of the colormap array. Default is 0.
    end : int, optional
        Ending element of the colormap array. Default is 255.
    lightdark : bool, optional
        Wheter to reverse the colormap if the darkest side is at the beggining
    
    Returns
    -------
    cmap : list
        List of strings with the colors of the colormap in the format 'r g b'.
    """
    colors = cm.get_cmap(colormap)(range(256))[:,:-1]
    if lightdark:
        if np.sum(colors[0]) < np.sum(colors[-1]):
            colors = colors[::-1]
    cmap = []
    for lev in isolevels:
        m = (end-start)/(np.max(isolevels)-np.min(isolevels))
        pos = int((m*lev-m*np.min(isolevels))+start)
        cmap.append(f'{colors[pos][0]:.5e} {colors[pos][1]:.5e} {colors[pos][2]:.5e}')
    return cmap

def tabs(n):
    """
    Create a string with n tabs.
    """
    return '\t'*n

def insert_3darray(big, small):
    """Insert values of smaller 3D array into the middle of the zero array."""
    b_shape = big.shape
    s_shape = small.shape
    start_x = (b_shape[0] - s_shape[0]) // 2
    start_y = (b_shape[1] - s_shape[1]) // 2
    start_z = (b_shape[2] - s_shape[2]) // 2

    big[start_x:start_x+s_shape[0], start_y:start_y+s_shape[1], start_z:start_z+s_shape[2]] = small

    return big

def calc_isolevels(cube):
    """
    Function to calculate isolevels if not given by the user.

    Parameters
    ----------
    cube : 3D array
        Datacube.
    """
    if np.min(cube) < 0:
        isolevels = [np.max(cube)/10., np.max(cube)/5., np.max(cube)/3., np.max(cube)/1.5]
    elif np.min(cube) < np.max(cube)/5.:
        isolevels = [np.min(cube), np.max(cube)/5., np.max(cube)/3., np.max(cube)/1.5]
    return np.array(isolevels)

def objquery(result, coords, otype):
    """
    Constrain query table to certain coordinates and object type
    """
    result = result[result['Type'] == otype]
    result = result[result['Velocity'] >= coords[2][0]]
    result = result[result['Velocity'] <= coords[2][1]]
    result = result[result['RA'] >= coords[0][0]]
    result = result[result['RA'] <= coords[0][1]]
    result = result[result['DEC'] >= coords[1][0]]
    result = result[result['DEC'] <= coords[1][1]]
    return result

def calc_step(cube, isolevels):
    """
    To automatically calculate best step size (marching cubes algorithm) to obtain light models.
    """
    npix = np.sum(cube > np.min(isolevels))
    if npix > 5e6:
        step = 1
    else:
        step = npix*2.5e-6
    return step


def preview2d(cube, v1=None, v2=None, norm='asinh', figsize=(10,8)):
    """
    

    Parameters
    ----------
    cube : 3d array
        The data cube. Must be unitless.
    v1 : array, optional
        Minimum  and maximum values for the colormap.
        If None the minimum and maximum of image 1 are taken.
        Default is None.
    v2 : float, optional
        Minimum and maximum values for the colormap.
        If None the minimum and maximum of image 2 are taken.
        Default is None.
    norm : string
        A scale name, one of 'asinh', 'function', 'functionlog', 'linear', 'log', 'logit' or
        'symlog'. Default is 'asinh'.
        For more information see `~matplotlib.colors.Normalize`.
    figsize : tuple, optional
        Figure size. Default is (10,8).

    Returns
    -------
    None.

    """
    # nz, ny, nx = cube.shape
    # cs1 = np.sum(cube, axis=0)
    # cs2 = np.sum(cube, axis=2)
    # vmin1, vmax1 = v1
    # vmin2, vmax2 = v2
    # if vmin1 is None:
    #     vmin1 = np.min(cs1)
    # if vmax1 is None:
    #     vmax1 = np.max(cs1)
    # if vmin2 is None:
    #     vmin2 = np.min(cs2)
    # if vmax2 is None:
    #     vmax2 = np.max(cs2)

    # _, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    # ax[0,0].hist(cs1.flatten(), density=True)
    # #imshow plots axes fist -> y , second -> x
    # ax[0, 1].imshow(cs1, vmin=vmin1, vmax=vmax1, norm=norm, origin='lower')
    # ax[0, 1].set_ylabel('DEC')
    # ax[0, 1].set_xlabel('RA')

    # ax[0, 1].set_yticks(np.arange(0, ny+1, 50), labels=np.arange(0, ny+1, 50), minor=False)
    # ax[0, 1].set_xticks(np.arange(0, nx+1, 50), labels=np.arange(0, nx+1, 50), minor=False)
    # ax[0, 1].grid(which='major')

    # ax[1, 0].hist(cs2.flatten(), density=True)
    # #imshow plots axes fist -> y , second -> x
    # ax[1, 1].imshow(cs2.transpose(), vmin=vmin2, vmax=vmax2, norm=norm, origin='lower')
    # ax[1, 1].set_ylabel('DEC')
    # ax[1, 1].set_xlabel('V')

    # ax[1, 1].set_yticks(np.arange(0, ny+1, 50), labels=np.arange(0, ny+1, 50), minor=False)
    # ax[1, 1].set_xticks(np.arange(0, nz+1, 50), labels=np.arange(0, nz+1, 50), minor=False)
    # ax[1, 1].grid(which='major')
    pass

def get_imcol(position, survey, verts, unit='deg', cmap='Greys', **kwargs):
    """
    Downloads an image from astroquery and returns the colors of the pixels using
    a certain colormap, in hexadecimal format, as required by 'write_x3d().make_image2d'.
    See astroquery.skyview.SkyView.get_images() for more information.

    Having a large field of view (verts) might disalign the image with the cube.
    This issue will be fixed in the future.

    Parameters
    ----------
    position : string or SkyCoord
        Name of an object or it position coordinates.
    survey : string
        Survey from which to make the query. See astroquery.skyview.SkyView.list_surveys().
    verts : array
        Minimum RA, maximum RA, minimum DEC and maximum DEC of the data cube, in that order.
    **kwargs : 
        Other parameters for astroquery.skyview.SkyView.get_images(). Useful parameters
        are 'unit', 'pixels' and 'coordinates'.

    Returns
    -------
    imcol : array
        Array with the colors of each pixel in hexadecimal format.
    shape : tuple
        Shape of the image.
    img : array
        Image data.

    """

    img = SkyView.get_images(position=position, survey=survey, **kwargs)[0]
    imw = wcs.WCS(img[0].header)
    img = img[0].data

    try:
        verts = verts.to('deg')
    except AttributeError:
        verts = verts * u.Unit(unit)

    ll_ra,ll_dec = imw.world_to_pixel(SkyCoord(verts[0],verts[2]))
    lr_ra,_ = imw.world_to_pixel(SkyCoord(verts[1],verts[2]))
    _,ul_dec = imw.world_to_pixel(SkyCoord(verts[0],verts[3]))
    if ll_ra < 0 or ll_dec < 0 or lr_ra < 0 or ul_dec < 0:
        print('ERROR: The image is smaller than the cube. Increase parameter "pixels"')
        print("Pixel indices for [ra1, dec1, ra2, dec2] = " \
              +str([ll_ra, ll_dec, lr_ra, ul_dec])+ \
              ". Set 'pixels' parameter higher than the difference between 1 and 2.")
        raise ValueError

    img = img[int(ll_dec):int(ul_dec), int(lr_ra):int(ll_ra)] #dec first, ra second!!
    img = img-np.min(img)
    img = (img)/np.max(img)

    colimg = cm.get_cmap(cmap)(img)[:,:,0:3]
    colimg = colimg.reshape((-1,3),order='F')

    imcol = [mcolors.rgb2hex(c).replace('#','0x') for c in colimg]
    if len(imcol)% 8 == 0:
        imcol = np.array(imcol).reshape(int(len(imcol)/8),8)

    return imcol, img.shape, img

def transpose(array, delta):
    """
    Transpose data array taking the direction of delta into account.
    """
    return np.transpose(array, (2,1,0))[::int(np.sign(delta[0])),
                                        ::int(np.sign(delta[1])),::int(np.sign(delta[2]))]

# Some attributes for the classes and functions

roundto = "\t<script>\n\t\t //Round a float value to x.xx format\n" \
    +tabs(2)+"function roundTo(value, decimals)\n\t\t{\n" \
    +tabs(3)+"return (Math.round(value * 10**decimals)) / 10**decimals;\n\t\t }\n\t</script>\n"

labpos = (np.array([[0,-1000*1.1,-1000],
                   [1000*1.1, 0,-1000],
                   [-1000,-1000*1.1,0],
                   [-1000,0,-1000*1.1],
                   [-1000*1.1, 1000, 0],
                   [0, 1000, -1000*1.1]]),
        np.array([[1000, -1000*1.1, -1000],
                  [-1000, -1000*1.1, -1000],
                  [1000*1.1, 1000, -1000],
                  [1000*1.1, -1000, -1000],
                  [-1000, -1000*1.1, -1000],
                  [-1000, -1000*1.1, 1000],
                  [-1000, 1000, -1000*1.1],
                  [-1000, -1000, -1000*1.1],
                  [-1000*1.1, 1000, -1000],
                  [-1000*1.1, 1000, 1000],
                  [1000, 1000, -1000*1.1],
                  [-1000, 1000, -1000*1.1]]))

ticklineindex = np.array([[0, 1, -1],
                          [2, 3, -1],
                          [4, 5, -1],
                          [6, 7, -1],
                          [8, 9, -1],
                          [10, 11, -1]])
outlineindex = np.array([[0, 1, -1],
                         [2, 3, -1],
                         [4, 5, -1],
                         [6, 7, -1],
                         [0, 2, -1],
                         [1, 3, -1],
                         [4, 6, -1],
                         [5, 7, -1],
                         [0, 4, -1],
                         [1, 5, -1],
                         [2, 6, -1],
                         [3, 7, -1]])

# html code for the navigation table
tablehtml = '\n<!--A table with navigation info for X3DOM-->\n<br/>\n<hr>\n<h3><b>Navigation:</b></h3>\n<table style="border-collapse: collapse; border: 2px solid rgb(0,0,0);">\n<tbody><tr style="background-color: rgb(220,220,220); border: 1px solid rgb(0,0,0);">\n<th width="250px">Function</th>\n<th>Mouse Button</th>\n</tr>\n</tbody><tbody>\n<tr style="background-color: rgb(240,240,240);"><td>Rotate</td>\n<td>Left / Left + Shift</td>\n</tr>\n<tr><td>Pan</td>\n<td>Mid / Left + Ctrl</td>\n</tr>\n<tr style="background-color: rgb(240,240,240);"><td>Zoom</td>\n<td>Right / Wheel / Left + Alt</td>\n</tr>\n<tr><td>Set center of rotation</td>\n<td>Double-click left</td>\n</tr>\n</tbody>\n</table>'

#name of ax labels for difference from center
axlabname1 = np.array(['R.A. [arcsec]', 'Dec. [arcsec]', 'V [km/s]',
                'Dec. [arcsec]', 'V [km/s]', 'R.A. [arcsec]'])

def get_axlabnames(mags):
    """
    Parameters:
    ----------
    mags : array
        Array with the names of the magnitudes. Must be of length 3.
    units : array
        Array with the names of the units. Must be length 4, the units of the corresponding
        magnitudes being the last 3 elements.
    """
    return np.array([mags[1].split('-')[0], # +' ('+units[1]+')'
                     mags[2].split('-')[0], # +' ('+units[2]+')'
                     mags[3].split('-')[0], # +' ('+units[3]+')'
                     mags[2].split('-')[0], # +' ('+units[2]+')'
                     mags[3].split('-')[0], # +' ('+units[3]+')'
                     mags[1].split('-')[0]]) # +' ('+units[1]+')'

#name of ax labels
axlabname2 = np.array(['R.A.', 'Dec.', 'V [km/s]',
                'Dec.', 'V [km/s]', 'R.A.'])

# justification of axes labels
axlabeljustify = np.array(['"MIDDLE" "END"', '"MIDDLE" "BEGIN"',
                      '"MIDDLE" "END"', '"MIDDLE" "BEGIN"',
                      '"MIDDLE" "END"', '"MIDDLE" "BEGIN"'])
# justification of axes tick labels
axticklabjus = np.array(['"MIDDLE" "END"', '"MIDDLE" "END"',
                     '"END" "END"','"END" "BEGIN"',
                     '"MIDDLE" "END"', '"MIDDLE" "END"',
                     '"END" "END"', '"END" "BEGIN"',
                     '"MIDDLE" "END"', '"MIDDLE" "END"',
                     '"END" "END"','"END" "BEGIN"'])
# rotation of ax labels
axlabrot = np.array(['0 1 0 3.14','1 1 0 3.14','0 1 0 -1.57',
                 '1 1 -1 -2.0944','1 1 1 -2.0944','1 0 0 -1.57'])

# side and corresponding name of html buttons
side,nam = np.array([['front',"R.A. - Dec."],['side',"Z - Dec."],
                     ['side2',"Z - R.A."],['perspective',"Perspective View"]]).T

default_cmaps = ['CMRmap_r', 'magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight',
                 'twilight_shifted', 'turbo', 'Blues', 'BrBG', 'BuGn', 'BuPu','CMRmap', 'GnBu',
                 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr',
                 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Spectral',
                 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary',
                 'bone', 'brg', 'bwr','cool', 'coolwarm', 'copper', 'cubehelix', 'flag',
                 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern',
                 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'jet', 'nipy_spectral',
                 'ocean', 'pink', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'terrain',
                 'winter', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2',
                 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'magma_r', 'inferno_r', 'plasma_r',
                 'viridis_r', 'cividis_r', 'twilight_r', 'twilight_shifted_r', 'turbo_r', 'Blues_r',
                   'BrBG_r', 'BuGn_r', 'BuPu_r', 'GnBu_r', 'Greens_r', 'Greys_r',
                   'OrRd_r', 'Oranges_r', 'PRGn_r', 'PiYG_r', 'PuBu_r', 'PuBuGn_r', 'PuOr_r',
                   'PuRd_r', 'Purples_r', 'RdBu_r', 'RdGy_r', 'RdPu_r', 'RdYlBu_r', 'RdYlGn_r',
                   'Reds_r', 'Spectral_r', 'Wistia_r', 'YlGn_r', 'YlGnBu_r', 'YlOrBr_r',
                   'YlOrRd_r', 'afmhot_r', 'autumn_r', 'binary_r', 'bone_r', 'brg_r', 'bwr_r',
                   'cool_r', 'coolwarm_r', 'copper_r', 'cubehelix_r', 'flag_r', 'gist_earth_r',
                   'gist_gray_r', 'gist_heat_r', 'gist_ncar_r', 'gist_rainbow_r', 'gist_stern_r',
                   'gist_yarg_r', 'gnuplot_r', 'gnuplot2_r', 'gray_r', 'hot_r', 'hsv_r', 'jet_r',
                   'nipy_spectral_r', 'ocean_r', 'pink_r', 'prism_r', 'rainbow_r', 'seismic_r',
                   'spring_r', 'summer_r', 'terrain_r', 'winter_r', 'Accent_r', 'Dark2_r',
                   'Paired_r', 'Pastel1_r', 'Pastel2_r', 'Set1_r', 'Set2_r', 'Set3_r', 'tab10_r',
                   'tab20_r', 'tab20b_r', 'tab20c_r']

astropy_prefixes = ['','k','m','M','u','G','n','d','c','da','h','p','T','f','a','P','E',
                    'z','y','Z','Y','r','q','R','Q']
angular_units = ['arcsec', 'arcmin', 'deg', 'rad']
