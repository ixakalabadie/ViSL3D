#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:57:23 2024

@author: ixakalabadie
"""
from skimage import measure
from astropy.coordinates import SkyCoord
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
    shift : tuple, optional
        Shift in RA, DEC and V in same units as delta and mins. The default is (0,0,0).
    step_size : int, optional
        Step size for the marching_cubes algorithm. Sets the resolution. High step sizes produce
        low resolution models. Default is 1. 
    
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
    Obtain a dictionary with galaxy names, coordinates and colors to introduce in a Cube object to
    use in writers.make_galaxies().

    Parameters
    ----------
    galaxies : list or string
        List with the names of galaxies to include in the model.
        If 'query' a NED query is made within the limits of the cube.
        If None no galaxies are included.
    cubecoords : array-like
        3x2 array with the coordinates of the cube in the format
        [[ramin, ramax], [decmin, decmax], [zmin, zmax]] and in the same units as cubeunits.
    cubeunits : array-like
        len 4 array with the units of the cube as strings.
    obj : string
        Name of the object to query in NED.
    delta : array-like
        len 3 array with the delta values of the cube.
    trans : array-like
        len 3 array with the scale of each coordinate axis. It is calculated like
        [2000/nx, 2000/ny, 2000/nz].
    
    Returns
    -------
    galdict : dict
        Dictionary with the names of the galaxies as keys and two dictionaries with the coordinates
        and color of the galaxy as values.
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
            if galra > 180 * u.deg:
                galra = galra - 360*u.deg
            galdec = float(result['DEC'])*result['DEC'].unit
            galv = float(result['Velocity'])*result['Velocity'].unit
            galra = (galra - np.mean(cubecoords[0])*u.Unit(cubeunits[1])) \
                * np.cos(cubecoords[1][0]*u.Unit(cubeunits[2]).to('rad'))
            galdec = galdec - np.mean(cubecoords[1])*u.Unit(cubeunits[2])
            galv = galv - np.mean(cubecoords[2])*u.Unit(cubeunits[3])
            galra = galra.to(cubeunits[1])
            galdec = galdec.to(cubeunits[2])
            galv = galv.to(cubeunits[3])
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

def calc_isolevels(cube, unit=None):
    """
    Function to calculate isolevels if not given by the user.

    Parameters
    ----------
    cube : 3D array
        Datacube.
    """
    if unit == 'percent':
        isolevels = [10, 30, 50, 70, 90]
    else:
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
    TO DO

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

def get_imcol(image=None, position=None, survey=None, cmap='Greys', **kwargs):
    """
    Downloads an image from astroquery and returns the colors of the pixels using
    a certain colormap, in hexadecimal format, as required by 'write_x3d().make_image2d'.
    See astroquery.skyview.SkyView.get_images() for more information.

    Having a large field of view (verts) might disalign the image with the cube.
    This issue will be fixed in the future.

    Parameters
    ----------
    image : 2D or 3D array, optional
        Image data in RGB format between 0 and 1 (3D). The RGB column must be last.
        If 2D, the image will be converted automatically.
        The image will cover the full FoV of the created cube model (after applying limits).
        Example for 3D array:
            image = np.array([img1, img2, img3])
            image = np.transpose(img, axes=(1,2,0)) # shape=(3,ny,nx)->(ny,nx,3)
    position : string or SkyCoord, optional
        Name of an object or it position coordinates.
    survey : string, optional
        Survey from which to make the query. See astroquery.skyview.SkyView.list_surveys().
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
    if image is None:
        image = SkyView.get_images(position=position, survey=survey, **kwargs)[0]
        image = image[0].data
        image = image-np.min(image)
        image = (image)/np.max(image)
        colimg = cm.get_cmap(cmap)(image)[:,:,0:3] # convert to RBG remove alpha channel
        colimg = colimg.reshape((-1,3),order='F') # flatten the array
    elif image.ndim == 2:
        image = image-np.min(image)
        image = (image)/np.max(image)
        colimg = cm.get_cmap(cmap)(image)[:,:,0:3] # convert to RBG remove alpha channel
        colimg = colimg.reshape((-1,3),order='F')
    elif image.ndim == 3:
        colimg = image.reshape((-1,3),order='F')

    imcol = [mcolors.rgb2hex(c).replace('#','0x') for c in colimg]
    if len(imcol)% 8 == 0:
        imcol = np.array(imcol).reshape(int(len(imcol)/8),8)

    return imcol, image.shape[:2], image

def transpose(array, delta):
    """
    Transpose data array taking the direction of delta into account.
    """
    return np.transpose(array, (2,1,0))[::int(np.sign(delta[0])),
                                        ::int(np.sign(delta[1])),::int(np.sign(delta[2]))]
def calc_camera_position(vector):
    axis = np.array(vector[:3])
    angle = vector[3]

    # Calculate sine and cosine of the angle
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    # Normalize axis
    length = np.linalg.norm(axis)
    if length != 0:
        axis = axis/length

    # Calculate rotation matrix elements
    m00 = c + axis[0] * axis[0] * t
    m11 = c + axis[1] * axis[1] * t
    m22 = c + axis[2] * axis[2] * t
    tmp1 = axis[0] * axis[1] * t
    tmp2 = axis[2] * s
    m01 = tmp1 + tmp2
    m10 = tmp1 - tmp2
    tmp1 = axis[0] * axis[2] * t
    tmp2 = axis[1] * s
    m02 = tmp1 - tmp2
    m20 = tmp1 + tmp2
    tmp1 = axis[1] * axis[2] * t
    tmp2 = axis[0] * s
    m12 = tmp1 + tmp2
    m21 = tmp1 - tmp2

    # Apply rotation matrix to default camera position
    camera = -np.dot(np.array([0, 0, -1]), np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]]))

    return np.round(camera,6)

def calc_axis_angle(point):
    # Calculate the vector from the origin to the given point
    vector_to_point = np.array(point)
    
    # Normalize the vector to obtain the axis of rotation
    axis = vector_to_point / np.linalg.norm(vector_to_point)
    
    # Determine the angle of rotation
    # Here, we'll choose the angle to rotate the axis of rotation to align with the z-axis
    angle = np.arctan2(axis[1], axis[0])  # Angle between the x-axis and the vector
    
    return np.array([axis[0],axis[1],axis[2], angle])

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
tablehtml = '\n<!--A table with navigation info for X3DOM-->\n<br/>\n<hr>\n<h3><b>Navigation:</b></h3>\n<table style="border-collapse: collapse; border: 2px solid rgb(0,0,0);">\n<tbody><tr style="background-color: rgb(220,220,220); border: 1px solid rgb(0,0,0);">\n<th width="250px">Function</th>\n<th>Mouse Button</th>\n</tr>\n</tbody><tbody>\n<tr style="background-color: rgb(240,240,240);"><td>Rotate</td>\n<td>Left / Left + Shift</td>\n</tr>\n<tr><td>Pan</td>\n<td>Mid / Left + Ctrl</td>\n</tr>\n<tr style="background-color: rgb(240,240,240);"><td>Zoom</td>\n<td>Right / Wheel / Left + Alt</td>\n</tr>\n<tr><td>Set center of rotation</td>\n<td>Double-click left</td>\n</tr>\n</tbody>\n</table>\n<p>Zooming with the mouse wheel is not as smooth as with the right mouse button or with Alt + Left.<p>'

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
