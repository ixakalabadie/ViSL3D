#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:05:52 2024

@author: ixakalabadie
"""
from astropy.io import fits
from scipy.stats import norm

from . import misc, writers
from . import np
from . import u

class Cube:
    """
    Class to store relevant data to create the visualisation.

    Parameters
    ----------
    l_cubes : list
        List of 3D arrays with the data. Even if it is only one cube it must be a list.
    name : str
        Name of the object.
    coords : array
        Array with the coordinates of the cube. It must be a sorted 3x2 array with the minimum and maximum
        values of the spatial and spectral axes.
    l_isolevels : list
        List of arrays with the isosurface levels for each cube.
    l_colors : list
        RGB color for each isolevel as a string ('122 233 20').
        Must have the same shape as l_isolevels.
    units : len 4 array
        Array of strings with the units of the data cube. The first element is the unit of the data.
    mags : len 4 array
        Array of strings with the magnitudes of the data cube. E.g. ['I', 'RA', 'DEC', 'V'].
    rms : float, optional
        RMS of the data cube.
    resol : float, optional
        Step size for the marching cubes algorithm. Higher value means lower resolution. Default is 1.
    iso_split : list, optional
        List of floats with the values to split the isosurfaces. It is not expected to be entered by the user,
        it will be calculated automatically.
    galaxies : dict, optional
        Dictionary with galaxies to include in the model. Keys are names of galaxies and it must
        have two more dictionaries inside with the keys 'coords' and 'col'. 'coords' must have the
        coordinates of the galaxy (transformed to units of the cube between -1000 and 1000) and 'col' the color of each galaxy as a string. E.g. {'NGC 1234': {'coords': [-560, 790, 134], 'col': '255 0 0'}}.
        'coord' is calculated with '[RA - mean(cubeRA)]/abs(deltaRA)*2000/nPixRA'. The whole dictionary
        can be obtained with misc.get_galaxies().
    image2d : tuple, optional
        Tuple with the image in hexadecimal format and the shape of the image.
        It can be calculated with the function misc.get_imcol.
        If (None,None) a blank plane will be used.
    lines : dict, optional
        Dictionary with the names of spectral lines as keys and a tuple with the centre and
        the full width of the line as value. The centre must be an astropy quantity and the width
        can be a quantity or an integer representing the number of pixels.
        E.g. {'NII': (6583*u.Angstrom, 50 [pixels])), 'Halpha': (6562*u.Angstrom, 10*u.Angstrom)}.
    interface : str, optional
        Format of the web interface. 'full' gives a fully interactive web page and 'minimal' just
        the 3D model. Default is 'full'.
    """
    def __init__(self, l_cubes, name, coords, l_isolevels, l_colors, units, mags, rms=None,
                 resol=1, iso_split=[], galaxies=None, image2d=None, lines=None, interface='full'):
        self.l_cubes = l_cubes
        self.name = name
        self.coords = coords
        self.units = units
        self.mags = mags
        self.l_colors = l_colors
        self.rms = rms
        self.l_isolevels = l_isolevels
        self.resol = resol
        self.iso_split = iso_split
        self.galaxies = galaxies
        self.image2d = image2d
        self.lines = lines
        self.interface = interface

    def __str__(self):
        s = f'Cube object for {self.name}' #\n\tCoordinates: {self.coords}\n\tUnits: {self.units}\n\tMagnitudes: {self.mags}\n\tRMS: {self.rms}\n\tResolution: {self.resol}\n\tInterface: {self.interface}'
        return s

def prep_one(cube, header=None, lims=None, unit=None, isolevels=None, colormap='CMRmap_r',
             image2d=None, galaxies=None):
    """
    Prepare the Cube class for a cube with a single spectral line.

    Parameters
    ----------
    cube : str or np.ndarray
        Path to the FITS file or to the data cube.
    header : astropy.io.fits.header.Header, optional
        Header of the FITS file.
    lims : array-like, optional
        3x2 array with the minimum and maximum limits of the cube. If None the whole cube is used.
        It can contain limits in astropy quantities or in pixels.
        E.g. [[ramin, ramax], [decmin, decmax], [vmin, vmax]]. Default is None.
    unit : str, optional
        Unit to convert the data cube. If None the original unit is used.
        If 'rms' data is divided by the RMS of the cube.
        If 'percent' data is divided by the maximum value.
    isolevels : array-like, optional
        Array with the isosurface levels in the given units. If None it is calculated automatically.
    colormap : str, optional
        Name of the colormap to use. Default is 'CMRmap_r'.
    image2d : str, optional
        Name of the survey to query a 2D image. See survey options in https://astroquery.readthedocs.io/en/latest/skyview/skyview.html.
        If 'blank' a blank plane is used. If None no image is created.
    galaxies : list or str, optional
        List with the names of galaxies to include in the model. If 'query' a NED query is made within the limits of the cube. If None no galaxies are included.
    
    Returns
    -------
    Cube
        An object of the Cube class for a single spectral line model.
    """
    if isinstance(cube, str):
        with fits.open(cube) as hdul:
            cube = hdul[0].data # this is pol, v, dec, ra
            header = hdul[0].header
    elif header is None:
        raise AttributeError('No header provided')

    if len(cube.shape) < 3:
        raise ValueError('Not enough axes')
    if len(cube.shape) >= 4:
        print('Warning: Cube with polarization axis. Using first one.')
        cube = cube[0]
    cube = np.squeeze(cube)

    delta = np.array([header['CDELT1'], header['CDELT2'], header['CDELT3']])
    cubeunits = np.array([header['BUNIT'],header['CUNIT1'], header['CUNIT2'], header['CUNIT3']])
    if cubeunits[0] == 'JY/BEAM':
        cubeunits[0] = 'Jy/beam'
    cubemags = np.array([header['BTYPE'],header['CTYPE1'], header['CTYPE2'], header['CTYPE3']])

    cube = misc.transpose(cube, delta)
    nx, ny, nz = cube.shape

    if lims is None:
        lims = np.array([[0,nx], [0,ny], [0,nz]], dtype=int)

    if isinstance(lims, list) and isinstance(lims[0][0], u.quantity.Quantity):
        lims[0].to(u.Unit(cubeunits[1]))
        lims[1].to(u.Unit(cubeunits[2]))
        lims[2].to(u.Unit(cubeunits[3]))
        cubecoords = np.array(lims)
        lims = np.array([
            [np.round((cubecoords[0][0]-header['CRVAL1'])/delta[0] + header['CRPIX1']),
            np.round((cubecoords[0][1]-header['CRVAL1'])/delta[0] + header['CRPIX1'])],
            [np.round((cubecoords[1][0]-header['CRVAL2'])/delta[1] + header['CRPIX2']),
            np.round((cubecoords[1][1]-header['CRVAL2'])/delta[1] + header['CRPIX2'])],
            [np.round((cubecoords[2][0]-header['CRVAL3'])/delta[2] + header['CRPIX3']),
            np.round((cubecoords[2][1]-header['CRVAL3'])/delta[2] + header['CRPIX3'])]
            ])
    else:
        cubecoords = np.array([
            [header['CRVAL1']+delta[0]*(lims[0][0]-header['CRPIX1']),
            header['CRVAL1']+delta[0]*(lims[0][1]-header['CRPIX1'])],
            [header['CRVAL2']+delta[1]*(lims[1][0]-header['CRPIX2']),
            header['CRVAL2']+delta[1]*(lims[1][1]-header['CRPIX2'])],
            [header['CRVAL3']+delta[2]*(lims[2][0]-header['CRPIX3']),
            header['CRVAL3']+delta[2]*(lims[2][1]-header['CRPIX3'])],
            ])

    for i in range(3):
        if lims[i][0] < 0:
            raise ValueError('lims out of range')
        if lims[i][1] > cube.shape[i]:
            raise ValueError('lims out of range')

    cubecoords = np.sort(cubecoords)
    cube = cube[lims[0][0]:lims[0][1],lims[1][0]:lims[1][1],lims[2][0]:lims[2][1]]
    cube[np.isnan(cube)] = 0

    _, rms = norm.fit(np.hstack([cube[0 > cube].flatten(),
                            -cube[0 > cube].flatten()]))
    if unit == 'rms':
        cube = cube/rms #rms is in units of cubeunits[0]
    elif unit == 'percent':
        cube = cube/np.nanmax(cube)*100
    elif unit is None:
        pass
    elif unit is not cubeunits[0]:
        cube = cube*u.Unit(cubeunits[0]).to(unit).value

    if isolevels is None:
        isolevels = misc.calc_isolevels(cube)
    else:
        if np.min(isolevels) < np.min(cube):
            raise ValueError('isolevels out of range')
        if np.max(isolevels) > np.max(cube):
            raise ValueError('isolevels out of range')

    colors = misc.create_colormap(colormap, isolevels)

    if image2d is None:
        pass
    elif image2d == 'blank':
        image2d = None, None
    else:
        pixels = 5000
        verts = ((cubecoords[0][0] * u.Unit(cubeunits[1])).to('deg'),
                 (cubecoords[0][1] * u.Unit(cubeunits[1])).to('deg'),
                 (cubecoords[1][0] * u.Unit(cubeunits[2])).to('deg'),
                 (cubecoords[1][1] * u.Unit(cubeunits[2])).to('deg'))
        imcol, img_shape, _ = misc.get_imcol(position=header['OBJECT'], survey=image2d, verts=verts,
                unit='deg', pixels=f'{pixels}', coordinates='J2000') # , grid=True, gridlabels=True
        image2d = imcol, img_shape
        
    if galaxies is not None:
        nx, ny, nz = cube.shape
        trans = (2000/nx, 2000/ny, 2000/nz)
        galdict = misc.get_galaxies(galaxies, cubecoords, cubeunits, header['OBJECT'], delta, trans)
    else: 
        galdict = None

    return Cube(l_cubes=[cube], name=header['OBJECT'], coords=cubecoords, units=cubeunits,
                mags=cubemags, l_colors=[colors], rms=rms, image2d=image2d, galaxies=galdict, 
                l_isolevels=[isolevels])

def prep_mult(cube, spectral_lims, header=None, spatial_lims=None, l_isolevels=None, unit=None,
               colormap=None, image2d=None):
    """
    Prepare the Cube class for a cube with multiple spectral lines.
    Makes a subcube for each spectral line.

    Parameters
    ----------
    cube : str or np.ndarray
        Path to the FITS file or the data cube.
    spectral_lims : list
        List of lists with the limits (minimum and maximum) of the spectral axis.
        E.g. [[vmin1, vmax1], [vmin2, vmax2]].
    header : astropy.io.fits.header.Header, optional
        Header of the FITS file.
    spatial_lims : list, optional
        List of lists with the limits (minimum and maximum) of the spatial axes.
        E.g. [[[ramin1, ramax1], [decmin1, decmax1]], [[ramin2, ramax2], [decmin2, decmax2]]].
        Each list will correspond to the subcubes defined in spectral_lims in the same order.
        If only one set of limits is provided it will be used for all the subcubes.
        If None the whole cube is used. Default is None.
    l_isolevels : list, optional
        List of arrays with the isosurface levels for each cube. If None it is calculated automatically.
    unit : str, optional
        Unit to convert the data cube. If None the original unit is used.
        If 'rms' data is divided by the RMS of the cube.
        If 'percent' data is divided by the maximum value.
    colormap : str, optional
        Name of the colormaps to use. Must be the same length as the number of spectral lines to model.
        If None and the number of lines is <7 they are be chosen automatically. Default is None.
    image2d : str, optional
        Name of the survey to query a 2D image. See survey options in https://astroquery.readthedocs.io/en/latest/skyview/skyview.html.
        If 'blank' a blank plane is used. If None no image is created.
    
    Returns
    -------
    Cube
        An object of the Cube class for a model with multiple spectral lines.
    """
    if isinstance(cube, str):
        with fits.open(cube) as hdul:
            cube = hdul[0].data # this is pol, v, dec, ra
            header = hdul[0].header
    elif header is None:
        raise AttributeError('No header provided')
    
    if len(cube.shape) < 3:
        raise ValueError('Not enough axes')
    if len(cube.shape) >= 4:
        print('Warning: Cube with polarization axis. Using first one.')
        cube = cube[0]
    cube = np.squeeze(cube)

    delta = np.array([header['CDELT1'], header['CDELT2'], header['CDELT3']])
    cubeunits = np.array([header['BUNIT'],header['CUNIT1'], header['CUNIT2'], header['CUNIT3']])
    if cubeunits[0] == 'JY/BEAM':
        cubeunits[0] = 'Jy/beam'
    cubemags = np.array([header['BTYPE'],header['CTYPE1'], header['CTYPE2'], header['CTYPE3']])

    cube = misc.transpose(cube, delta)
    nx, ny, nz = cube.shape

    lims = np.array([[0,nx], [0,ny], [0,nz]], dtype=int)
    cubecoords = np.array([
        [header['CRVAL1']+delta[0]*(lims[0][0]-header['CRPIX1']),
        header['CRVAL1']+delta[0]*(lims[0][1]-header['CRPIX1'])],
        [header['CRVAL2']+delta[1]*(lims[1][0]-header['CRPIX2']),
        header['CRVAL2']+delta[1]*(lims[1][1]-header['CRPIX2'])],
        [header['CRVAL3']+delta[2]*(lims[2][0]-header['CRPIX3']),
        header['CRVAL3']+delta[2]*(lims[2][1]-header['CRPIX3'])]
        ])
    cubecoords = np.sort(cubecoords)

    lims = [[],[]]
    coords = [[],[]]
    if spatial_lims is None:
        lims[0] = [np.array([[0,nx], [0,ny]], dtype=int)] # lims[0] are spatial axes
        coords[0] = [cubecoords[:2]]
    if spatial_lims is not None and isinstance(spatial_lims[0][0][0], u.quantity.Quantity):
        for i in range(len(spatial_lims)):
            for j in range(2):
                spatial_lims[i][j][0] = spatial_lims[i][j][0].to(u.Unit(cubeunits[1])).to_value()
                spatial_lims[i][j][1] = spatial_lims[i][j][1].to(u.Unit(cubeunits[2])).to_value()
            coords[0].append(np.array(spatial_lims[i]))
            lims[0].append(np.array([
                [np.round((coords[0][i][0][0]-header['CRVAL1'])/delta[0] + header['CRPIX1']),
                np.round((coords[0][i][0][1]-header['CRVAL1'])/delta[0] + header['CRPIX1'])],
                [np.round((coords[0][i][1][0]-header['CRVAL2'])/delta[1] + header['CRPIX2']),
                np.round((coords[0][i][1][1]-header['CRVAL2'])/delta[1] + header['CRPIX2'])]
                ], dtype=int))
    elif spatial_lims is not None:
        lims[0] = np.array(spatial_lims)
        for i in range(len(spatial_lims)):
            coords[0].append(np.array([
                [header['CRVAL1']+header['CDELT1']*(spatial_lims[i][0][0]-header['CRPIX1']),
                header['CRVAL1']+header['CDELT1']*(spatial_lims[i][0][1]-header['CRPIX1'])],
                [header['CRVAL2']+header['CDELT2']*(spatial_lims[i][1][0]-header['CRPIX2']),
                header['CRVAL2']+header['CDELT2']*(spatial_lims[i][1][1]-header['CRPIX2'])]
                ]))

    if len(spectral_lims) == 1:
        raise AttributeError('No or single limits for spectral axis. Use prep_one instead.')
    elif len(spectral_lims) < len(lims[0]) :
        raise AttributeError('Not enough spectral limits for the number of spatial limits.')
    elif len(lims[0]) > 1 and len(spectral_lims) != len(lims[0]):
        raise AttributeError('Different number of spectral and spatial limits.')

    if isinstance(spectral_lims[0][0], u.quantity.Quantity):
        for i in range(len(spectral_lims)):
            spectral_lims[i][0] = spectral_lims[i][0].to(u.Unit(cubeunits[3]))
            spectral_lims[i][1] = spectral_lims[i][1].to(u.Unit(cubeunits[3]))
            coords[1].append([spectral_lims[i][0].value, spectral_lims[i][1].value])
            lims[1].append(np.array(
                [np.round((coords[1][i][0]-header['CRVAL3'])/delta[2] + header['CRPIX3']),
                np.round((coords[1][i][1]-header['CRVAL3'])/delta[2] + header['CRPIX3'])],
                dtype=int
                ))
    else:
        for i in range(len(spectral_lims)):
            lims[1].append(np.array(spectral_lims[i]))
            coords[1].append(np.array([
                [header['CRVAL3']+header['CDELT3']*(spectral_lims[i][0]-header['CRPIX3']),
                header['CRVAL3']+header['CDELT3']*(spectral_lims[i][1]-header['CRPIX3'])]
                ]))

    coords[0] = np.sort(coords[0])
    coords[1] = np.sort(coords[1])
    lims[0] = np.sort(lims[0])
    lims[1] = np.sort(lims[1])

    if np.min(lims[0]) < 0:
        raise ValueError('Spatial lims out of range')
    for i in range(len(lims[0])):
        if lims[0][i][0][1] > cube.shape[0] or lims[0][i][1][1] > cube.shape[1]:
            raise ValueError('Spatial lims out of range')
    if np.min(lims[1]) < 0:
        raise ValueError('Spectral lims out of range')
    if np.max(lims[1]) > cube.shape[2]:
        raise ValueError('Spectral lims out of range')

    l_cubes = []
    for i in range(len(coords[1])):
        l_cubes.append(np.zeros(cube.shape))
        if len(coords[0]) == 1:
            j = 0
        else:
            j = i
        l_cubes[i][lims[0][j][0][0]:lims[0][j][0][1], lims[0][j][1][0]:lims[0][j][1][1],
                   lims[1][i][0]:lims[1][i][1]] = \
            cube[lims[0][j][0][0]:lims[0][j][0][1], lims[0][j][1][0]:lims[0][j][1][1],
                 lims[1][i][0]:lims[1][i][1]]
        l_cubes[i][np.isnan(l_cubes[i])] = 0

    _, rms = norm.fit(np.hstack([cube[0 > cube].flatten(),
                            -cube[0 > cube].flatten()]))
    for i in range(len(l_cubes)):
        if unit == 'rms':
            l_cubes[i] = l_cubes[i]/rms
        elif unit == 'percent':
            l_cubes[i] = l_cubes[i]/np.nanmax(l_cubes[i])*100
        elif unit is None:
            pass
        elif unit is not cubeunits[0]:
            l_cubes[i] = l_cubes[i]*u.Unit(cubeunits[0]).to(unit).value

    del cube

    if l_isolevels is None:
        l_isolevels = []
        for c in l_cubes:
            l_isolevels.append(misc.calc_isolevels(c))
    else:
        if len(l_isolevels) != len(l_cubes):
            raise AttributeError('Different number of isolevels and cubes')
        for i in range(len(l_cubes)):
            if np.min(l_isolevels[i]) < np.min(l_cubes[i]):
                raise ValueError('isolevels out of range')
            if np.max(l_isolevels[i]) > np.max(cube[i]):
                raise ValueError('isolevels out of range')

    l_colors = []
    if colormap is None and len(l_cubes) < 7:
        colormap = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'Greys']
    elif colormap is None and len(l_cubes) >= 7:
        raise AttributeError('Too many l_cubes for default colormap. Set colormaps manually.')
    elif len(colormap) != len(l_cubes):
        raise AttributeError('Different number of colormaps and l_cubes')
    for i in range(len(l_cubes)):
        l_colors.append(misc.create_colormap(colormap[i], l_isolevels[i]))

    if image2d is None:
        pass
    elif image2d == 'blank':
        image2d = None, None
    else:
        pixels = 5000
        verts = ((coords[0][0][0][0] * cubeunits[1]).deg, (coords[0][0][0][1] * cubeunits[1]).deg,
                 (coords[0][0][1][0] * cubeunits[2]).deg, (coords[0][0][1][1] * cubeunits[2]).deg)
        imcol, img_shape, _ = misc.get_imcol(position=header['OBJECT'], survey=image2d, verts=verts,
                unit='deg', pixels=f'{pixels}', coordinates='J2000',grid=True, gridlabels=True)
        image2d = imcol, img_shape

    return Cube(l_cubes=l_cubes, name=header['OBJECT'], coords=cubecoords, units=cubeunits,
                mags=cubemags, l_colors=l_colors, rms=rms, image2d=image2d, galaxies=None, 
                l_isolevels=l_isolevels)

def prep_overlay(cube, header=None, spectral_lims=None, lines=None, spatial_lims=None,
                 l_isolevels=None, unit=None, colormap=None, image2d=None):
    """
    Prepare the Cube class for an overlay of spectral lines. Makes a subcube for each spectral line
    and overlays the centre of each line in the same point.

    Parameters
    ----------
    cube : str or np.ndarray
        Path to the FITS file or the data cube.
    header : astropy.io.fits.header.Header, optional
        Header of the FITS file.
    spectral_lims : list, optional
        List of lists with the limits (minimum and maximum) of the spectral axis.
        E.g. [[vmin1, vmax1], [vmin2, vmax2]].
    lines : dict, optional
        Dictionary with the names of spectral lines as keys and a tuple with the centre and
        the full width of the line as value. The centre must be an astropy quantity and the width
        can be a quantity or an integer representing the number of the pixel.
        E.g. {'NII': (6583*u.Angstrom, 50 [pixels])), 'Halpha': (6562*u.Angstrom, 10*u.Angstrom)}.
        If None the centres will be the centres of the subcubes.
    spatial_lims : list, optional
        List of lists with the limits (minimum and maximum) of the spatial axes.
        E.g. [[[ramin1, ramax1], [decmin1, decmax1]], [[ramin2, ramax2], [decmin2, decmax2]]].
        Each list will correspond to the subcubes defined in spectral_lims in the same order.
        If only one set of limits is provided it will be used for all the subcubes.
        If None the whole cube is used. Default is None.
    l_isolevels : list, optional
        List of arrays with the isosurface levels for each cube. If None it is calculated automatically.
    unit : str, optional
        Unit to convert the data cube. If None the original unit is used.
        If 'rms' data is divided by the RMS of the cube.
        If 'percent' data is divided by the maximum value.
    colormap : str, optional
        Name of the colormaps to use. Must be the same length as the number of spectral lines to model.
        If None and the number of lines is <7 they are be chosen automatically. Default is None.
    image2d : str, optional
        Name of the survey to query a 2D image. See survey options in https://astroquery.readthedocs.io/en/latest/skyview/skyview.html.
        If 'blank' a blank plane is used. If None no image is created.
    
    Returns
    -------
    Cube
        An object of the Cube class for an overlay of spectral lines.
    """
    if isinstance(cube, str):
        with fits.open(cube) as hdul:
            cube = hdul[0].data # this is pol, v, dec, ra
            header = hdul[0].header
    elif header is None:
        raise AttributeError('No header provided')
    
    if len(cube.shape) < 3:
        raise ValueError('Not enough axes')
    if len(cube.shape) >= 4:
        print('Warning: Cube with polarization axis. Using first one.')
        cube = cube[0]
    cube = np.squeeze(cube)

    delta = np.array([header['CDELT1'], header['CDELT2'], header['CDELT3']])
    cubeunits = np.array([header['BUNIT'],header['CUNIT1'], header['CUNIT2'], header['CUNIT3']])
    if cubeunits[0] == 'JY/BEAM':
        cubeunits[0] = 'Jy/beam'
    cubemags = np.array([header['BTYPE'],header['CTYPE1'], header['CTYPE2'], header['CTYPE3']])

    cube = misc.transpose(cube, delta)
    nx, ny, nz = cube.shape

    lims = np.array([[0,nx], [0,ny], [0,nz]], dtype=int)
    cubecoords = np.array([
        [header['CRVAL1']+delta[0]*(lims[0][0]-header['CRPIX1']),
        header['CRVAL1']+delta[0]*(lims[0][1]-header['CRPIX1'])],
        [header['CRVAL2']+delta[1]*(lims[1][0]-header['CRPIX2']),
        header['CRVAL2']+delta[1]*(lims[1][1]-header['CRPIX2'])],
        [header['CRVAL3']+delta[2]*(lims[2][0]-header['CRPIX3']),
        header['CRVAL3']+delta[2]*(lims[2][1]-header['CRPIX3'])]
        ])
    cubecoords = np.sort(cubecoords)

    lims = [[],[]]
    coords = [[],[]]
    if spatial_lims is None:
        lims[0] = [np.array([[0,nx], [0,ny]], dtype=int)] # lims[0] are spatial axes
        coords[0] = [cubecoords[:2]]
    if spatial_lims is not None and isinstance(spatial_lims[0][0][0], u.quantity.Quantity):
        for i in range(len(spatial_lims)):
            for j in range(2):
                spatial_lims[i][j][0] = spatial_lims[i][j][0].to(u.Unit(cubeunits[1])).to_value()
                spatial_lims[i][j][1] = spatial_lims[i][j][1].to(u.Unit(cubeunits[2])).to_value()
            coords[0].append(np.array(spatial_lims[i]))
            lims[0].append(np.array([
                [np.round((coords[0][i][0][0]-header['CRVAL1'])/delta[0] + header['CRPIX1']),
                np.round((coords[0][i][0][1]-header['CRVAL1'])/delta[0] + header['CRPIX1'])],
                [np.round((coords[0][i][1][0]-header['CRVAL2'])/delta[1] + header['CRPIX2']),
                np.round((coords[0][i][1][1]-header['CRVAL2'])/delta[1] + header['CRPIX2'])]
                ], dtype=int))
    elif spatial_lims is not None:
        lims[0] = np.array(spatial_lims)
        for i in range(len(spatial_lims)):
            coords[0].append(np.array([
                [header['CRVAL1']+header['CDELT1']*(spatial_lims[i][0][0]-header['CRPIX1']),
                header['CRVAL1']+header['CDELT1']*(spatial_lims[i][0][1]-header['CRPIX1'])],
                [header['CRVAL2']+header['CDELT2']*(spatial_lims[i][1][0]-header['CRPIX2']),
                header['CRVAL2']+header['CDELT2']*(spatial_lims[i][1][1]-header['CRPIX2'])]
                ]))

    if lines is not None:
        if len(lines) < 2:
            raise AttributeError('Not enough lines for overlay. Use prep_one instead.')
        elif len(spatial_lims) != 1 and len(lines) != len(spatial_lims):
            raise AttributeError('Different number of lines and spatial limits.')

        for i, (center, width) in enumerate(lines.values()):
            center = center.to(u.Unit(cubeunits[3]))
            if isinstance(width, u.quantity.Quantity):
                width = width.to(u.Unit(cubeunits[3]))
                coords[1].append(np.array([center.value-width.value/2, center.value+width.value/2]))
                lims[1].append(np.array(
                    [np.round((coords[1][i][0]-header['CRVAL3'])/delta[2] + header['CRPIX3']),
                    np.round((coords[1][i][1]-header['CRVAL3'])/delta[2] + header['CRPIX3'])],
                    dtype=int
                    ))
            else:
                cent_pix = np.round((center.value-header['CRVAL3'])/delta[2] + header['CRPIX3'])
                lims[1].append(np.array([cent_pix-width/2, cent_pix+width/2]))
                coords[1].append(np.array(
                    [header['CRVAL3']+header['CDELT3']*(lims[1][i][0]-header['CRPIX3']),
                    header['CRVAL3']+header['CDELT3']*(lims[1][i][1]-header['CRPIX3'])]
                    ))
    else:
        lines=True

    if spectral_lims is not None:
        if len(spectral_lims) < 2:
            raise AttributeError('Not enough spectral limits for overlay. Use prep_one instead.')
        elif len(spatial_lims) != 1 and len(spectral_lims) != len(spatial_lims):
            raise AttributeError('Different number of spectral and spatial limits.')

        if isinstance(spectral_lims[0][0], u.quantity.Quantity):
            for i in range(len(spectral_lims)):
                spectral_lims[i][0] = spectral_lims[i][0].to(u.Unit(cubeunits[3]))
                spectral_lims[i][1] = spectral_lims[i][1].to(u.Unit(cubeunits[3]))
                coords[1].append([spectral_lims[i][0].value, spectral_lims[i][1].value])
                lims[1].append(np.array(
                    [np.round((coords[1][i][0]-header['CRVAL3'])/delta[2] + header['CRPIX3']),
                    np.round((coords[1][i][1]-header['CRVAL3'])/delta[2] + header['CRPIX3'])],
                    dtype=int
                    ))
        else:
            for i in range(len(spectral_lims)):
                lims[1].append(np.array(spectral_lims[i], dtype=int))
                coords[1].append(np.array([
                    [header['CRVAL3']+header['CDELT3']*(spectral_lims[i][0]-header['CRPIX3']),
                    header['CRVAL3']+header['CDELT3']*(spectral_lims[i][1]-header['CRPIX3'])]
                    ]))

    coords[0] = np.sort(coords[0])
    coords[1] = np.sort(coords[1])
    lims[0] = np.array(np.sort(lims[0]), dtype=int)
    lims[1] = np.array(np.sort(lims[1]), dtype=int)

    if np.min(lims[0]) < 0:
        raise ValueError('Spatial lims out of range')
    for i in range(len(lims[0])):
        if lims[0][i][0][1] > cube.shape[0] or lims[0][i][1][1] > cube.shape[1]:
            raise ValueError('Spatial lims out of range')
    if np.min(lims[1]) < 0:
        raise ValueError('Spectral lims out of range')
    if np.max(lims[1]) > cube.shape[2]:
        raise ValueError('Spectral lims out of range')

    shape = np.array([
        np.squeeze(np.max(np.diff(lims[0]),axis=0)[0]),
        np.squeeze(np.max(np.diff(lims[0]),axis=0)[1]),
        np.max(np.diff(lims[1]),axis=None)], dtype=int)

    l_cubes = []
    for i in range(len(lims[1])):
        l_cubes.append(np.zeros(shape))
        if len(coords[0]) == 1:
            j = 0
        else:
            j = i
        l_cubes[i] = misc.insert_3darray(l_cubes[i],
                    cube[lims[0][j][0][0]:lims[0][j][0][1],lims[0][j][1][0]:lims[0][j][1][1],
                         lims[1][i][0]:lims[1][i][1]])
        l_cubes[i][np.isnan(l_cubes[i])] = 0

    _, rms = norm.fit(np.hstack([cube[0 > cube].flatten(),
                            -cube[0 > cube].flatten()]))
    for i in range(len(l_cubes)):
        if unit == 'rms':
            l_cubes[i] = l_cubes[i]/rms
        elif unit == 'percent':
            l_cubes[i] = l_cubes[i]/np.nanmax(l_cubes[i])*100
        elif unit is None:
            pass
        elif unit is not cubeunits[0]:
            l_cubes[i] = l_cubes[i]*u.Unit(cubeunits[0]).to(unit).value

    del cube

    if l_isolevels is None:
        l_isolevels = []
        for c in l_cubes:
            l_isolevels.append(misc.calc_isolevels(c))
    else:
        if len(l_isolevels) != len(l_cubes):
            raise AttributeError('Different number of isolevels and cubes')
        for i in range(len(l_cubes)):
            if np.min(l_isolevels[i]) < np.min(l_cubes[i]):
                raise ValueError('isolevels out of range')
            if np.max(l_isolevels[i]) > np.max(cube[i]):
                raise ValueError('isolevels out of range')

    l_colors = []
    if colormap is None and len(l_cubes) < 7:
        colormap = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'Greys']
    elif colormap is None and len(l_cubes) >= 7:
        raise AttributeError('Too many l_cubes for default colormap. Set colormaps manually.')
    elif len(colormap) != len(l_cubes):
        raise AttributeError('Different number of colormaps and l_cubes')
    for i in range(len(l_cubes)):
        l_colors.append(misc.create_colormap(colormap[i], l_isolevels[i]))

    if image2d is None:
        pass
    elif image2d == 'blank':
        image2d = None, None
    else:
        pixels = 5000
        verts = ((coords[0][0][0][0] * cubeunits[1]).deg, (coords[0][0][0][1] * cubeunits[1]).deg,
                 (coords[0][0][1][0] * cubeunits[2]).deg, (coords[0][0][1][1] * cubeunits[2]).deg)
        imcol, img_shape, _ = misc.get_imcol(position=header['OBJECT'], survey=image2d, verts=verts,
                unit='deg', pixels=f'{pixels}', coordinates='J2000',grid=True, gridlabels=True)
        image2d = imcol, img_shape

    return Cube(l_cubes=l_cubes, name=header['OBJECT'], coords=cubecoords, units=cubeunits,
                mags=cubemags, l_colors=l_colors, rms=rms, image2d=image2d, galaxies=None, 
                l_isolevels=l_isolevels, lines=lines)

def createX3D(cube, filename, shifts=None):
    """
    Create X3D file from a Cube object.

    Parameters
    ----------
    cube : Cube
        Object of the Cube class.
    filename : str
        Name of the X3D file including the extension (.x3d).
    shift : list, optional
        A list with a arrays of 3D vectors giving the shift in RA, DEC and spectral axis in
        the same units given to the cube. Similar to l_cube or l_isolevels.
    """
    file = writers.WriteX3D(filename, cube)
    file.make_layers(shifts=shifts)
    file.make_outline()
    if cube.galaxies is not None:
        file.make_galaxies(cube.galaxies)
    if cube.image2d is not None:
        file.make_image2d(cube.image2d[0], cube.image2d[1])
    file.make_ticklines()
    file.make_animation()
    file.make_labels(cube.galaxies)
    file.close()

def createHTML(cube, filename, description=None, pagetitle=None):
    """
    Create HTML file from a Cube object.

    Parameters
    ----------
    cube : Cube
        Object of the Cube class.
    filename : str
        Name of the HTML file including the extension (.html).
    description : str, optional
        A description for the web page.
    pagetitle : str, optional
        The title of the web page.
    """
    file = writers.WriteHTML(filename, cube, description, pagetitle)
    if cube.interface == 'minimal':
        file.start_x3d()
        file.viewpoints()
        file.close_x3d(filename.split("/")[-1])
        file.close_html()
    else:
        file.func_layers()
        if cube.galaxies is not None:
            file.func_galaxies()
            file.func_gallab()
        file.func_grids()
        file.func_axes()
        # file.func_pick
        file.func_animation()
        file.start_x3d()
        file.viewpoints()
        file.close_x3d(filename.split("/")[-1])
        file.buttons(centrot=False)
        # mandatory after buttons
        if cube.galaxies is not None:
            file.func_galsize()
        if cube.image2d is not None:
            file.func_image2d()
            file.func_move2dimage()
        file.func_scalev()
        file.func_markers()
        # html.func_setCenterOfRotation(centers)
        file.func_background()
        file.func_colormaps()
        file.close_html()