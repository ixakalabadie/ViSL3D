#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:02:38 2023

@author: ixakalabadie
"""

import numpy as np
from skimage import measure
from matplotlib import cm
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from scipy.stats import norm
from astroquery.ipac.ned import Ned
from astroquery.skyview import SkyView
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy import wcs

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
                * np.cos(cubecoords[1,0]*u.Unit(cubeunits[2]).to('rad'))
            galdec = galdec - np.mean(cubecoords[1])*u.Unit(cubeunits[2])
            galv = galv - np.mean(cubecoords[2])*u.Unit(cubeunits[3])
            galdict[gal['Object Name']] = {
                    'coord': np.array([galra/np.abs(delta[0])*trans[0],
                            galdec/np.abs(delta[1])/trans[2], galv/np.abs(delta[1])/trans[2]]),
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
                * np.cos(cubecoords[1,0]*u.Unit(cubeunits[2]).to('rad'))
            galdec = galdec - np.mean(cubecoords[1])*u.Unit(cubeunits[2])
            galv = galv - np.mean(cubecoords[2])*u.Unit(cubeunits[3])
            galdict[gal] = {
                'coord': np.array([galra/np.abs(delta[0])*trans[0],
                            galdec/np.abs(delta[1])/trans[2], galv/np.abs(delta[1])/trans[2]]),
                'col': '0 0 1'}
    return galdict

class Cube:
    """
    Class to store relevant data to create the visualisation.
    """
    def __init__(self, data, name, coords, units, mags, colors, rms=None, isolevels=None, resol=1, iso_split=None,
                  galaxies=None, image2d=None, interface='full'):
        self.data = data
        self.name = name
        self.coords = coords
        self.units = units
        self.mags = mags
        self.colors = colors
        self.rms = rms
        self.isolevels = isolevels
        self.resol = resol
        self.iso_split = iso_split
        self.galaxies = galaxies
        self.image2d = image2d
        self.interface = interface

def prep_one(cube, header=None, lims=None, unit=None, isolevels=None, colormap='CMRmap_r', image2d=None, galaxies=None):
    """
    
    """
    if isinstance(cube, str):
        with fits.open(cube) as hdul:
            cube = hdul[0].data # this is pol, v, dec, ra
            header = hdul[0].header
    elif header is None:
        raise AttributeError('No header provided')
    
    if len(cube.shape) < 3:
        raise ValueError('Not enough axes')
    if len(cube.shape) == 4 and len(cube[0]) > 1:
        print('Warning: more than one polarization. Using first one.')
        cube = cube[0]
    cube = np.squeeze(cube)

    delta = np.array([header['CDELT1'], header['CDELT2'], header['CDELT3']])
    cubeunits = np.array([header['BUNIT'],header['CUNIT1'], header['CUNIT2'], header['CUNIT3']])
    if cubeunits[0] == 'JY/BEAM':
        cubeunits[0] = 'Jy/beam'
    cubemags = np.array([header['BTYPE'],header['CTYPE1'], header['CTYPE2'], header['CTYPE3']])

    cube = transpose(cube, delta)
    nx, ny, nz = cube.shape
    trans = (2000/nx, 2000/ny, 2000/nz)

    if lims is None:
        lims = np.array([[0, nx-1], [0, ny-1], [0, nz-1]])

    if isinstance(lims, list) and isinstance(lims[0,0], u.quantity.Quantity):
        cubecoords = np.array(lims)
        cubeunits[1:] = lims[0][0].unit, lims[1][0].unit, lims[2][0].unit
        lims = np.array([
            [(header['CRVAL1']-cubecoords[0,0]-delta[0]*(header['CRPIX1']-1))/delta[0],
            (cubecoords[0,1]-header['CRVAL1']+delta[0]*header['CRPIX1'])/delta[0]],
            [(header['CRVAL2']-cubecoords[1,0]-delta[1]*(header['CRPIX2']-1))/delta[1],
            (cubecoords[1,1]-header['CRVAL2']+delta[1]*header['CRPIX2'])/delta[1]],
            [(header['CRVAL3']-cubecoords[2,0]-delta[2]*(header['CRPIX3']-1))/delta[2],
            (cubecoords[2,1]-header['CRVAL3']+delta[2]*header['CRPIX3'])/delta[2]]])
        
    elif isinstance(lims[0,0], int):
        for i in range(3):
            if lims[i,0] < 0:
                raise ValueError('lims out of range')
            if lims[i,1] > cube.shape[i-2]:
                raise ValueError('lims out of range')
    
        cubecoords = np.array([[header['CRVAL1']-delta[0]*(header['CRPIX1']-1-lims[0,0]),
                            header['CRVAL1']+delta[0]*(lims[0,1]-header['CRPIX1'])],
                            [header['CRVAL2']-delta[1]*(header['CRPIX2']-1-lims[1,0]),
                            header['CRVAL2']+delta[1]*(lims[1,1]-header['CRPIX2'])],
                            [header['CRVAL3']-delta[2]*(header['CRPIX3']-1-lims[2,0]),
                            header['CRVAL3']+delta[2]*(lims[2,1]-header['CRPIX3'])]])
    elif lims is not None:
        raise TypeError('lims in wrong format')
        
    cubecoords = np.sort(cubecoords)
    cube = cube[lims[2,0]:lims[2,1],lims[1,0]:lims[1,1],lims[0,0]:lims[0,1]]
    cube[np.isnan(cube)] = 0

    _, rms = norm.fit(np.hstack([cube[0 > cube].flatten(),
                            -cube[0 > cube].flatten()]))
    if unit == 'rms':
        cube = cube/rms #rms is in units of cubeunits[0]
    elif unit == 'percent':
        cube = cube/np.nanmax(cube)*100
    elif unit is not cubeunits[0]:
        cube = cube*u.Unit(cubeunits[0]).to(unit).value

    if isolevels is None:
        calc_isolevels(cube)
    else:
        if np.min(isolevels) < np.min(cube):
            raise ValueError('isolevels out of range')
        if np.max(isolevels) > np.max(cube):
            raise ValueError('isolevels out of range')
    
    colors = create_colormap(colormap, isolevels)
    
    if image2d is None:
        pass
    elif image2d == 'blank':
        imcol, img_shape = None, None
    else:
        pixels = 5000
        verts = ((cubecoords[0,0] * cubeunits[1]).deg, (cubecoords[0,1] * cubeunits[1]).deg,
                 (cubecoords[1,0] * cubeunits[2]).deg, (cubecoords[1,1] * cubeunits[2]).deg)
        imcol, img_shape, _ = get_imcol(position=header['OBJECT'], survey=image2d, verts=verts,
                unit='deg', pixels=f'{pixels}', coordinates='J2000',grid=True, gridlabels=True)
        
    galdict = get_galaxies(galaxies, cubecoords, cubeunits, header['OBJECT'], delta, trans)

    return Cube(data=cube, name=header['OBJECT'], coords=cubecoords, units=cubeunits,
                mags=cubemags, colors=colors, rms=rms, image2d=(imcol, img_shape), galaxies=galdict)


# std_coord = 1000

def createX3D(cube, filename, shifts=None):

    file = WriteX3D(filename, cube)
    file.make_layers(cube.data, cube.isolevels, cube.colors, shifts=shifts)
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

    file = WriteHTML(filename, cube, description, pagetitle)
    if cube.interface == 'minimal':
        file.start_x3d()
        file.viewpoints()
        file.close_x3d(filename)
        file.close_html()
    else:
        file.func_layers(cube.isolevels)
        if cube.galaxies is not None:
            file.func_galaxies()
            file.func_gallab()
        file.func_grids()
        file.func_axes()
        # file.func_pick
        file.func_animation()
        file.start_x3d()
        file.viewpoints()
        file.close_x3d(filename)
        file.buttons(cube.isolevels, cube.colors, linelabs=False, centrot=False)
        if cube.image2d is not None:
            file.func_image2d()
            file.func_move2dimage()
        file.func_scalev()
        file.func_tubes()
        # html.func_setCenterOfRotation(centers)
        file.func_background()
        file.func_colormaps(cube.isolevels)
        file.close_html()


class WriteX3D:
    """
    Class to create a X3D model of iso-surfaces with 3D spectral line data.
    creates an X3D file with the model.
    """

    def __init__(self, filename, cube):
        self.cube = cube
        self.file_x3d = open(filename, encoding="utf-8")
        self.file_x3d.write('<?xml version="1.0" encoding="UTF-8"?>\n <!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN" \n "http://www.web3d.org/specifications/x3d-3.3.dtd">')
        self.file_x3d.write('\n <X3D profile="Immersive" version="3.3" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.3.xsd">')
        self.file_x3d.write(f'\n <head>\n\t<meta name="file" content="{filename}"/>\n')
        # Additional metadata MAKE AUTOMATIC
        # if meta != None:
        #     for met in meta.keys():
        #         self.file_x3d.write('\n\t<meta name="%s" content="%s"/>'%(met,meta[met]))
        self.file_x3d.write(
            f'</head>\n\t<Scene doPickPass="{cube.picking}">\n' + \
                '\t\t<Background DEF="back" skyColor="0.6 0.6 0.6"/>\n')
        self.file_x3d.write(
            tabs(2)+'<NavigationInfo type=\'"EXAMINE" "ANY"\' speed="4" headlight="true"/>\n')
        self.file_x3d.write(
            tabs(2)+'<DirectionalLight ambientIntensity="1" intensity="0" color="1 1 1"/>\n')
        self.file_x3d.write(tabs(2)+'<Transform DEF="ROOT" translation="0 0 0">\n')

    def make_layers(self, l_cubes, l_isolevels, l_colors, shifts=None, add_normals=False):
        """
        Calculate iso-surfaces from the data and write the objects in the X3D file.

        Parameters
        ----------
        l_cube : list of 3d arrays
            The cubes to plot. Axis must be RA, DEC and spectral axis.
            If more than one cube, the shape of all cubes must be the same,
            just set unknown or unwanted values to 0 (OR NAN?).
        l_isolevels : list of arrays
            A list of arrays with the values of each isosurface layer, one array for each cube. 
            E.g.[2,5,9] for three layers at values 2,5 and 9 of the cube.
            Should be in increasing order. (REAL?)
        l_colors : list of arrays
            RGB color for each isolevel a string ('122 233 20'). There must be a list of colors for each cube.
        shift : list, optional
            A list with a arrays of 3D vectors giving the shift in RA, DEC and spectral axis in the same units given to the cube. Similar to l_cube or l_isolevels.

        """

        numcubes = len(l_cubes)
        
        for nc in range(numcubes):
            cube_full = l_cubes[nc]
            isolevels = l_isolevels[nc]
            self.cube.iso_split.append(np.zeros((len(isolevels)), dtype=int))
            for i in range(len(isolevels)):

                split = int(np.sum(cube_full>isolevels[i])/700000)+1 # calculate how many times to split the cube, 1 means the cube stays the same
                self.cube.iso_split[nc][i] = split
                nx,ny,nz = cube_full.shape

                for sp in range(split):

                    cube = cube_full[:,:,int(nz/split*sp):int(nz/split*(sp+1))]
                    try:
                        if shifts is not None:
                            verts, faces, normals = marching_cubes(cube, level=isolevels[i],
                                        shift=shifts[nc], step_size=self.cube.resol)
                        else:
                            verts, faces, normals = marching_cubes(cube, level=isolevels[i],
                                                            step_size=self.cube.resol)
                    except Exception as ex:
                        print(ex)
                        continue
                    self.file_x3d.write('\n\t\t\t<Transform DEF="%slt%s_sp%s" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">'%(nc,i,sp))
                    self.file_x3d.write('\n\t\t\t\t<Shape DEF="%slayer%s_sp%s_shape">'%(nc,i,sp))
                    self.file_x3d.write('\n\t\t\t\t\t<Appearance sortKey="%s">'%(len(isolevels)-1-i))
                    style = 'transparent'
                    if style == 'transparent':
                        #set color and transparency of layer
                        if i == len(isolevels)-1:
                            op = 0.4
                        else:
                            op = 0.8
                        self.file_x3d.write('\n'+tabs(6)+'<Material DEF="%slayer%s_sp%s" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="%s" specularColor="0 0 0" shininess="0.0078" transparency="%s"/>'%(nc,i,sp,l_colors[nc][i],op))
                    elif style == 'opaque':
                        #set color of layer, transparency is set in HTML
                        self.file_x3d.write('\n'+tabs(6)+'<Material DEF="%slayer%s_sp%s" ambientIntensity="0" emissiveColor="%s" diffuseColor="%s" specularColor="0 0 0" shininess="0.0078"/>'%(nc,i,sp,l_colors[nc][i],l_colors[nc][i]))
                    #correct color with depthmode (ALSO FOR LAST LAYER?)
                    # if i != len(isolevels)-1:
                    self.file_x3d.write('\n'+tabs(6)+'<DepthMode readOnly="true"></DepthMode>')
                    self.file_x3d.write('\n'+tabs(5)+'</Appearance>')
                    #define the layer object
                    if add_normals:
                        self.file_x3d.write('\n'+tabs(5)+'<IndexedFaceSet solid="false" colorPerVertex="false" normalPerVertex="true" coordIndex="\n\t\t\t\t\t\t')
                    else:
                        self.file_x3d.write('\n'+tabs(5)+'<IndexedFaceSet solid="false" colorPerVertex="false" normalPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
                    #write indices
                    np.savetxt(self.file_x3d, faces, fmt='%i', newline=' -1\n\t\t\t\t\t\t')
                    self.file_x3d.write('">')
                    self.file_x3d.write('\n\t\t\t\t\t\t<Coordinate DEF="%sCoordinates%s_sp%s" point="\n\t\t\t\t\t\t'%(nc,i,sp))
                    #write coordinates
                    np.savetxt(self.file_x3d, verts,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
                    self.file_x3d.write('"/>')
                    if add_normals:
                        self.file_x3d.write('\n\t\t\t\t\t\t<Normal DEF="%sNormals%s_sp%s" vector="\n\t\t\t\t\t\t'%(nc,i,sp))
                        #write normals
                        np.savetxt(self.file_x3d, normals,fmt='%.5f', newline=',\n\t\t\t\t\t\t')
                        self.file_x3d.write('"/>')
                    self.file_x3d.write('\n\t\t\t\t\t</IndexedFaceSet>\n\t\t\t\t</Shape>\n\t\t\t</Transform>')

    def make_outline(self):
        """
        Creates an object for an outline in the X3D file.
        """
        outlinecoords = np.array([[-1000,-1000,-1000],
                                  [1000,-1000,-1000],
                                  [-1000,1000,-1000],
                                  [1000,1000,-1000],
                                  [-1000,-1000,1000],
                                  [1000,-1000,1000],
                                  [-1000,1000,1000],
                                  [1000,1000,1000]])
        # Create outline
        self.file_x3d.write('\n\t\t\t<Transform DEF="ot" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">')
        self.file_x3d.write('\n\t\t\t\t<Shape ispickable="false">')
        self.file_x3d.write('\n\t\t\t\t\t<Appearance>')
        #define ouline ID
        col = '0 0 0'
        self.file_x3d.write('\n\t\t\t\t\t\t<Material DEF="outline" emissiveColor="%s" diffuseColor="0 0 0"/>'%col)
        self.file_x3d.write('\n\t\t\t\t\t</Appearance>')
        self.file_x3d.write('\n\t\t\t\t\t<IndexedLineSet colorPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
        #write indices
        np.savetxt(self.file_x3d, outlineindex, fmt='%i', newline='\n\t\t\t\t\t\t')
        self.file_x3d.write('">')
        self.file_x3d.write('\n\t\t\t\t\t\t<Coordinate DEF="OutlineCoords" point="\n\t\t\t\t\t\t')
        #write coordinates
        np.savetxt(self.file_x3d, outlinecoords,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
        self.file_x3d.write('"/>')
        self.file_x3d.write('\n\t\t\t\t\t</IndexedLineSet>\n\t\t\t\t</Shape>\n\t\t\t</Transform>\n')
        
    def make_galaxies(self, gals):
        """
        Creates spheres and labels in the model at the location of galaxies given as input.

        Parameters
        ----------
        gals : dictionary
            DESCRIPTION.

        """        
        sphereradius = 2000/45
        crosslen = 2000/20
        #create galaxy crosses and spheres
        for i, gal in enumerate(gals.keys()):
            #galaxy crosses
            self.file_x3d.write(tabs(3)+f'<Transform DEF="{gal}_cross_tra" ' \
                                +'translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n')
            self.file_x3d.write(tabs(4)+'<Shape ispickable="false">\n')
            self.file_x3d.write(tabs(5)+'<Appearance>\n')
            col = '0 0 0'
            self.file_x3d.write(tabs(6)+f'<Material DEF="{gal}_cross" emissiveColor="{col}" ' \
                                + 'diffuseColor="0 0 0"/>\n')
            self.file_x3d.write(tabs(5)+'</Appearance>\n')
            #cross indices
            self.file_x3d.write(tabs(5)+'<IndexedLineSet colorPerVertex="true" coordIndex="\n' \
                        +tabs(6)+'0 1 -1\n'+tabs(6)+'2 3 -1\n'+tabs(6)+'4 5 -1\n'+tabs(6)+'">\n')
            self.file_x3d.write(tabs(5)+'<Coordinate DEF="CrossCoords%s" point="\n\t\t\t\t\t\t'%i)
            vec = gals[gal]['coord']
            crosscoords = np.array([[vec[0]-crosslen,vec[1],vec[2]],
                              [vec[0]+crosslen,vec[1],vec[2]],
                              [vec[0],vec[1]-crosslen, vec[2]],
                              [vec[0],vec[1]+crosslen, vec[2]],
                              [vec[0],vec[1],vec[2]-crosslen],
                              [vec[0],vec[1],vec[2]+crosslen]])
            #cross coordinates
            np.savetxt(self.file_x3d, crosscoords, fmt='%.3f', newline='\n\t\t\t\t\t\t')
            self.file_x3d.write(tabs(6)+'"/>\n')
            self.file_x3d.write(tabs(3)+'</IndexedLineSet>\n\t\t\t\t</Shape>\n\t\t\t</Transform>\n')
            #galaxy spheres (ADD SCALE, ROTATION, ETC.??)
            self.file_x3d.write(tabs(3)+'<Transform DEF="%s_sphere_tra" translation="%s %s %s">\n'%(gal,vec[0],vec[1],vec[2]))
            self.file_x3d.write(tabs(4)+'<Shape ispickable="false">\n')
            self.file_x3d.write(tabs(5)+'<Sphere radius="%s" solid="false"/>\n'%sphereradius)
            self.file_x3d.write(tabs(5)+'<Appearance>\n')
            self.file_x3d.write(tabs(6)+'<Material DEF="%s" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="%s" specularColor="0 0 0" shininess="0.0078" transparency="0"/>\n'%(gal,gals[gal]['col']))
            self.file_x3d.write(tabs(5)+'</Appearance>\n\t\t\t\t</Shape>\n\t\t\t</Transform>\n')
            
    def make_image2d(self, imcol=None, img_shape=None):
        """
        Create a 2D image object in the X3D model.

        Parameters
        ----------
        imcol : array, optional
            Array with hexadecimal colors of each pixel for a 2D image. If None, a white image is created.
            Default is None
        img_shape : tuple, optional
            Shape of the 2D image. Use None for white image. Default is None.

        """

        # coordinates of 2d image
        coords2d = np.array([[1000,-1000,1000],
                             [1000,1000,1000],
                             [-1000,-1000,1000],
                             [-1000,1000,1000]])
        
        self.file_x3d.write(tabs(3)+'<Transform DEF="image2d" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n')
        self.file_x3d.write(tabs(4)+'<Shape ispickable="False">\n')
        self.file_x3d.write(tabs(5)+'<Appearance>\n')
        self.file_x3d.write(tabs(6)+'<Material DEF="immat" ambientIntensity="1" emissiveColor="0 0 0" diffuseColor="1 1 1" shininess="0.0078"/>\n')
        if imcol is not None and img_shape is not None:
            self.file_x3d.write(tabs(6)+'<PixelTexture repeatS="false" repeatT="false" image=" %s %s 3 \n'%(img_shape[0], img_shape[1]))
            # write pixel colors
            np.savetxt(self.file_x3d, imcol, fmt='%s', delimiter=' ', newline='\n')
            self.file_x3d.write('"/>\n')
        self.file_x3d.write(tabs(5)+'</Appearance>\n')            
        #SOLID=TRUE makes it transparent from one side
        self.file_x3d.write(tabs(5)+'<IndexedFaceSet solid="false" colorPerVertex="false" normalPerVertex="false" coordIndex="2 3 1 0 -1">\n')
        self.file_x3d.write(tabs(6)+'<Coordinate DEF="imgCoords" point="\n\t\t\t\t\t\t')
        # write coordinates
        np.savetxt(self.file_x3d, coords2d, fmt='%.3f', newline='\n\t\t\t\t\t\t')
        self.file_x3d.write('"/>\n')
        self.file_x3d.write(tabs(6)+'<TextureCoordinate DEF="imgTexCoords" point="\n'+tabs(6)+' 0 0, 1 0, 0 1, 1 1"/>\n')
        self.file_x3d.write(tabs(5)+'</IndexedFaceSet>\n')
        self.file_x3d.write(tabs(4)+'</Shape>\n')
        self.file_x3d.write(tabs(3)+'</Transform>\n')
            
    def make_ticklines(self):
        """
        Create tickline objects in the X3D model.
        Must be change to close the Transform "ROOT" somewhere else, in case this is not used.
        Change method to end ROOT with new module structure
        """
        # coordinates of tick lines
        ticklinecoords = np.array([[-1000,0,-1000],
                                   [1000,0,-1000],
                                   [0,-1000,-1000],
                                   [0,1000,-1000],
                                   [-1000,0,-1000],
                                   [-1000,0,1000],
                                   [-1000,-1000,0],
                                   [-1000,1000,0],
                                   [0,1000,-1000],
                                   [0,1000,1000],
                                   [-1000,1000,0],
                                   [1000,1000,0]])
        #Create ticklines
        self.file_x3d.write(tabs(3)+'<Transform DEF="tlt" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n')
        self.file_x3d.write(tabs(4)+'<Shape ispickable="false">\n')
        self.file_x3d.write(tabs(5)+'<Appearance>\n')
        #set color
        col = '0 0 0'
        self.file_x3d.write(tabs(6)+'<Material DEF="ticklines"  emissiveColor="%s" diffuseColor="0 0 0"/>\n'%col)
        self.file_x3d.write(tabs(5)+'</Appearance>\n')
        self.file_x3d.write(tabs(5)+'<IndexedLineSet colorPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
        #write indices
        np.savetxt(self.file_x3d, ticklineindex, fmt='%i', newline='\n\t\t\t\t\t\t')
        self.file_x3d.write('">\n')
        self.file_x3d.write(tabs(6)+'<Coordinate DEF="ticklineCoords" point="\n\t\t\t\t\t\t')
        #write coordinates
        np.savetxt(self.file_x3d, ticklinecoords,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
        self.file_x3d.write('"/>\n')
        self.file_x3d.write(tabs(5)+'</IndexedLineSet>\n')
        self.file_x3d.write(tabs(4)+'</Shape>\n')
        self.file_x3d.write(tabs(3)+'</Transform>\n')
        self.file_x3d.write(tabs(2)+'</Transform>\n')

    def make_animation(self, cycleinterval=10, axis=0):
        """
        Create an animation to rotate the X3D model along one axis.

        Must be outside the Transform "ROOT" element (for now write after make_ticklines). 
        """
        vec = np.zeros(3,dtype=int)
        vec[axis] = 1
        vec = str(vec)[1:-1]
        self.file_x3d.write('\n'+tabs(2)+'<timeSensor DEF="time" cycleInterval="%s" loop="true" enabled="true" startTime="-1"></timeSensor>'%cycleinterval)
        self.file_x3d.write('\n'+tabs(2)+'<OrientationInterpolator DEF="move" key="0 0.5 1" keyValue="%s 0  %s 3.14  %s 6.28"/>'%(vec,vec,vec))
        self.file_x3d.write('\n'+tabs(2)+'<Route fromNode="time" fromField ="fraction_changed" toNode="move" toField="set_fraction"></Route>')
        self.file_x3d.write('\n'+tabs(2)+'<Route fromNode="move" fromField ="value_changed" toNode="ROOT" toField="rotation"></Route>')
                
        
    def make_labels(self, gals=None):
        """
        Create the labels of different elements in the figure.

        Parameters
        ----------
        gals : dictionary, optional
            Dictionary with the names of the galaxies as keys and another
            dictionary inside them with: 'v': the radio velocity, 'col': the RGB color
            and 'coord' the spatial coordinates with the same format as the rest
            of the elements. If None galaxy labels are not created. The default is None.
        axlab : string, optional
            A string indicating what axis labels to include. Can be 'real' for the equatorial
            coordinates; 'diff' for the difference from the center of the cube or
            'both' for both options. Leave None for no axis labels.
            The default is None.

        """
        self.file_x3d.write('\n\t\t<ProximitySensor DEF="PROX_LABEL" size="1.0e+06 1.0e+06 1.0e+06"/>')
        self.file_x3d.write('\n\t\t<Collision enabled="false">')
        self.file_x3d.write('\n\t\t\t<Transform DEF="TRANS_LABEL">')

        ramin1, _, ramax1 = (self.cube.cubecoords[0]-np.mean(self.cube.cubecoords[0])) \
                    * np.cos(self.cube.cubecoords[1,0]*u.Unit(self.cube.cubeunits[2]).to('rad')) \
                    * u.Unit(self.cube.cubeunits[1]).to('arcsec')
        decmin1, _, decmax1 = (self.cube.cubecoords[1]-np.mean(self.cube.cubecoords[1])) \
                    * u.Unit(self.cube.cubeunits[2]).to('arcsec')
        vmin1, _, vmax1 = (self.cube.cubecoords[2]-np.mean(self.cube.cubecoords[2])) \
                    * u.Unit(self.cube.cubeunits[3])
        
        ramin2, _, ramax2 = Angle(self.cube.cubecoords[0] \
                            * u.Unit(self.cube.cubeunits[1])).to_string(u.hour, precision=0)
        decmin2, _, decmax2 = Angle(self.cube.cubecoords[1] \
                            * u.Unit(self.cube.cubeunits[2])).to_string(u.degree, precision=0)
        vmin2, _, vmax2 = self.cube.cubecoords[2]
        
        # scale of labels
        labelscale = 20

        ax, axtick = labpos
        
        #Names for the axes tick labels
        axticknames1 = np.array([f'{ramax1:.2f}',f'{ramin1:.2f}',f'{decmax1:.2f}',
                       f'{decmin1:.2f}',f'{vmin1:.0f}',f'{vmax1:.0f}',
                       f'{decmax1:.2f}',f'{decmin1:.2f}',f'{vmin1:.0f}',
                       f'{vmax1:.0f}',f'{ramax1:.2f}',f'{ramin1:.2f}'])
        
        axticknames2 = np.array([ramax2, ramin2, decmax2,
                       decmin2, f'{vmin2:.0f}', f'{vmax2:.0f}',
                       decmax2, decmin2, f'{vmin2:.2f}',
                       f'{vmax2:.0f}', ramax2, ramin2])
        
        col = '0 0 0'
        
        #galaxy labels
        if gals:
            for (i,gal) in enumerate(gals.keys()):
                self.file_x3d.write('\n\t\t\t\t<Transform DEF="glt%s" translation="%s %s %s" rotation="0 1 0 3.14" scale="%s %s %s">'%(i,gals[gal]['coord'][0],gals[gal]['coord'][1], gals[gal]['coord'][2], labelscale, labelscale, labelscale))
                self.file_x3d.write('\n\t\t\t\t\t<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">')
                self.file_x3d.write('\n\t\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t<Material DEF="%s" diffuseColor="0 0 0" emissiveColor="%s"/>'%('label_'+gal,col))
                self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t<Text string="%s">'%gal)
                self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle DEF="%s_fs" family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="8"/>'%gal)
                self.file_x3d.write('\n\t\t\t\t\t\t</Text> \n\t\t\t\t\t\t</Shape>\n\t\t\t\t\t</Billboard>\n\t\t\t\t</Transform>')
        
        axlabnames = get_axlabnames(mags=self.cube.mags, units=self.cube.units)



        for i in range(6):
            self.file_x3d.write('\n\t\t\t\t<Transform DEF="alt_diff%s" translation="%s %s %s" rotation="%s" scale="%s %s %s">'%(i,ax[i,0],ax[i,1],ax[i,2],axlabrot[i], labelscale, labelscale, labelscale))
            self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
            self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axlab_diff%s" diffuseColor="0 0 0" emissiveColor="%s"/>'%(i,col))
            self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
            self.file_x3d.write("\n\t\t\t\t\t\t<Text string='%s'>"%axlabnames[i])
            self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'%s\' size="10"/>'%axlabeljustify[i])
            self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
        #ax tick labels
        for i in range(12):
            if i<4: rot = axlabrot[0]
            elif i<8: rot = axlabrot[2]
            else: rot = axlabrot[4]
            self.file_x3d.write('\n\t\t\t\t<Transform DEF="att_diff%s" translation="%s %s %s" rotation="%s" scale="%s %s %s">'%(i,axtick[i,0],axtick[i,1],axtick[i,2], rot, labelscale, labelscale, labelscale))
            self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
            self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axtick_diff%s" diffuseColor="0 0 0" emissiveColor="%s"/>'%(i,col))
            self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
            self.file_x3d.write('\n\t\t\t\t\t\t<Text string="%s">'%axticknames1[i])
            self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'%s\' size="8"/>'%axticklabjus[i])
            self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
    
        #ax labels
        for i in range(6):
            self.file_x3d.write('\n\t\t\t\t<Transform DEF="alt_real%s" translation="%s %s %s" rotation="%s" scale="%s %s %s">'%(i,ax[i,0],ax[i,1],ax[i,2],axlabrot[i], labelscale, labelscale, labelscale))
            self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
            self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axlab_real%s" diffuseColor="0 0 0" emissiveColor="%s" transparency="1"/>'%(i,col))
            self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
            self.file_x3d.write("\n\t\t\t\t\t\t<Text string='%s'>"%axlabnames[i])
            self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'%s\' size="10"/>'%axlabeljustify[i])
            self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
        #ax tick labels
        for i in range(12):
            if i<4: rot = axlabrot[0]
            elif i<8: rot = axlabrot[2]
            else: rot = axlabrot[4]
            self.file_x3d.write('\n\t\t\t\t<Transform DEF="att_real%s" translation="%s %s %s" rotation="%s" scale="%s %s %s">'%(i,axtick[i,0],axtick[i,1],axtick[i,2], rot, labelscale, labelscale, labelscale))
            self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
            self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axtick_real%s" diffuseColor="0 0 0" emissiveColor="%s" transparency="1"/>'%(i,col))
            self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
            self.file_x3d.write('\n\t\t\t\t\t\t<Text string="%s">'%axticknames2[i])
            self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'%s\' size="8"/>'%axticklabjus[i])
            self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
            
        self.file_x3d.write('\n\t\t\t</Transform>')
        self.file_x3d.write('\n\t\t</Collision>')
                
    def close(self):
        """
        Closes the X3D file. Not using this function at the end results in
        an error.
        """
        #ending, close all
        self.file_x3d.write('\n\t\t<ROUTE fromNode="PROX_LABEL" fromField="position_changed" toNode="TRANS_LABEL" toField="set_translation"/>')
        self.file_x3d.write('\n\t\t<ROUTE fromNode="PROX_LABEL" fromField="orientation_changed" toNode="TRANS_LABEL" toField="set_rotation"/>')
        self.file_x3d.write('\n\t</Scene>')
        self.file_x3d.write('\n</X3D>')
        self.file_x3d.close()

class WriteHTML:
    """
    Create an HTML file with an embbeded X3D figure and functions to interact with it.

    Parameters
    ----------
    filename : string
        Name of the file to be created, should have ".html" extension.
    units : list
        Strings with units of the cube. Don't have to be real units, they are used just to make labels.
    l_isolevels : list
        list of lists of isolevels. Must be a list even if its just one line. e.g. [[0.5,2,5]].
    tabtitle : string, optional
        Title of the tab in the web browser. The default is 'new_html_x3d'.
    pagetitle : string, optional
        Title of the web page. The default is None.
    description : string, optional
        Description of the figure or any other text to be included in the web page.
        Should follow HTML format. The default is None.
    style : string, optional
        Style of the surfaces in the model. Can be 'transparent' or 'opaque'. The default is 'transparent'.
    format : string, optional
        Format of the HTML. Can be 'full' for a fully interactive web page or
        'minimal' to just show the model with default X3DOM interactive options.
        The default is 'full'.
        
    """
    def __init__(self, filename, cube, description=None, pagetitle=None):
        #some attributes to use later
        self.cube = cube
        if pagetitle is None:
            pagetitle = self.cube.name
        self.file_html = open(filename, encoding="utf-8")
        self.file_html.write('<html>\n\t <head>\n')
        self.file_html.write(tabs(2)+"<script type='text/javascript' src='x3dom/x3dom.js'></script>\n")
        self.file_html.write(tabs(2)+"<script type='text/javascript'  src='https://www.maths.nottingham.ac.uk/plp/pmadw/LaTeXMathML.js'></script>\n")
        self.file_html.write(tabs(2)+"<link rel='stylesheet' type='text/css' href='x3dom/x3dom.css'></link>\n")
        self.file_html.write(tabs(2)+"<script type='text/javascript' src='https://code.jquery.com/jquery-3.6.3.min.js'></script>\n")
        self.file_html.write(tabs(2)+'<script src="x3dom/js-colormaps.js"></script> <!-- FOR COLORMAPS IN JS-->\n')
        if self.cube.interface == 'minimal':
            self.file_html.write("\n\t\t<style>\n"+tabs(3)+"x3d\n"+tabs(4)+"{\n"+tabs(5)+"border:2px solid darkorange;\n"+tabs(5)+"width:100%;\n"+tabs(5)+"height: 100%;\n"+tabs(3)+"}\n"+tabs(3)+"</style>\n\t</head>\n\t<body>\n")
        else:
            self.file_html.write(tabs(2)+f'<title> {self.cube.name} </title>\n')
            self.file_html.write("\n\t\t<style>\n"+tabs(3)+"x3d\n"+tabs(4)+"{\n"+tabs(5)+"border:2px solid darkorange;\n"+tabs(5)+"width:95%;\n"+tabs(5)+"height: 80%;\n"+tabs(3)+"}\n"+tabs(3)+"</style>\n\t</head>\n\t<body>\n")
        self.file_html.write(f'\t<h1 align="middle"> {pagetitle} </h1>\n')
        self.file_html.write('\t<hr/>\n')
        if description is not None:
            self.file_html.write(f"\t<p>\n\t {description}</p> \n")

        #ANOTHER WAY TO CHANGE TRANSPARENCY instead of loading()
        # self.file_html.write(tabs(3)+"const nl = [%s,%s,%s];"%())
        # self.file_html.write(tabs(3)+"for (let nc = 0; nc < %s; nc++) {\n"%len(l_isolevels))
        # self.file_html.write(tabs(4)+"for (let nl = 0; nl < ; nl++) {\n"%len(l_isolevels[nc]))
        # self.file_html.write(tabs(5)+"if (nl === %s) \n"+tabs(6)+"const op = 0.4;\n"+tabs(5)+"else\n"+tabs(6)+"const op = 0.8;\n")
        # self.file_html.write(tabs(5)+"document.getElementById('cube__'+nc+'layer'+nl).setAttribute('transparency', op);\n")

        # if self.style == 'opaque':
        #     self.file_html.write(tabs(1)+'<script>\n')
        #     self.file_html.write(tabs(2)+'function loading() {\n')
        #     numcubes = len(l_isolevels)
        #     for nc in range(numcubes):
        #         isolevels = l_isolevels[nc]
        #         for nl in range(len(isolevels)):
        #             if nl == len(isolevels)-1:
        #                 op = 0.4
        #             else:
        #                 op = 0.8
        #             for sp in range(self.iso_split[nc][nl]):
        #                 self.file_html.write(tabs(3)+"document.getElementById('cube__%slayer%s_sp%s').setAttribute('transparency', '%s');\n"%(nc,nl,sp,op))
            
            self.file_html.write(tabs(2)+'}\n')
            self.file_html.write(tabs(1)+'</script>\n')

        # setTimeout(loading, 5000); option to execute function after time
        
    def func_layers(self, l_isolevels):
        """
        Make the funcion to hide/show layers. If this is used, the buttons()
        function with isolevels should also be used to create the buttons.
        The X3D file must have layers for this to work.

        Parameters
        ----------
        l_isolevels : list
            list of lists of isolevels. Same as for make_layers().
        split : array
            self.iso_split attribute from the write_x3d() class.

        """
        numcubes = len(l_isolevels)
        nlayers = [len(l) for l in l_isolevels]
        self.file_html.write(tabs(2)+"<script>\n")
        self.file_html.write(tabs(3)+"function hideall() {\n")
        for nc in range(numcubes):
            for i in range(nlayers[nc]):
                self.file_html.write(tabs(4)+"setHI%slayer%s();\n"%(nc,i))
        self.file_html.write(tabs(3)+"}\n")
        self.file_html.write(tabs(2)+"</script>\n")
        for nc in range(numcubes):
            for i in range(nlayers[nc]):
                if i != nlayers[nc]-1:
                    self.file_html.write("\t <script>\n\t \t function setHI%slayer%s()\n\t \t {\n\t \t if(document.getElementById('cube__%slayer%s_sp0').getAttribute('transparency') != '0.8') {\n"%(nc,i,nc,i))
                    self.file_html.write("\t\t document.getElementById('%sbut%s').style.border = '5px dashed black';\n"%(nc,i))
                    for sp in range(self.cube.iso_split[nc][i]):
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s').setAttribute('transparency', '0.8');\n"%(nc,i,sp))
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s_shape').setAttribute('ispickable', 'true');\n"%(nc,i,sp))
                    self.file_html.write("\t\t } else { \n")
                    self.file_html.write("\t\t document.getElementById('%sbut%s').style.border = 'inset black';\n"%(nc,i))
                    for sp in range(self.cube.iso_split[nc][i]):
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s').setAttribute('transparency', '1');\n"%(nc,i,sp))
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s_shape').setAttribute('ispickable', 'false');\n"%(nc,i,sp))
                    self.file_html.write("\t\t } \n\t\t }\n\t </script>\n")
                else:
                    self.file_html.write("\t <script>\n\t\t function setHI%slayer%s()\n\t\t {\n\t \t if(document.getElementById('cube__%slayer%s_sp0').getAttribute('transparency') != '0.4') {\n"%(nc,i,nc,i))
                    self.file_html.write("\t\t document.getElementById('%sbut%s').style.border = '5px dashed black';\n"%(nc,i))
                    for sp in range(self.cube.iso_split[nc][i]):
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s').setAttribute('transparency', '0.4');\n"%(nc,i,sp))
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s_shape').setAttribute('ispickable', 'true');\n"%(nc,i,sp))
                    self.file_html.write("\t\t } else { \n")
                    self.file_html.write("\t\t document.getElementById('%sbut%s').style.border = 'inset black';\n"%(nc,i))
                    for sp in range(self.cube.iso_split[nc][i]):
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s').setAttribute('transparency', '1');\n"%(nc,i,sp))
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s_shape').setAttribute('ispickable', 'false');\n"%(nc,i,sp))
                    self.file_html.write("\t\t } \n\t\t }\n\t </script>\n")

    def func_galaxies(self):
        """
        Make function to hide/show galaxies.
        The X3D file must have galaxies for this to work.

        Parameters
        ----------
        gals : dictionary
            Same dictionary as the one used in make_galaxies().

        """
        for i,gal in enumerate(self.cube.galaxies):
            if i == 0:
                self.file_html.write("\t \t <script>\n\t \t function setgals()\n\t \t {\n\t \t if(document.getElementById('cube__%s_cross').getAttribute('transparency')!= '0'){\n"%gal)
            self.file_html.write("\t \t document.getElementById('cube__%s_cross').setAttribute('transparency', '0');\n"%gal)
            self.file_html.write("\t \t document.getElementById('cube__%s').setAttribute('transparency', '0');\n"%gal)
        self.file_html.write("\t \t }\n\t \t else {\n")
        for i,gal in enumerate(self.cube.galaxies):
            self.file_html.write("\t \t document.getElementById('cube__%s_cross').setAttribute('transparency', '1');\n"%gal)
            self.file_html.write("\t \t document.getElementById('cube__%s').setAttribute('transparency', '1');\n"%gal)
        self.file_html.write("\t\t }\n\t\t }\n\t\t </script>\n")
    
    def func_gallab(self):
        """
        Make function to hide/show galaxy labels. If this is used, the buttons()
        function with gallabs=True should also be used to create the buttons.
        The X3D file must have galaxy labels for this to work.

        """
        for i,gal in enumerate(self.cube.galaxies):
            if i == 0:
                self.file_html.write("\t\t <script>\n\t \t function setgallabels()\n\t \t {\n\t \t if(document.getElementById('cube__label_%s').getAttribute('transparency')!= '0'){\n"%gal)
            self.file_html.write("\t \t document.getElementById('cube__label_%s').setAttribute('transparency', '0');\n"%gal)
        self.file_html.write("\t \t }\n\t \t else {\n")
        for i,gal in enumerate(self.cube.galaxies):
            self.file_html.write("\t \t document.getElementById('cube__label_%s').setAttribute('transparency', '1');\n"%gal)
        self.file_html.write("\t\t }\n\t\t }\n\t\t </script>\n")
        
    def func_grids(self):
        self.file_html.write(tabs(2)+"<script>\n\t\tfunction setgrids()\n\t\t{\n")
        self.file_html.write(tabs(3)+"if(document.getElementById('cube__ticklines').getAttribute('transparency') == '0') {\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__ticklines').setAttribute('transparency', '1');\n")
        self.file_html.write(tabs(3)+"} else if (document.getElementById('cube__outline').getAttribute('transparency') == '0') {\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__outline').setAttribute('transparency', '1');\n")
        self.file_html.write(tabs(3)+"} else {\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__ticklines').setAttribute('transparency', '0');\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__outline').setAttribute('transparency', '0');\n")
        self.file_html.write(tabs(3)+"}\n")
        self.file_html.write(tabs(2)+"}\n\t\t </script>\n")
            
    def func_axes(self):
        """
        axes : string
            A string indicating what axis labels to include. Can be 'real' for the equatorial
            coordinates; 'diff' for the difference from the center of the cube or
            'both' for both options. Leave None for no axis labels.
            Must be the same as in make_labels(). The default is None.
        """

        self.file_html.write(tabs(2)+"<script>\n")
        self.file_html.write(tabs(2)+"function setaxes()\n")
        self.file_html.write(tabs(2)+"{\n")
        self.file_html.write(tabs(2)+"if(document.getElementById('cube__axlab_diff1').getAttribute('transparency') == '0') {\n")
        self.file_html.write(tabs(3)+"for (i=0; i<12; i++) {\n")
        self.file_html.write(tabs(3)+"if (i<6) {\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__axlab_diff'+i).setAttribute('transparency', '1');\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__axlab_real'+i).setAttribute('transparency', '0');\n")
        self.file_html.write(tabs(3)+"}\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__axtick_diff'+i).setAttribute('transparency', '1');\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__axtick_real'+i).setAttribute('transparency', '0');\n")
        self.file_html.write(tabs(3)+"}\n")
        self.file_html.write(tabs(2)+"}\n")
        self.file_html.write(tabs(2)+"else if (document.getElementById('cube__axlab_real1').getAttribute('transparency') == '0') {\n")
        self.file_html.write(tabs(3)+"for (i=0; i<12; i++) {\n")
        self.file_html.write(tabs(3)+"if (i<6) {\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__axlab_real'+i).setAttribute('transparency', '1');\n")
        self.file_html.write(tabs(3)+"}\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__axtick_real'+i).setAttribute('transparency', '1');\n")
        self.file_html.write(tabs(3)+"}\n")
        self.file_html.write(tabs(2)+"}\n")
        self.file_html.write(tabs(2)+"else {\n")
        self.file_html.write(tabs(3)+"for (i=0; i<12; i++) {\n")
        self.file_html.write(tabs(3)+"if (i<6) {\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__axlab_diff'+i).setAttribute('transparency', '0');\n")
        self.file_html.write(tabs(3)+"}\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__axtick_diff'+i).setAttribute('transparency', '0');\n")
        self.file_html.write(tabs(3)+"}\n")                
        self.file_html.write("\t\t }\n\t\t }\n\t\t </script>\n")
        
    def func_pick(self):
        """
        Allows picking the coordinates by clicking in the figure.
        WORKS WITH VIEWPOINT NOT WITH ORTHOVIEWPOINT.
        NOT FINISHED, DON'T USE.

        """
        self.file_html.write(roundto) #premade string with function to round to two decimals
        self.file_html.write(tabs(1)+"<script>\n")
        self.file_html.write(tabs(3)+"const picksca = document.querySelector('#scalev');\n")
        self.file_html.write(tabs(2)+"function handleClick(event) {\n")
        self.file_html.write(tabs(3)+"const sca = picksca.value;\n")
        self.file_html.write(tabs(3)+"var coordinates = event.hitPnt;\n")
        self.file_html.write(tabs(3)+"$('#coordX').html(roundTo(coordinates[0], 2)+' %s');\n"%self.cube.units[1])
        self.file_html.write(tabs(3)+"$('#coordY').html(roundTo(coordinates[1], 2)+' %s');\n"%self.cube.units[2])
        self.file_html.write(tabs(3)+"$('#coordZ').html(roundTo(coordinates[2], 2)/sca+' %s');\n"%self.cube.units[3])
        self.file_html.write(tabs(2)+"}\n\t </script>\n")

    def func_animation(self):
        """
        Function to start/stop the animation of the X3D models.
        """
        self.file_html.write('\n'+tabs(1)+"<script>")
        self.file_html.write('\n'+tabs(2)+"function animation() {")
        self.file_html.write('\n'+tabs(3)+"if (document.getElementById('cube__time').getAttribute('elapsedTime') == '0') {")
        self.file_html.write('\n'+tabs(4)+"document.getElementById('cube__time').setAttribute('startTime', document.getElementById('cube__time').getAttribute('time'));")
        self.file_html.write('\n'+tabs(4)+"document.getElementById('cube__time').setAttribute('isPaused', 'false');")
        self.file_html.write('\n'+tabs(3)+"} else if (document.getElementById('cube__time').getAttribute('isPaused') == 'false') {")
        self.file_html.write('\n'+tabs(4)+"document.getElementById('cube__time').setAttribute('loop', 'false');")
        self.file_html.write('\n'+tabs(4)+"document.getElementById('cube__time').setAttribute('isPaused', 'true')")
        self.file_html.write('\n'+tabs(3)+"} else {")
        self.file_html.write('\n'+tabs(4)+"document.getElementById('cube__time').setAttribute('loop', 'true');")
        self.file_html.write('\n'+tabs(4)+"document.getElementById('cube__time').setAttribute('isPaused', 'false');")
        self.file_html.write('\n'+tabs(3)+"}\n"+tabs(2)+'}\n'+tabs(1)+'</script>\n')

    def start_x3d(self):
        """
        Start the X3D part of the HTML. Must go before viewpoints() and close_x3d().

        """
        self.file_html.write(tabs(1)+"<center><x3d id='cubeFixed'>\n")

    def close_x3d(self, filename):
        """
        Insert the X3D file and close the X3D part of the HTML.
        Must go after viewpoints() and start_x3d().

        Parameters
        ----------
        x3dname : string
            Name of the X3D file to be inserted.

        """
        filename = filename.split('.')[0]+'.x3d'
        # if self.hclick:
        #     self.file_html.write(tabs(3)+'<inline url="%s" nameSpaceName="cube" mapDEFToID="true" onclick="handleClick(event)" onload="loading()"/>\n'%x3dname)
        # else:
        self.file_html.write(tabs(3)+f'<inline url="{filename}" nameSpaceName="cube" mapDEFToID="true" onclick="" onload="loading()"/>\n')
        self.file_html.write(tabs(2)+"</scene>\n\t</x3d></center>\n")
        
    def viewpoints(self):
        """
        Define viewpoints for the X3D figure. Must go after start_x3d() and
        before close_x3d(). Mandatory if the X3D does not have a Viewpoint,
        which is the case with the one created with this module.

        Parameters
        ----------
        maxcoord : len 3 array
            Maximum values of RA, DEC and third axis, in that order, for the difference to the center.

        """
        self.file_html.write("\t\t <scene>\n")
        #correct camera postition and FoV, not to clip (hide) the figure
        self.file_html.write(tabs(3)+"<OrthoViewpoint id=\"front\" bind='false' centerOfRotation='0,0,0' description='RA-Dec view' fieldOfView='[%s,%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='0,1,0,3.141593' position='0,0,-%s' zFar='10000' zNear='0.0001' ></OrthoViewpoint>\n"%(-1000*1.4,-1000*1.4,1000*1.4,1000*1.4,1000*1.4))
        self.file_html.write("\t\t\t <OrthoViewpoint id=\"side\" bind='false' centerOfRotation='0,0,0' description='Z - Dec view' fieldOfView='[%s,%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='0,-1,0,1.570796' position='-%s,0,0' zFar='10000' zNear='0.0001' ></OrthoViewpoint>\n"%(-1000*1.4,-1000*1.4,1000*1.4,1000*1.4,1000*1.4))
        self.file_html.write("\t\t\t <OrthoViewpoint id=\"side2\" bind='false' centerOfRotation='0,0,0' description='Z - RA view' fieldOfView='[%s,%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='1,1,1,4.1888' position='0,%s,0' zFar='10000' zNear='0.0001' ></OrthoViewpoint>\n"%(-1000*1.4,-1000*1.4,1000*1.4,1000*1.4,1000*1.4))

    def func_background(self):
        """
        Function to change the background color of the X3D figure.
        Must be after buttons()
        """
        self.file_html.write(tabs(3)+"<script>\n")
        self.file_html.write(tabs(4)+"function hex2Rgb(hex) {\n")
        self.file_html.write(tabs(5)+"var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);\n")
        self.file_html.write(tabs(5)+"var r = parseInt(result[1], 16)/255.;\n")
        self.file_html.write(tabs(5)+"var g = parseInt(result[2], 16)/255.;\n")
        self.file_html.write(tabs(5)+"var b = parseInt(result[3], 16)/255.;\n")
        self.file_html.write(tabs(5)+"return r.toString()+' '+g.toString()+' '+b.toString()\n")
        self.file_html.write(tabs(4)+"}\n")
        self.file_html.write(tabs(4)+"const background = document.querySelector('#back-choice');\n")
        self.file_html.write(tabs(4)+"background.addEventListener('change', change_background());\n")
        self.file_html.write(tabs(4)+"function change_background() {\n")
        self.file_html.write(tabs(5)+"const backCol = background.value; \n")
        self.file_html.write(tabs(5)+"document.getElementById('cube__back').setAttribute('skyColor', hex2Rgb(backCol));\n")
        self.file_html.write(tabs(4)+"}\n")
        self.file_html.write(tabs(3)+"</script>\n")

    # TUBESTUBESTUBES

    def func_tubes(self):
        """
        Function to create tubes for the X3D model interactively in the web page.
        Must be after buttons(). It is assumed that scalev is used. It will fail otherwise. 
        To use without scalev, remove(modify parts with sca).
        """
        self.file_html.write(tabs(2) + '\n')
        self.file_html.write(tabs(2) + '<div id="divmaster" style="margin-left: 2%">\n')
        self.file_html.write(tabs(3) + '<br>\n')
        self.file_html.write(tabs(3) + '<!-- BUTTON TO CHANGE LAYOUT -->\n')
        self.file_html.write(tabs(3) + '<label for="markers-choice"><b>Markers:</b> </label>\n')
        self.file_html.write(tabs(3) + '<select id="markers-choice">\n')
        self.file_html.write(tabs(4) + '<option value="none">None</option>\n')
        self.file_html.write(tabs(4) + '<option value="sphere">Sphere</option>\n')
        self.file_html.write(tabs(4) + '<option value="box">Box</option>\n')
        self.file_html.write(tabs(4) + '<option value="tube">Tube</option>\n')
        self.file_html.write(tabs(4) + '<option value="cone">Cone</option>\n')
        self.file_html.write(tabs(3) + '</select>\n')
        self.file_html.write(tabs(3) + '<input type="color" id="butcol" value="#ff0000">\n')
        self.file_html.write(tabs(3) + '<button id="butcreate" onclick="createmarker()">Create</button>\n')
        self.file_html.write(tabs(3) + '<button id="butremove" onclick="removemarker()">Remove</button> <br><br>\n')
        self.file_html.write(tabs(2) + '</div>\n')
        self.file_html.write(tabs(2) + '<!-- create various layouts for different objects -->\n')
        self.file_html.write(tabs(2) + '<div id="spherediv" style="display:none ; margin-left: 2%">\n')
        self.file_html.write(tabs(3) + '<input type="number" id="sphX0" step="1" placeholder="RA">\n')
        self.file_html.write(tabs(3) + '<input type="number" id="sphY0" step="1" placeholder="Dec">\n')
        self.file_html.write(tabs(3) + '<input type="number" id="sphZ0" step="1" placeholder="V">\n')
        self.file_html.write(tabs(3) + '<input type="number" id="sphrad0" min="0" max="50" step="0.5" placeholder="Radius">\n')
        self.file_html.write(tabs(2) + '</div>\n')
        self.file_html.write(tabs(2) + '<div id="boxdiv" style="display:none ; margin-left: 2%">\n')
        self.file_html.write(tabs(3) + '<input type="number" id="boxX0" step="1" placeholder="RA">\n')
        self.file_html.write(tabs(3) + '<input type="number" id="boxY0" step="1" placeholder="Dec">\n')
        self.file_html.write(tabs(3) + '<input type="number" id="boxZ0" step="1" placeholder="V">\n')
        self.file_html.write(tabs(3) + '<input type="text" id="boxrad0" placeholder="shape, e.g. \'20 20 20\'">\n')
        self.file_html.write(tabs(2) + '</div>\n')
        self.file_html.write(tabs(2) + '<div id="tubediv" style="display:none ; margin-left: 2% ; width: 80%">\n')
        self.file_html.write(tabs(3) + '<br>\n')
        self.file_html.write(tabs(3) + '<button id="newtube" onclick="newtube()">New</button>\n')
        self.file_html.write(tabs(3) + '<button id="addpoint" onclick="addpoint()">Add point</button>\n')
        self.file_html.write(tabs(3) + '<select id="new-tubes">\n')
        self.file_html.write(tabs(4) + '<option value="none">None</option>\n')
        self.file_html.write(tabs(3) + '</select>\n')
        self.file_html.write(tabs(3) + '<br>\n')
        self.file_html.write(tabs(2) + '</div>\n')

        self.file_html.write(tabs(2)+'<script>\n')
        self.file_html.write(tabs(3) + 'var ngeo = 0; //number of new geometrical shapes\n')
        self.file_html.write(tabs(3) + 'var tubelen = []; //number of cylinders in each tube\n')
        self.file_html.write(tabs(3) + 'var npoints = 0; // number of points of a tube (number of cylinders -1)(is changed for each tube)\n')
        self.file_html.write(tabs(3) + 'var tpars = []; // parameters needed in changescalev() for tubes\n')
        self.file_html.write(tabs(3) + 'const marktype = document.querySelector(\'#markers-choice\');\n')
        self.file_html.write(tabs(3) + 'marktype.addEventListener(\'change\', newlayout);\n')
        self.file_html.write(tabs(3) + 'const selTube = document.querySelector(\'#new-tubes\');\n')
        self.file_html.write(tabs(3) + 'selTube.addEventListener(\'change\', changeTube)\n')
        self.file_html.write(tabs(3) + 'const sscasv = document.querySelector(\'#scalev\');\n')
        self.file_html.write(tabs(3) + 'const col = document.querySelector(\'#butcol\');\n')

        self.file_html.write(tabs(3) + 'function addpoint() {\n')
        self.file_html.write(tabs(4) + 'const tubeIndex = selTube.value.split(\'\');\n')
        self.file_html.write(tabs(4) + 'const tubeInd = tubeIndex[tubeIndex.length-1];\n')
        self.file_html.write(tabs(4) + 'npoints = tubelen[tubeInd] + 2; // add 2 to add one point more. tubelen gives the number of cylinders.\n')
        self.file_html.write(tabs(4) + 'const newra = document.createElement(\'input\');\n')
        self.file_html.write(tabs(4) + 'newra.setAttribute(\'type\', \'number\');\n')
        self.file_html.write(tabs(4) + 'newra.setAttribute(\'id\', \'tub\'+tubeInd+\'X\'+(npoints-1));\n')
        self.file_html.write(tabs(4) + 'newra.setAttribute(\'placeholder\', \'RA\');\n')
        self.file_html.write(tabs(4) + 'newra.setAttribute(\'step\', \'1\');\n')
        self.file_html.write(tabs(4) + '\n')
        self.file_html.write(tabs(4) + 'const newdec = document.createElement(\'input\');\n')
        self.file_html.write(tabs(4) + 'newdec.setAttribute(\'type\', \'number\');\n')
        self.file_html.write(tabs(4) + 'newdec.setAttribute(\'id\', \'tub\'+tubeInd+\'Y\'+' '(npoints-1));\n')
        self.file_html.write(tabs(4) + 'newdec.setAttribute(\'placeholder\', \'Dec\');\n')
        self.file_html.write(tabs(4) + 'newdec.setAttribute(\'step\', \'1\');\n')
        self.file_html.write(tabs(4) + '\n')
        self.file_html.write(tabs(4) + 'const newv = document.createElement(\'input\');\n')
        self.file_html.write(tabs(4) + 'newv.setAttribute(\'type\', \'number\');\n')
        self.file_html.write(tabs(4) + 'newv.setAttribute(\'id\', \'tub\'+tubeInd+\'Z\'+' '(npoints-1));\n')
        self.file_html.write(tabs(4) + 'newv.setAttribute(\'placeholder\', \'V\');\n')
        self.file_html.write(tabs(4) + 'newv.setAttribute(\'step\', \'1\');\n')
        self.file_html.write(tabs(4) + 'const newline = document.createElement("br");\n')
        self.file_html.write(tabs(4) + 'document.getElementById(\'div\'+selTube.value).appendChild(newline);\n')
        self.file_html.write(tabs(4) + 'document.getElementById(\'div\'+selTube.value).appendChild(newra);\n')
        self.file_html.write(tabs(4) + 'document.getElementById(\'div\'+selTube.value).appendChild(newdec);\n')
        self.file_html.write(tabs(4) + 'document.getElementById(\'div\'+selTube.value).appendChild(newv);\n')
        self.file_html.write(tabs(4) + '\n')
        self.file_html.write(tabs(4) + 'tubelen[tubeInd] = npoints - 1;\n')
        self.file_html.write(tabs(3) + '}\n')
        self.file_html.write(tabs(4) + '\n')

        self.file_html.write(tabs(3) + 'function newtube() {\n')
        self.file_html.write(tabs(4) + 'tubelen.push(1);\n')
        self.file_html.write(tabs(4) + 'const l = tubelen.length-1;\n')
        self.file_html.write(tabs(4) + '\n')
        self.file_html.write(tabs(4) + 'if (selTube.value != \'none\') {\n')
        self.file_html.write(tabs(5) + 'document.getElementById(\'div\'+selTube.value).style.display = \'none\'\n')
        self.file_html.write(tabs(5) + 'const len = document.getElementById("new-tubes").length\n')
        self.file_html.write(tabs(5) + 'document.getElementById("new-tubes")[len] = new Option(\'tube\'+l, \'tube\'+l, true, true);\n')
        self.file_html.write(tabs(4) + '} else {\n')
        self.file_html.write(tabs(5) + 'document.getElementById("new-tubes")[0] = new Option(\'tube\'+l, \'tube\'+l, true, true); //this value is selected in the input button\n')
        self.file_html.write(tabs(4) + '}\n')
        self.file_html.write(tabs(4) + '// move tubediv0 here. use new button to create new div, no coordinate inputs from start\n')
        self.file_html.write(tabs(4) + 'const newdiv = document.createElement(\'div\');\n')
        self.file_html.write(tabs(4) + 'document.getElementById(\'tubediv\').appendChild(newdiv);\n')
        self.file_html.write(tabs(4) + 'newdiv.setAttribute(\'id\', \'divtube\'+l);\n')
        self.file_html.write(tabs(4) + 'newdiv.setAttribute(\'style\', \'margin-left: 2%\');\n')
        self.file_html.write(tabs(4) + 'newdiv.appendChild(document.createElement("br"));\n')
        self.file_html.write(tabs(4) + '\n')
        self.file_html.write(tabs(4) + '// create different radius for each tube\n')
        self.file_html.write(tabs(4) + '//<input type="number" id="tubrad" min="0" max="50" step="0.5" placeholder="Radius">\n')
        self.file_html.write(tabs(4) + 'const newrad = document.createElement(\'input\');\n')
        self.file_html.write(tabs(4) + 'newrad.setAttribute(\'type\', \'number\');\n')
        self.file_html.write(tabs(4) + 'newrad.setAttribute(\'id\', \'tubrad\'+l);\n')
        self.file_html.write(tabs(4) + 'newrad.setAttribute(\'min\', \'0\');\n')
        self.file_html.write(tabs(4) + 'newrad.setAttribute(\'max\', \'50\');\n')
        self.file_html.write(tabs(4) + 'newrad.setAttribute(\'step\', \'0.5\');\n')
        self.file_html.write(tabs(4) + 'newrad.setAttribute(\'placeholder\', \'Radius\');\n')
        self.file_html.write(tabs(4) + 'newdiv.appendChild(newrad);\n')
        self.file_html.write(tabs(4) + '\n')
        self.file_html.write(tabs(4) + 'for (i=0 ; i<2 ; i++) {\n')
        self.file_html.write(tabs(5) + 'const newra = document.createElement(\'input\');\n')
        self.file_html.write(tabs(5) + 'newra.setAttribute(\'type\', \'number\');\n')
        self.file_html.write(tabs(5) + 'newra.setAttribute(\'id\', \'tub\'+l+\'X\'+i);\n')
        self.file_html.write(tabs(5) + 'newra.setAttribute(\'placeholder\', \'RA\');\n')
        self.file_html.write(tabs(5) + 'newra.setAttribute(\'step\', \'1\');\n')
        self.file_html.write(tabs(5) + '\n')
        self.file_html.write(tabs(5) + 'const newdec = document.createElement(\'input\');\n')
        self.file_html.write(tabs(5) + 'newdec.setAttribute(\'type\', \'number\');\n')
        self.file_html.write(tabs(5) + 'newdec.setAttribute(\'id\', \'tub\'+l+\'Y\'+i);\n')
        self.file_html.write(tabs(5) + 'newdec.setAttribute(\'placeholder\', \'Dec\');\n')
        self.file_html.write(tabs(5) + 'newdec.setAttribute(\'step\', \'1\');\n')
        self.file_html.write(tabs(5) + '\n')
        self.file_html.write(tabs(5) + 'const newv = document.createElement(\'input\');\n')
        self.file_html.write(tabs(5) + 'newv.setAttribute(\'type\', \'number\');\n')
        self.file_html.write(tabs(5) + 'newv.setAttribute(\'id\', \'tub\'+l+\'Z\'+i);\n')
        self.file_html.write(tabs(5) + 'newv.setAttribute(\'placeholder\', \'V\');\n')
        self.file_html.write(tabs(5) + 'newv.setAttribute(\'step\', \'1\');\n')
        self.file_html.write(tabs(5) + '\n')
        self.file_html.write(tabs(5) + 'const newline = document.createElement("br");\n')
        self.file_html.write(tabs(5) + 'newdiv.appendChild(newline);\n')
        self.file_html.write(tabs(5) + 'newdiv.appendChild(newra);\n')
        self.file_html.write(tabs(5) + 'newdiv.appendChild(newdec);\n')
        self.file_html.write(tabs(5) + 'newdiv.appendChild(newv);\n')
        self.file_html.write(tabs(4) + '}\n')
        self.file_html.write(tabs(3) + '}\n')

        self.file_html.write(tabs(3) + 'function newlayout() {\n')
        self.file_html.write(tabs(4) + 'document.getElementById(\'spherediv\').style.display = \'none\'\n')
        self.file_html.write(tabs(4) + 'document.getElementById(\'boxdiv\').style.display = \'none\'\n')
        self.file_html.write(tabs(4) + 'document.getElementById(\'tubediv\').style.display = \'none\'\n')
        self.file_html.write(tabs(4) + 'name = marktype.value+\'div\'\n')
        self.file_html.write(tabs(4) + 'document.getElementById(name).style.display = \'inline-block\'\n')
        self.file_html.write(tabs(3) + '}\n')


        self.file_html.write(tabs(3) + 'function createmarker() {\n')
        self.file_html.write(tabs(4) + 'const sca = inpscasv.value;\n')
        self.file_html.write(tabs(4) + 'const tubeIndex = selTube.value.split(\'\');\n')
        self.file_html.write(tabs(4) + 'const tubeInd = tubeIndex[tubeIndex.length-1];\n')
        self.file_html.write(tabs(4) + 'const tpars_s = [];\n')
        self.file_html.write(tabs(4) + 'npoints = tubelen[tubeInd] + 1\n')
        self.file_html.write(tabs(4) + 'if (document.getElementById(tubeInd+\'newtra0\') == null) {\n')
        self.file_html.write(tabs(5) + 'for (i=0; i<npoints-1; i++) {\n')
        self.file_html.write(tabs(6) + 'const x0 = Number(document.querySelector(\'#tub\'+tubeInd+\'X\'+i).value);\n')
        self.file_html.write(tabs(6) + 'const y0 = Number(document.querySelector(\'#tub\'+tubeInd+\'Y\'+i).value);\n')
        self.file_html.write(tabs(6) + 'const z0 = Number(document.querySelector(\'#tub\'+tubeInd+\'Z\'+i).value);\n')
        self.file_html.write(tabs(6) + 'const x1 = Number(document.querySelector(\'#tub\'+tubeInd+\'X\'+(i+1)).value);\n')
        self.file_html.write(tabs(6) + 'const y1 = Number(document.querySelector(\'#tub\'+tubeInd+\'Y\'+(i+1)).value);\n')
        self.file_html.write(tabs(6) + 'const z1 = Number(document.querySelector(\'#tub\'+tubeInd+\'Z\'+(i+1)).value);\n')
        self.file_html.write(tabs(6) + 'const rad = document.querySelector(\'#tubrad\'+tubeInd);\n')
        self.file_html.write(tabs(6) + 'const trans = [(x0+x1)/2, (y0+y1)/2, (z0+z1)/2]\n')
        self.file_html.write(tabs(6) + 'const diff = [x1-x0, y1-y0, z1-z0]\n')
        self.file_html.write(tabs(6) + 'const height = Math.sqrt(diff[0]**2+diff[1]**2+(sca*diff[2])**2)*1.03;\n')
        self.file_html.write(tabs(6) + 'const angle = Math.acos(diff[1]/height);\n')
        self.file_html.write(tabs(6) + 'tpars_s.push([trans,diff]);\n')
        self.file_html.write(tabs(6) + 'const newtra = document.createElement(\'transform\');\n')
        self.file_html.write(tabs(6) + 'newtra.setAttribute(\'id\', tubeInd+\'newtra\'+i);\n')
        self.file_html.write(tabs(6) + 'const newshape = document.createElement(\'shape\');\n')
        self.file_html.write(tabs(6) + 'newshape.setAttribute(\'id\', tubeInd+\'newsha\'+i);\n')
        self.file_html.write(tabs(6) + 'const newape = document.createElement(\'appearance\');\n')
        self.file_html.write(tabs(6) + 'const newmat = document.createElement(\'material\');\n')
        self.file_html.write(tabs(6) + 'newmat.setAttribute(\'diffuseColor\', col.value);\n')
        self.file_html.write(tabs(6) + 'newmat.setAttribute(\'id\', tubeInd+\'mat\'+i);\n')
        self.file_html.write(tabs(6) + 'const newgeo = document.createElement(\'cylinder\');\n')
        self.file_html.write(tabs(6) + 'newgeo.setAttribute(\'id\', tubeInd+\'newcyl\'+i);\n')
        self.file_html.write(tabs(6) + 'newgeo.setAttribute(\'radius\', rad.value);\n')
        self.file_html.write(tabs(6) + 'newgeo.setAttribute(\'solid\', \'false\');\n')
        self.file_html.write(tabs(6) + 'newgeo.setAttribute(\'height\', height.toString());\n')
        self.file_html.write(tabs(6) + 'newtra.setAttribute(\'translation\', trans[0]+\' \'+trans[1]+\' \'+sca*trans[2]);\n')
        self.file_html.write(tabs(6) + 'newtra.setAttribute(\'rotation\', sca*diff[2]+\' 0 \'+(-diff[0])+\' \'+angle);\n')
        self.file_html.write(tabs(6) + 'newshape.appendChild(newape).appendChild(newmat);\n')
        self.file_html.write(tabs(6) + 'newtra.appendChild(newshape).appendChild(newgeo);\n')
        self.file_html.write(tabs(6) + 'document.getElementById(\'cube__ROOT\').appendChild(newtra);\n')
        self.file_html.write(tabs(5) + '}\n')
        self.file_html.write(tabs(5) + 'tpars.push(tpars_s);\n')
        self.file_html.write(tabs(4) + '}\n')
        self.file_html.write(tabs(4) + 'else {\n')
        self.file_html.write(tabs(5) + 'for (i=0; i<npoints-1; i++) {\n')
        self.file_html.write(tabs(6) + 'const x0 = Number(document.querySelector(\'#tub\'+tubeInd+\'X\'+i).value);\n')
        self.file_html.write(tabs(6) + 'const y0 = Number(document.querySelector(\'#tub\'+tubeInd+\'Y\'+i).value);\n')
        self.file_html.write(tabs(6) + 'const z0 = Number(document.querySelector(\'#tub\'+tubeInd+\'Z\'+i).value);\n')
        self.file_html.write(tabs(6) + 'const x1 = Number(document.querySelector(\'#tub\'+tubeInd+\'X\'+(i+1)).value);\n')
        self.file_html.write(tabs(6) + 'const y1 = Number(document.querySelector(\'#tub\'+tubeInd+\'Y\'+(i+1)).value);\n')
        self.file_html.write(tabs(6) + 'const z1 = Number(document.querySelector(\'#tub\'+tubeInd+\'Z\'+(i+1)).value);\n')
        self.file_html.write(tabs(6) + 'const rad = document.querySelector(\'#tubrad\'+tubeInd);\n')
        self.file_html.write(tabs(6) + 'const trans = [(x0+x1)/2, (y0+y1)/2, (z0+z1)/2]\n')
        self.file_html.write(tabs(6) + 'const diff = [x1-x0, y1-y0, z1-z0]\n')
        self.file_html.write(tabs(6) + 'const height = Math.sqrt(diff[0]**2+diff[1]**2+(sca*diff[2])**2)*1.03;\n')
        self.file_html.write(tabs(6) + 'const angle = Math.acos(diff[1]/height);\n')
        self.file_html.write(tabs(6) + 'tpars_s.push([trans,diff]);\n')
        self.file_html.write(tabs(6) + 'document.getElementById(tubeInd+\'newtra\'+i).setAttribute(\'rotation\', sca*diff[2]+\' 0 \'+(-diff[0])+\' \'+angle);\n')
        self.file_html.write(tabs(6) + 'document.getElementById(tubeInd+\'newtra\'+i).setAttribute(\'translation\', trans[0]+\' \'+trans[1]+\' \'+sca*trans[2]);\n')
        self.file_html.write(tabs(6) + 'document.getElementById(tubeInd+\'newcyl\'+i).setAttribute(\'height\', height.toString());\n')
        self.file_html.write(tabs(6) + 'document.getElementById(tubeInd+\'newcyl\'+i).setAttribute(\'radius\', rad.value);\n')
        self.file_html.write(tabs(5) + '}\n')
        self.file_html.write(tabs(5) + 'tpars[tubeInd] = tpars_s;\n')
        self.file_html.write(tabs(4) + '}\n')
        self.file_html.write(tabs(3) + '}\n')

        self.file_html.write(tabs(3) + 'function removemarker() {\n')
        self.file_html.write(tabs(4) + 'const tubeIndex = selTube.value.split(\'\');\n')
        self.file_html.write(tabs(4) + 'const tubeInd = tubeIndex[tubeIndex.length-1];\n')
        self.file_html.write(tabs(4) + 'npoints = tubelen[tubeInd] + 1\n')
        self.file_html.write(tabs(4) + 'if (document.getElementById("new-tubes").length === 1) {\n')
        self.file_html.write(tabs(5) + 'document.getElementById("new-tubes")[0] = new Option("None", "none", true, true);\n')
        self.file_html.write(tabs(4) + '} else {\n')
        self.file_html.write(tabs(5) + 'for (i=0; i<document.getElementById("new-tubes").length; i++) {\n')
        self.file_html.write(tabs(6) + 'if ("tube"+i === document.getElementById("new-tubes")[i].value) {\n')
        self.file_html.write(tabs(7) + 'document.getElementById("new-tubes")[i].remove();\n')
        self.file_html.write(tabs(7) + 'break;\n')
        self.file_html.write(tabs(6) + '}\n')
        self.file_html.write(tabs(5) + '}\n')
        self.file_html.write(tabs(5) + 'document.getElementById("new-tubes")[0].setAttribute(\'selected\', \'selected\');\n')
        self.file_html.write(tabs(5) + 'const nexttube = document.getElementById("new-tubes")[0].value;\n')
        self.file_html.write(tabs(5) + 'document.getElementById(\'div\'+nexttube).style.display = \'inline-block\';\n')
        self.file_html.write(tabs(4) + '}\n')
        self.file_html.write(tabs(4) + 'for (i=0; i<npoints-1; i++){\n')
        self.file_html.write(tabs(5) + 'document.getElementById(tubeInd+\'newtra\'+i).remove();\n')
        self.file_html.write(tabs(4) + '}\n')
        self.file_html.write(tabs(4) + 'document.getElementById(\'divtube\'+tubeInd).remove();\n')
        self.file_html.write(tabs(3) + '}\n')

        self.file_html.write(tabs(3) + 'function changeTube() {\n')
        self.file_html.write(tabs(4) + 'for (i=0; i<tubelen.length; i++) {\n')
        self.file_html.write(tabs(5) + 'if (\'tube\'+i != selTube.value) {\n')
        self.file_html.write(tabs(6) + 'if (document.getElementById(\'divtube\'+i) != null) {\n')
        self.file_html.write(tabs(7) + 'document.getElementById(\'divtube\'+i).style.display = \'none\';\n')
        self.file_html.write(tabs(6) + '}\n')
        self.file_html.write(tabs(5) + '}\n')
        self.file_html.write(tabs(5) + 'document.getElementById(\'div\'+selTube.value).style.display = \'inline-block\';\n')
        self.file_html.write(tabs(4) + '}\n')
        self.file_html.write(tabs(3) + '}\n')

        self.file_html.write(tabs(2)+'</script>\n')
        

    #TUBESTUBESTUBES

    def buttons(self, l_isolevels, l_colors=None, linelabs=False, centrot=False):
        """
        Makes the buttons to apply different functions in the web page.

        Parameters
        ----------
        l_isolevels : list of arrays
            A list of arrays with the values of each isosurface layer, one array for each cube. 
            E.g.[2,5,9] for three layers at values 2,5 and 9 of the cube.
            Should be in increasing order. (REAL?)
        l_colors : list of arrays
            RGB color for each isolevel a string ('122 233 20'). There must be a list of colors for each cube.
        colormaps : list of strings
            Matplotlib colormaps to introduce in the web page. Use plt.colormaps() for all.
        hide2d : bool
            Create button to hide and show the 2D image.
        scalev : bool
            Create button to change the scale of the spectral axis.
        move2d : bool
            Create button to move the 2D image along the spectral axis.
        linelabs : False or list of strings
            Strings with the labels to use for different cubes. If False, the labels will be 'Cube 0', 'Cube 1', etc.
        centrot : bool
            Create button to change the center of rotation.
        background : bool
            Create button to change the background color.
        """
        self.file_html.write(tabs(1)+'<div style="width:90%">\n')
        self.file_html.write(tabs(2)+'<br/>\n')
        # Viewpoint buttons
        self.file_html.write(tabs(2)+"&nbsp <b>Viewpoints:</b>\n")
        self.file_html.write(tabs(3)+"<button onclick=\"document.getElementById('cubeFixed').runtime.resetView();\">Reset View</button>\n")
        for i in range(3): # 3 w/o perspective, 4 with
            self.file_html.write(tabs(3)+"<button onclick=\"document.getElementById('%s').setAttribute('set_bind','true');\"> %s </button>\n"%(side[i], nam[i]))
        #unccoment next line for next view button
        #self.file_html.write('\t\t   <button onclick="document.getElementById(\'cubeFixed\').runtime.nextView();">Next View</button>\n')
        
        # Grids and ax labels
        self.file_html.write('\n'+tabs(2)+'&nbsp <b>Labels:</b>\n')
        self.file_html.write(tabs(3)+'<button onclick="setgrids();" >Grids</button>\n')
        self.file_html.write(tabs(3)+'<button onclick="setaxes();" >Axes labels</button>\n')

        if self.cube.galaxies is not None:
            self.file_html.write(tabs(3)+'<button onclick="setgals();" >Galaxies</button>\n')
            self.file_html.write(tabs(3)+'<button onclick="setgallabels();" >Galaxy Labels</button>\n')

        if self.cube.image2d is not None:
            self.file_html.write(tabs(3)+'<button onclick="setimage2d();" >2D image</button>\n')
        
        if centrot:
            self.file_html.write(tabs(2)+'&nbsp <label for="rotationCenter"><b>Center of Rotation</b> </label>\n')
            self.file_html.write(tabs(3)+'<select id="rotationCenter">\n')
            self.file_html.write(tabs(4)+'<option value="Origin">Origin</option>\n')
            for nc in range(len(l_isolevels)):
                if type(linelabs) == list:
                    self.file_html.write(tabs(4)+'<option value="op%s">%s</option>\n'%(nc,linelabs[nc]))
                else:
                    self.file_html.write(tabs(4)+'<option value="op%s">Center%s</option>\n'%(nc,nc))
            self.file_html.write(tabs(3)+'</select>\n')
            
        self.file_html.write(tabs(3)+'<button id="anim" onclick="animation()">Animation</button>')
        
        # Background
        self.file_html.write(tabs(3)+'&nbsp <label for="back-choice"><b>Background:</b> </label>\n')
        self.file_html.write(tabs(3)+'<input oninput="change_background()" id="back-choice" type="color" value="#999999">\n')

        if self.cube.galaxies is not None:
            self.file_html.write(tabs(3)+'&nbsp <b>Font Size:</b>\n')
            self.file_html.write(tabs(3)+'&nbsp <label for="back-choice">Galaxy: </label>\n')
            self.file_html.write(tabs(3)+'<input oninput="change_galsize()" id="galsize-choice" type="number" min="2" max="100" value="8", step="2">\n')
        
        nlayers = [len(l) for l in l_isolevels]
        numcubes = len(nlayers)

        for nc in range(numcubes):
            self.file_html.write(tabs(2)+'<br><br>\n')
            if type(linelabs) == list:
                self.file_html.write(tabs(2)+'&nbsp <b>%s (%s):</b>\n'%(linelabs[nc],self.cube.units[0]))
            else:
                self.file_html.write(tabs(2)+'&nbsp <b>Cube %s (%s):</b>\n'%(nc,self.cube.units[0]))
            for i in range(nlayers[nc]):
                ca = np.array(l_colors[nc][i].split(' ')).astype(float)*255
                c = 'rgb('+str(ca.astype(int))[1:-1]+')'
                if (ca[0]*0.299 + ca[1]*0.587 + ca[2]*0.114) > 130:
                    self.file_html.write(tabs(3)+'<button id="%sbut%s" onclick="setHI%slayer%s();" style="font-size:20px ; border:5px dashed black ; background:%s ; color:black"><b>%s</b></button>\n'%(nc,i,nc,i,c,np.round(l_isolevels[nc][i],1)))
                else:
                    self.file_html.write(tabs(3)+'<button id="%sbut%s" onclick="setHI%slayer%s();" style="font-size:20px ; border:5px dashed black ; background:%s ; color:white"><b>%s</b></button>\n'%(nc,i,nc,i,c,np.round(l_isolevels[nc][i],1)))
        self.file_html.write(tabs(3)+'&nbsp <button id="all" onclick="hideall()"><b>Invert</b></button>')
        self.file_html.write(tabs(2)+'<br><br>\n')

                    
        # to separate buttons in two parts
        #if self.grids or self.gals or self.gallabs or self.axes or self.hclick or colormaps is not None:
            #self.file_html.write('\n\t <div style="position:absolute;left:800px;top:140px;width:600px">\n')
        
        # Colormaps
        for nc in range(numcubes):
            #self.file_html.write('\t\t <br><br>\n')
            self.file_html.write(tabs(2)+'&nbsp <label for="cmaps-choice%s"><b>Cmap %s</b> </label>\n'%(nc,nc))
            self.file_html.write(tabs(2)+'<select id="cmaps-choice%s">\n'%nc)
            for c in default_cmaps:
                self.file_html.write(tabs(3)+'<option value="%s">%s</option>\n'%(c,c))
            self.file_html.write(tabs(2)+'</select>\n')
            self.file_html.write(tabs(2) + '<label for="cmaps-min%s"><b>Min %s:</b> </label>\n'%(nc,nc))
            self.file_html.write(tabs(2) + '<input id="cmaps-min%s" type="number" value="%s">\n'%(nc, np.min(l_isolevels[nc])))
            self.file_html.write(tabs(2) + '<label for="cmaps-max%s"><b>Max %s:</b> </label>\n'%(nc,nc))
            self.file_html.write(tabs(2) + '<input id="cmaps-max%s" type="number" value="%s">\n'%(nc, np.max(l_isolevels[nc])))
            self.file_html.write(tabs(2) + '<label for="cmaps-min%s"><b>Scale %s:</b> </label>\n'%(nc,nc))
            self.file_html.write(tabs(2) + '<select id="cmaps-scale%s">\n'%nc)
            self.file_html.write(tabs(3) + '<option value="linear" selected="selected">linear</option>\n')
            self.file_html.write(tabs(3) + '<option value="log">log</option>\n')
            self.file_html.write(tabs(3) + '<option value="sqrt">sqrt</option>\n')
            self.file_html.write(tabs(3) + '<option value="power">power</option>\n')
            self.file_html.write(tabs(3) + '<option value="asinh">asinh</option>\n')
            self.file_html.write(tabs(2) + '</select>\n')

        # SCALEV
        #self.file_html.write(tabs(2)+'<br><br>\n')
        self.file_html.write(tabs(2)+'&nbsp <label for="scalev"><b>Z scale:</b> </label>\n')
        self.file_html.write(tabs(2)+'<input oninput="changescalev()" id="scalev" type="range" list="marker" min="0" max="5" step="0.001" value="1"/>\n')
        self.file_html.write(tabs(2)+'<datalist id="marker">\n')
        self.file_html.write(tabs(3)+'<option value="1"></option>\n')
        self.file_html.write(tabs(2)+'</datalist>\n')
            
        if self.cube.image2d is not None:
            #self.file_html.write('\t\t <br><br>\n')
            self.file_html.write(tabs(2)+'&nbsp <label for="move2dimg"><b>2D image:</b> </label>\n')
            self.file_html.write(tabs(2)+'<input oninput="move2d()" id="move2dimg" type="range" min="-1" max="1" step="0.0001" value="1"/>\n')
            self.file_html.write(tabs(2)+'<b>$Z=$</b> <output id="showvalue"></output> %s\n'%self.cube.units[3])
            # display chosen velocity of bar too

        # PICKING
        # self.file_html.write(tabs(2)+'<br><br>\n')
        # self.file_html.write(tabs(2)+'&nbsp <label for="clickcoords"><b>Click coordinates:</b></label>\n')
        # self.file_html.write(tabs(2)+'<table id="clickcoords">\n\t\t <tbody>\n')
        # self.file_html.write(tabs(3)+'<tr><td>&nbsp RA: </td><td id="coordX">--</td></tr>\n')
        # self.file_html.write(tabs(3)+'<tr><td>&nbsp Dec: </td><td id="coordY">--</td></tr>\n')
        # self.file_html.write(tabs(3)+'<tr><td>&nbsp V: </td><td id="coordZ">--</td></tr>\n')
        # self.file_html.write(tabs(2)+'</tbody></table>\n')
            
        self.file_html.write(tabs(1)+'</div>\n')
        
    def func_move2dimage(self):
        """
        Function to move the 2D image along the spectral axis.

        Parameters
        ----------
        diff_vmax : float
            Maximum value of the spectral axis as a difference from the centre of the cube.
        real_vmax : float
            Maximum value of the spectral axis in the cube with respect to earth.
            If None it will not be shown in the interface. Default is None.
        """
        self.file_html.write(tabs(2)+"<script>\n")
        self.file_html.write(tabs(2)+"const inpscam2 = document.querySelector('#scalev');\n")
        self.file_html.write(tabs(2)+"const inpmovem2 = document.querySelector('#move2dimg');\n")
        self.file_html.write(tabs(2)+"const showval = document.querySelector('#showvalue');\n")
        self.file_html.write(tabs(2)+"function move2d()\n\t\t{\n")
        self.file_html.write(tabs(3)+"const sca = inpscam2.value;\n")
        self.file_html.write(tabs(3)+"const move = inpmovem2.value;\n")
        self.file_html.write(tabs(3)+f"showval.textContent = roundTo({self.cube.coords[2,1]}+(move-1)*1000, 3);\n")
        self.file_html.write(tabs(3)+"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(sca*move-1)*1000); }}\n")
        self.file_html.write(tabs(2)+"</script>\n")

    def func_galsize(self):
        """
        Function to change the size of the galaxy markers and their labels.

        Parameters
        ----------
        gals : dict
            Dictionary with the names of the galaxies as keys and their coordinates as values.
        """
        self.file_html.write(tabs(2)+"<script>\n")
        self.file_html.write(tabs(3)+"const galsize = document.querySelector('#galsize-choice');\n")
        self.file_html.write(tabs(3)+"galsize.addEventListener('change', change_galsize());\n")
        self.file_html.write(tabs(3)+"function change_galsize() {\n")
        self.file_html.write(tabs(4)+"let gals = %s;\n"%str(list(self.cube.galaxies.keys())))
        self.file_html.write(tabs(4)+"const gsval = galsize.value/8.;\n")
        self.file_html.write(tabs(4)+"for (const gal of gals) {\n")
        self.file_html.write(tabs(5)+"document.getElementById('cube__'+gal+'_sphere_tra').setAttribute('scale', gsval.toString()+' '+gsval.toString()+' '+gsval.toString());\n")
        self.file_html.write(tabs(5)+"document.getElementById('cube__'+gal+'_fs').setAttribute('size', gsval*8);\n")
        self.file_html.write(tabs(5)+"}\n")
        self.file_html.write(tabs(3)+"}\n")
        self.file_html.write(tabs(2)+"</script>\n")

    def func_setCenterOfRotation(self, centers):
        """
        Function to change the center of rotation.

        Parameters
        ----------
        centers : list
            List of strings with the coordinates (as different from the centre) of the centers of rotation to be added. E.g. ["0 10 0", "10 0 10"]
        """
        self.file_html.write(tabs(2)+"<script>\n")
        self.file_html.write(tabs(3)+"cor = document.querySelector('#rotationCenter');\n")
        self.file_html.write(tabs(3)+"cor.addEventListener('change', setCenterOfRotation);\n")
        self.file_html.write(tabs(2)+"function setCenterOfRotation() {\n")
        self.file_html.write(tabs(3)+"const cen = cor.value;\n")
        self.file_html.write(tabs(3)+"if (cen === 'Origin') {\n")
        self.file_html.write(tabs(4)+"document.getElementById('front').setAttribute('centerOfRotation','0 0 0');\n")
        self.file_html.write(tabs(4)+"document.getElementById('side').setAttribute('centerOfRotation','0 0 0');\n")
        self.file_html.write(tabs(4)+"document.getElementById('side2').setAttribute('centerOfRotation','0 0 0');\n")
        self.file_html.write(tabs(3)+"}\n")
        for nc in range(len(centers)):
            self.file_html.write(tabs(3)+"else if (cen === 'op%s') {\n"%nc)
            self.file_html.write(tabs(4)+"document.getElementById('front').setAttribute('centerOfRotation', '%s');\n"%centers[nc])
            self.file_html.write(tabs(4)+"document.getElementById('side').setAttribute('centerOfRotation', '%s');\n"%centers[nc])
            self.file_html.write(tabs(4)+"document.getElementById('side2').setAttribute('centerOfRotation', '%s');\n"%centers[nc])
            self.file_html.write(tabs(3)+"}\n")
        self.file_html.write(tabs(2)+"}\n"+tabs(2)+"</script>\n")
        
    def func_colormaps(self, l_isolevels):
        """
        Make function to change the colormap of the layers.
        Must be after buttons()

        Parameters
        ----------
        l_isolevels : list of arrays
            A list of arrays with the values of each isosurface layer, one array for each cube. 
            E.g.[2,5,9] for three layers at values 2,5 and 9 of the cube.
            Should be in increasing order. (REAL?)

        """
        self.file_html.write("\t\t <!--MUST BE BELOW THE <select> ELEMENT-->\n")
        numcubes = len(l_isolevels)
        for nc in range(numcubes):
            self.file_html.write(tabs(2)+"<script>\n")
            self.file_html.write(tabs(3)+"const cc%s = document.querySelector('#cmaps-choice%s');\n"%(nc,nc))
            self.file_html.write(tabs(3)+"cc%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.file_html.write(tabs(3)+"const cmapmin%s = document.querySelector('#cmaps-min%s');\n"%(nc,nc))
            self.file_html.write(tabs(3)+"cmapmin%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.file_html.write(tabs(3)+"const cmapmax%s = document.querySelector('#cmaps-max%s');\n"%(nc,nc))
            self.file_html.write(tabs(3)+"cmapmax%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.file_html.write(tabs(3)+"const cmapscale%s = document.querySelector('#cmaps-scale%s');\n"%(nc,nc))
            self.file_html.write(tabs(3)+"cmapscale%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.file_html.write(tabs(3)+"const isolevels%s = %s;\n"%(nc,repr(l_isolevels[nc]).replace('array(','').replace(')','')))

            self.file_html.write(tabs(3)+"function change_colormap%s() {\n"%nc)
            self.file_html.write(tabs(4)+"var cmap%s = cc%s.value;\n"%(nc,nc))
            self.file_html.write(tabs(4)+"var min%s = cmapmin%s.value;\n"%(nc,nc))
            self.file_html.write(tabs(4)+"var max%s = cmapmax%s.value;\n"%(nc,nc))
            self.file_html.write(tabs(4)+"const scale%s = cmapscale%s.value;\n"%(nc,nc))
            self.file_html.write(tabs(4)+"var reverse%s = false;\n"%(nc))
            self.file_html.write(tabs(4)+"var collevs%s = [];\n"%(nc))

            self.file_html.write(tabs(4)+"if (scale%s === 'linear') {\n"%(nc))
            self.file_html.write(tabs(5)+"for (const level of isolevels%s) {\n"%(nc))
            self.file_html.write(tabs(6)+"collevs%s.push(level);\n"%(nc))
            self.file_html.write(tabs(5)+"}\n")
            self.file_html.write(tabs(4)+"}\n")
            self.file_html.write(tabs(4)+"else if (scale%s === 'log') {\n"%(nc))
            self.file_html.write(tabs(5)+"min%s = Math.log(min%s);\n"%(nc,nc))
            self.file_html.write(tabs(5)+"max%s = Math.log(max%s);\n"%(nc,nc))
            self.file_html.write(tabs(5)+"for (const level of isolevels%s) {\n"%(nc))
            self.file_html.write(tabs(6)+"collevs%s.push(Math.log(level));\n"%nc)
            self.file_html.write(tabs(5)+"}\n")
            self.file_html.write(tabs(4)+"}\n")
            self.file_html.write(tabs(4)+"else if (scale%s === 'sqrt') {\n"%(nc))
            self.file_html.write(tabs(5)+"min%s = Math.sqrt(min%s);\n"%(nc,nc))
            self.file_html.write(tabs(5)+"max%s = Math.sqrt(max%s);\n"%(nc,nc))
            self.file_html.write(tabs(5)+"for (const level of isolevels%s) {\n"%(nc))
            self.file_html.write(tabs(6)+"collevs%s.push(Math.sqrt(level));\n"%(nc))
            self.file_html.write(tabs(5)+"}\n")
            self.file_html.write(tabs(4)+"}\n")
            self.file_html.write(tabs(4)+"else if (scale%s === 'power') {\n"%nc)
            self.file_html.write(tabs(5)+"min%s = Math.pow(min%s, 2);\n"%(nc,nc))
            self.file_html.write(tabs(5)+"max%s = Math.pow(max%s, 2);\n"%(nc,nc))
            self.file_html.write(tabs(5)+"for (const level of isolevels%s) {\n"%nc)
            self.file_html.write(tabs(6)+"collevs%s.push(Math.pow(level, 2));\n"%nc)
            self.file_html.write(tabs(5)+"}\n")
            self.file_html.write(tabs(4)+"}\n")
            self.file_html.write(tabs(4)+"else if (scale%s === 'asinh') {\n"%nc)
            self.file_html.write(tabs(5)+"min%s = Math.asinh(min%s);\n"%(nc,nc))
            self.file_html.write(tabs(5)+"max%s = Math.asinh(max%s);\n"%(nc,nc))
            self.file_html.write(tabs(5)+"for (const level of isolevels%s) {\n"%nc)
            self.file_html.write(tabs(6)+"collevs%s.push(Math.asinh(level));\n"%nc)
            self.file_html.write(tabs(5)+"}\n")
            self.file_html.write(tabs(4)+"}\n")
            self.file_html.write(tabs(4)+"for (let lev = 0; lev < isolevels%s.length; lev++) {\n"%nc)
            self.file_html.write(tabs(5)+"if (min%s >= collevs%s[lev]) {\n"%(nc,nc))
            self.file_html.write(tabs(6)+"collevs%s[lev] = 0;\n"%nc)
            self.file_html.write(tabs(5)+"} else if (max%s <= collevs%s[lev]) {\n"%(nc,nc))
            self.file_html.write(tabs(6)+"collevs%s[lev] = 1;\n"%nc)
            self.file_html.write(tabs(5)+"} else {\n")
            self.file_html.write(tabs(6)+"collevs%s[lev] = (collevs%s[lev] - min%s) / (max%s - min%s);\n"%(nc,nc,nc,nc,nc))
            self.file_html.write(tabs(5)+"}\n")
            self.file_html.write(tabs(5)+"if (cmap%s.endsWith('_r')) {\n"%nc)
            self.file_html.write(tabs(6)+"reverse%s = true\n"%nc)
            self.file_html.write(tabs(5)+"}\n")
            self.file_html.write(tabs(5)+"if (cmap%s.includes('gist_rainbow')) {\n"%nc)
            self.file_html.write(tabs(6)+"var color%s = evaluate_cmap(collevs%s[lev], 'gist_rainbow', reverse%s);\n"%(nc,nc,nc))
            self.file_html.write(tabs(5)+"} else {\n")
            self.file_html.write(tabs(6)+"var color%s = evaluate_cmap(collevs%s[lev], cmap%s.replace('_r', ''), reverse%s);\n"%(nc,nc,nc,nc))
            self.file_html.write(tabs(5)+"}\n")
            self.file_html.write(tabs(5)+"var split%s = %s;\n"%(nc,repr(self.cube.iso_split[nc]).replace('array(','').replace(')','')))
            self.file_html.write(tabs(5)+"for (let sp = 0; sp < split%s[lev]; sp++) {\n"%nc)
            self.file_html.write(tabs(6)+"document.getElementById('cube__%slayer'+lev+'_sp'+sp).setAttribute('diffuseColor', color%s[0]/255+' '+color%s[1]/255+' '+color%s[2]/255);\n"%(nc,nc,nc,nc))
            self.file_html.write(tabs(5)+"}\n")
            self.file_html.write(tabs(5)+"document.getElementById('%sbut'+lev).style.background = 'rgb('+color%s[0]+' '+color%s[1]+' '+color%s[2]+')';\n"%(nc,nc,nc,nc))
            self.file_html.write(tabs(5)+"if ((color%s[0]*0.299 + color%s[1]*0.587 + color%s[2]*0.114) > 130) {\n"%(nc,nc,nc))
            self.file_html.write(tabs(6)+"document.getElementById('%sbut'+lev).style.color = 'black';\n"%nc)
            self.file_html.write(tabs(5)+"} else {\n")
            self.file_html.write(tabs(6)+"document.getElementById('%sbut'+lev).style.color = 'white';\n"%nc)
            self.file_html.write(tabs(5)+"}\n")
            self.file_html.write(tabs(4)+"}\n")
            self.file_html.write(tabs(3)+"}\n")
            self.file_html.write(tabs(2)+"</script>\n")
            
            
    def func_scalev(self):
        """
        Funtion to change the scale of the spectral axis.
        Must be after buttons().

        Parameters
        ----------
        coords : 3x3 array
            Array with the coordinates of the cube as the difference from the centre.
            In the order RA, DEC and the spectral axis.
            Like write_x3d().diff_coords().
        gal : dict
            Dictionary with the coordinates of the galaxies.
            Like the one used for make_galaxies().
        axes : string
            A string indicating what axis labels to include. Can be 'real' for the equatorial
            coordinates; 'diff' for the difference from the center of the cube or
            'both' for both options. Leave None for no axis labels.
            Must be the same as in make_labels(). The default is 'both.
        move2d : bool
            Wether a 2D image is included in the web page. The default is True.

        """
        self.file_html.write(tabs(2)+"<script>\n")
        self.file_html.write(tabs(2)+"const inpscasv = document.querySelector('#scalev');\n")
        if self.cube.image2d is not None:
            self.file_html.write(tabs(2)+"const inpmovesv = document.querySelector('#move2dimg');\n")
        #self.file_html.write(tabs(2)+"inpscasv.addEventListener('change', changescalev);\n")
        self.file_html.write(tabs(2)+"function changescalev()\n\t\t {\n")
        self.file_html.write(tabs(3)+"const sca = inpscasv.value;\n")
        if self.cube.image2d is not None:
            self.file_html.write(tabs(3)+"const move = inpmovesv.value;\n")
            self.file_html.write(tabs(4)+f"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(sca*move-1)*{self.cube.coords[2,1]});\n")
        #scale layers
        nlayers = [len(l) for l in self.cube.isolevels]
        numcubes = len(nlayers)
        for nc in range(numcubes):
            for nlays in range(nlayers[nc]):
                for sp in range(self.cube.iso_split[nc][nlays]):
                    self.file_html.write(tabs(3)+"document.getElementById('cube__%slt%s_sp%s').setAttribute('scale', '1 1 '+sca);\n"%(nc,nlays,sp))
        #scale outline
        self.file_html.write(tabs(3)+"document.getElementById('cube__ot').setAttribute('scale', '1 1 '+sca);\n")
        #move galaxies
        if self.cube.galaxies is not None:
            for (n,gal) in enumerate(self.cube.galaxies):
                v = self.cube.galaxies[gal]['coord'][2]
                a = self.cube.galaxies[gal]['coord'][0]
                b = self.cube.galaxies[gal]['coord'][1]
                self.file_html.write(tabs(3)+"document.getElementById('cube__%s_cross_tra').setAttribute('translation', '0 0 '+(sca-1)*%s);\n"%(gal,v))
                self.file_html.write(tabs(3)+"document.getElementById('cube__%s_sphere_tra').setAttribute('translation', '%s %s '+sca*%s);\n"%(gal,a,b,v))
                self.file_html.write(tabs(3)+"document.getElementById('cube__glt%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(n,a,b,v))
        #scale grids
        self.file_html.write("\t\t\t document.getElementById('cube__tlt').setAttribute('scale', '1 1 '+sca);\n")

        #scale interactive tubes
        self.file_html.write(tabs(3)+"const arr = [];\n")
        self.file_html.write(tabs(3)+"const ind = 0;\n")
        self.file_html.write(tabs(3)+"for (sel=0; sel<document.getElementById('new-tubes').length; sel++) {\n")
        self.file_html.write(tabs(4)+"const tubeInd = document.getElementById('new-tubes')[sel].value.split('')[4];\n")
        self.file_html.write(tabs(4)+"arr.push(Number(tubeInd));\n")
        self.file_html.write(tabs(4)+"for (j=0; j<tubelen[arr[sel]]; j++) {\n")
        self.file_html.write(tabs(5)+"document.getElementById(arr[sel]+'newtra'+j).setAttribute('translation', tpars[arr[sel]][j][0][0]+' '+tpars[arr[sel]][j][0][1]+' '+sca*tpars[arr[sel]][j][0][2]);\n")
        self.file_html.write(tabs(5)+"const norm = Math.sqrt(tpars[arr[sel]][j][1][0]**2+tpars[arr[sel]][j][1][1]**2+(tpars[arr[sel]][j][1][2]*sca)**2)*1.03;\n")
        self.file_html.write(tabs(5)+"const angle = Math.acos(tpars[arr[sel]][j][1][1]/norm);\n")
        self.file_html.write(tabs(5)+"document.getElementById(arr[sel]+'newtra'+j).setAttribute('rotation', tpars[arr[sel]][j][1][2]*sca+' 0 '+(-tpars[arr[sel]][j][1][0])+' '+angle);\n")
        self.file_html.write(tabs(5)+"document.getElementById(arr[sel]+'newcyl'+j).setAttribute('height', norm);\n")
        self.file_html.write(tabs(4)+"}\n")
        self.file_html.write(tabs(3)+"}\n")

        #scale axes
        ax, axtick = labpos
        
        for i in range(12):
            if i < 6:
                self.file_html.write(tabs(3)+"document.getElementById('cube__alt_diff%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, ax[i][0], ax[i][1], ax[i][2])) #str(ax[i])[1:-1]
                self.file_html.write(tabs(3)+"document.getElementById('cube__alt_real%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, ax[i][0], ax[i][1], ax[i][2]))
            self.file_html.write(tabs(3)+"document.getElementById('cube__att_diff%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, axtick[i][0], axtick[i][1], axtick[i][2]))
            self.file_html.write(tabs(3)+"document.getElementById('cube__att_real%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, axtick[i][0], axtick[i][1], axtick[i][2]))
        
        self.file_html.write(tabs(2)+"}\n")
        self.file_html.write(tabs(2)+"</script>\n")
    
    def func_image2d(self):
        """
        Function to show/hide the 2D image.
        Must be after buttons().
        """
        # PICKING
        # self.file_html.write(roundTo) #premade string with function to round to two decimals

        self.file_html.write(tabs(2)+"<script>\n")
        
        self.file_html.write(tabs(2)+"function setimage2d()\n\t\t{\n")
        self.file_html.write(tabs(2)+"if(document.getElementById('cube__image2d').getAttribute('scale') != '1 1 1')\n")
        self.file_html.write(tabs(3)+"document.getElementById('cube__image2d').setAttribute('scale', '1 1 1');\n")
        self.file_html.write(tabs(2)+"else \n")
        self.file_html.write(tabs(3)+"document.getElementById('cube__image2d').setAttribute('scale', '0 0 0');\n")
        self.file_html.write(tabs(2)+"}\n\t\t</script>\n")

            
    def close_html(self):
        """
        Must be used to finish and close the HTML file. Not using this function results
        in an error.

        """
        if self.cube.interface != 'minimal':
            self.file_html.write(tablehtml)
        self.file_html.write('\n\t</body>\n</html>')
        self.file_html.close()

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
    for i in range(len(isolevels)):
        m = (end-start)/(np.max(isolevels)-np.min(isolevels))
        pos = int((m*isolevels[i]-m*np.min(isolevels))+start)
        cmap.append(f'{colors[pos][0]:.5e} {colors[pos][1]:.5e} {colors[pos][2]:.5e}')
    return cmap

def get_coords(ra, dec, v):
    ra_real = np.array([ra[0], np.mean(ra), ra[1]])
    dec_real = np.array([dec[0], np.mean(dec), dec[1]])
    v_real = np.array([v[0], np.mean(v), v[1]])
    
    ramin = ((ra[0]-np.mean(ra))*np.cos(np.radians(dec[0])))
    ramax = ((ra[1]-np.mean(ra))*np.cos(np.radians(dec[0])))
    decmin = (dec[0]-np.mean(dec))
    decmax = (dec[1]-np.mean(dec))
    vmin = (v[0]-np.mean(v))
    vmax = (v[1]-np.mean(v))
    
    ra_diff = np.array([ramin, 0, ramax])
    dec_diff = np.array([decmin, 0, decmax])
    v_diff = np.array([vmin, 0, vmax])
    
    return np.array([ra_real, dec_real, v_real]), np.array([ra_diff, dec_diff, v_diff])

def tabs(n):
    return '\t'*n

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
    return isolevels

def objquery(result, coords, otype):
    """
    Constrain query table to certain coordinates and object type
    """
    result = result[result['Type'] == otype]
    result = result[result['Velocity'] >= coords[2,0]]
    result = result[result['Velocity'] <= coords[2,1]]
    result = result[result['RA'] >= coords[0,0]]
    result = result[result['RA'] <= coords[0,1]]
    result = result[result['DEC'] >= coords[1,0]]
    result = result[result['DEC'] <= coords[1,1]]
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


def preview2d(cube, vmin1=None, vmax1=None, vmin2=None, vmax2=None, norm='asinh', figsize=(10,8)):
    """
    

    Parameters
    ----------
    cube : 3d array
        The data cube. Must be unitless.
    vmin(1,2) : float, optional
        Minimum value for the colormap. If None the minimum of the image is taken.
        The default is None.
    vmax(1,2) : float, optional
        Maximum value for the colormap. If None the maximum of the image is taken.
        The default is None.
    norm : string
        A scale name, one of 'asinh', 'function', 'functionlog', 'linear', 'log', 'logit' or
        'symlog'. Default is 'asinh'.
        For more information see `~matplotlib.colors.Normalize`.

    Returns
    -------
    None.

    """


    nz, ny, nx = cube.shape
    cs1 = np.sum(cube, axis=0)
    cs2 = np.sum(cube, axis=2)
    if vmin1 == None: vmin1 = np.min(cs1)
    if vmax1 == None: vmax1 = np.max(cs1)
    if vmin2 == None: vmin2 = np.min(cs2)
    if vmax2 == None: vmax2 = np.max(cs2)

    _, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    ax[0,0].hist(cs1.flatten(), density=True)
    #imshow plots axes fist -> y , second -> x
    ax[0, 1].imshow(cs1, vmin=vmin1, vmax=vmax1, norm=norm, origin='lower') 
    ax[0, 1].set_ylabel('DEC')
    ax[0, 1].set_xlabel('RA')

    ax[0, 1].set_yticks(np.arange(0, ny+1, 50), labels=np.arange(0, ny+1, 50), minor=False)
    ax[0, 1].set_xticks(np.arange(0, nx+1, 50), labels=np.arange(0, nx+1, 50), minor=False)
    ax[0, 1].grid(which='major')

    ax[1, 0].hist(cs2.flatten(), density=True)
    #imshow plots axes fist -> y , second -> x
    ax[1, 1].imshow(cs2.transpose(), vmin=vmin2, vmax=vmax2, norm=norm, origin='lower') 
    ax[1, 1].set_ylabel('DEC')
    ax[1, 1].set_xlabel('V')

    ax[1, 1].set_yticks(np.arange(0, ny+1, 50), labels=np.arange(0, ny+1, 50), minor=False)
    ax[1, 1].set_xticks(np.arange(0, nz+1, 50), labels=np.arange(0, nz+1, 50), minor=False)
    ax[1, 1].grid(which='major')    
    
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
    imhead = img[0].header
    img = img[0].data
    imw = wcs.WCS(imhead)
    
    try:
        verts = verts.to('deg')
    except AttributeError:
        verts = verts * u.Unit(unit)
    
    ll_ra,ll_dec = imw.world_to_pixel(SkyCoord(verts[0],verts[2]))
    lr_ra,_ = imw.world_to_pixel(SkyCoord(verts[1],verts[2]))
    _,ul_dec = imw.world_to_pixel(SkyCoord(verts[0],verts[3]))
    if ll_ra < 0 or ll_dec < 0 or lr_ra < 0 or ul_dec < 0:
        print('ERROR: The image is smaller than the cube. Increase parameter "pixels"')
        print("Pixel indices for [ra1, dec1, ra2, dec2] = "+str([ll_ra, ll_dec, lr_ra, ul_dec])+". Set 'pixels' parameter higher than the difference between 1 and 2.")
        raise ValueError
    img = img[int(ll_dec):int(ul_dec), int(lr_ra):int(ll_ra)] #dec first, ra second!!
    
    shape = img.shape
    
    img = img-np.min(img)
    img = (img)/np.max(img)
    
    colimg = cm.get_cmap(cmap)(img)[:,:,0:3]
    colimg = colimg.reshape((-1,3),order='F')
    
    imcol = [colors.rgb2hex(c).replace('#','0x') for c in colimg]
    if len(imcol)% 8 == 0:
        imcol = np.array(imcol).reshape(int(len(imcol)/8),8)
    
    return imcol, shape, img

def transpose(array, delta):
    """
    Transpose data array taking the direction of delta into account.
    """
    return np.transpose(array, (2,1,0))[::int(np.sign(delta[0])),
                                        ::int(np.sign(delta[1])),::int(np.sign(delta[2]))]
    
# Some attributes for the classes and functions

roundto = "\t<script>\n\t\t //Round a float value to x.xx format\n\t\t function roundTo(value, decimals)\n\t\t{\n\t\t\t return (Math.round(value * 10**decimals)) / 10**decimals;\n\t\t }\n\t</script>\n"

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

tablehtml = '\n<!--A table with navigation info for X3DOM-->\n<br/>\n<hr>\n<h3><b>Navigation:</b></h3>\n<table style="border-collapse: collapse; border: 2px solid rgb(0,0,0);">\n<tbody><tr style="background-color: rgb(220,220,220); border: 1px solid rgb(0,0,0);">\n<th width="250px">Function</th>\n<th>Mouse Button</th>\n</tr>\n</tbody><tbody>\n<tr style="background-color: rgb(240,240,240);"><td>Rotate</td>\n<td>Left / Left + Shift</td>\n</tr>\n<tr><td>Pan</td>\n<td>Mid / Left + Ctrl</td>\n</tr>\n<tr style="background-color: rgb(240,240,240);"><td>Zoom</td>\n<td>Right / Wheel / Left + Alt</td>\n</tr>\n<tr><td>Set center of rotation</td>\n<td>Double-click left</td>\n</tr>\n</tbody>\n</table>'

#name of ax labels for difference from center
axlabname1 = np.array(['R.A. [arcsec]', 'Dec. [arcsec]', 'V [km/s]',
                'Dec. [arcsec]', 'V [km/s]', 'R.A. [arcsec]'])

def get_axlabnames(mags, units):
    """
    Parameters:
    ----------
    mags : array
        Array with the names of the magnitudes. Must be of length 3.
    units : array
        Array with the names of the units. Must be length 4, the units of the corresponding
        magnitudes being the last 3 elements.
    """
    return np.array([mags.split('-')[0]+' ('+units[1]+')',
                     mags.split('-')[0]+' ('+units[2]+')',
                     mags.split('-')[0]+' ('+units[3]+')',
                     mags.split('-')[0]+' ('+units[2]+')',
                     mags.split('-')[0]+' ('+units[3]+')',
                     mags.split('-')[0]+' ('+units[1]+')'])
    
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
side,nam = np.array([['front',"R.A. - Dec."],['side',"Z - Dec."],['side2',"Z - R.A."],['perspective',"Perspective View"]]).T

default_cmaps = ['magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'twilight_shifted', 'turbo', 'Blues', 'BrBG', 'BuGn', 'BuPu',
 'CMRmap', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu',
 'RdYlBu', 'RdYlGn', 'Reds', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr',
 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg',
 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'jet', 'nipy_spectral', 'ocean', 'pink', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'terrain',
 'winter', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'magma_r', 'inferno_r',
 'plasma_r', 'viridis_r', 'cividis_r', 'twilight_r', 'twilight_shifted_r', 'turbo_r', 'Blues_r', 'BrBG_r', 'BuGn_r', 'BuPu_r', 'CMRmap_r',
 'GnBu_r', 'Greens_r', 'Greys_r', 'OrRd_r', 'Oranges_r', 'PRGn_r', 'PiYG_r', 'PuBu_r', 'PuBuGn_r', 'PuOr_r', 'PuRd_r', 'Purples_r', 'RdBu_r', 'RdGy_r',
 'RdPu_r', 'RdYlBu_r', 'RdYlGn_r', 'Reds_r', 'Spectral_r', 'Wistia_r', 'YlGn_r', 'YlGnBu_r', 'YlOrBr_r', 'YlOrRd_r', 'afmhot_r', 'autumn_r', 'binary_r',
 'bone_r', 'brg_r', 'bwr_r', 'cool_r', 'coolwarm_r', 'copper_r', 'cubehelix_r', 'flag_r', 'gist_earth_r', 'gist_gray_r', 'gist_heat_r', 'gist_ncar_r',
 'gist_rainbow_r', 'gist_stern_r', 'gist_yarg_r', 'gnuplot_r', 'gnuplot2_r', 'gray_r', 'hot_r', 'hsv_r', 'jet_r', 'nipy_spectral_r', 'ocean_r', 'pink_r',
 'prism_r', 'rainbow_r', 'seismic_r', 'spring_r', 'summer_r', 'terrain_r', 'winter_r', 'Accent_r', 'Dark2_r', 'Paired_r', 'Pastel1_r', 'Pastel2_r',
 'Set1_r', 'Set2_r', 'Set3_r', 'tab10_r', 'tab20_r', 'tab20b_r', 'tab20c_r']

astropy_prefixes = ['','k','m','M','u','G','n','d','c','da','h','p','T','f','a','P','E','z','y','Z','Y','r','q','R','Q']
angular_units = ['arcsec', 'arcmin', 'deg', 'rad']