#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:45:23 2024

@author: ixakalabadie
"""

from astropy.coordinates import Angle

from . import misc
from . import np
from . import u

class WriteX3D:
    """
    Class to create a X3D model of iso-surfaces with 3D spectral line data.
    Creates an X3D file with the model.

    Parameters
    ----------
    filename : str
        Name of the X3D file including the extension (.x3d).
    cube : Cube
        Object of the Cube class.
    """
    def __init__(self, filename, cube):
        self.cube = cube
        self.file_x3d = open(filename, 'w', encoding="utf-8")
        self.file_x3d.write('<?xml version="1.0" encoding="UTF-8"?>\n <!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN" \n "http://www.web3d.org/specifications/x3d-3.3.dtd">')
        self.file_x3d.write('\n <X3D profile="Immersive" version="3.3" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.3.xsd">')
        self.file_x3d.write(f'\n <head>\n\t<meta name="file" content="{filename}"></meta>\n')
        # Additional metadata MAKE AUTOMATIC
        # if meta != None:
        #     for met in meta.keys():
        #         self.file_x3d.write('\n\t<meta name="%s" content="%s"></meta>'%(met,meta[met]))
        # {cube.picking}
        self.file_x3d.write(
            '</head>\n\t<Scene doPickPass="False">\n' + \
                '\t\t<Background DEF="back" skyColor="0.6 0.6 0.6"></Background>\n')
        self.file_x3d.write(
            misc.tabs(2)+'<NavigationInfo type=\'"EXAMINE" "ANY"\' speed="4" headlight="true"></NavigationInfo>\n')
        self.file_x3d.write(
            misc.tabs(2)+'<DirectionalLight ambientIntensity="1" intensity="0" color="1 1 1"></DirectionalLight>\n')
        self.file_x3d.write(misc.tabs(2)+'<Transform DEF="ROOT" translation="0 0 0">\n')

    def make_layers(self, shifts=None, add_normals=False):
        """
        Calculate iso-surfaces from the data and write the objects in the X3D file.

        Parameters
        ----------
        shift : list, optional
            A list with a arrays of 3D vectors giving the shift in RA, DEC and spectral axis in
            the same units given to the cube. Similar to l_cube or l_isolevels.
        add_normals : bool, optional
            Whether to add normal vectors in the X3D model. Default is False.
        """
        numcubes = len(self.cube.l_cubes)
        self.cube.iso_split = []

        for nc in range(numcubes):
            cube_full = self.cube.l_cubes[nc]
            isolevels = self.cube.l_isolevels[nc]
            self.cube.iso_split.append(np.zeros((len(isolevels)), dtype=int))
            rgbcolors = misc.create_colormap(self.cube.cmaps[nc], isolevels)
            for (i,lev) in enumerate(isolevels):
                # calculate how many times to split the cube, 1 means the cube stays the same
                split = int(np.sum(cube_full>lev)/700000)+1
                self.cube.iso_split[nc][i] = split
                _, _, nz = cube_full.shape

                for sp in range(split):
                    cube = cube_full[:,:,int(nz/split*sp):int(nz/split*(sp+1))]
                    if lev > np.max(cube) or lev < np.min(cube):
                        print(f'Level {lev} is out of bounds for cube {nc} split {sp}. (min,max) = ({np.min(cube)},{np.max(cube)})')
                        verts, faces, normals = None, None, None
                    else:
                        try:
                            if shifts is not None:
                                verts, faces, normals = misc.marching_cubes(cube, level=lev,
                                            shift=shifts[nc], step_size=self.cube.resol)
                            else:
                                verts, faces, normals = misc.marching_cubes(cube, level=lev,
                                                                step_size=self.cube.resol)
                        except Exception as ex:
                            print(ex)
                            continue
                    self.file_x3d.write(f'\n\t\t\t<Transform DEF="{nc}lt{i}_sp{sp}" ' \
                                        +' translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">')
                    self.file_x3d.write(f'\n\t\t\t\t<Shape DEF="{nc}layer{i}_sp{sp}_shape">')
                    if self.cube.image2d[1] is not None:
                        sortType = 'transparent'
                    else:
                        sortType = 'opaque'
                    self.file_x3d.write(f'\n\t\t\t\t\t<Appearance DEF="{nc}layer{i}_sp{sp}_appe" sortType="{sortType}" sortKey="{len(isolevels)-1-i}">')
                    self.file_x3d.write(f'\n{misc.tabs(6)}<Material DEF="{nc}layer{i}_sp{sp}" '\
                            + 'ambientIntensity="0" emissiveColor="0 0 0" '\
                            + f'diffuseColor="{rgbcolors[i]}" specularColor=' \
                            +f'"0 0 0" shininess="0.0078" transparency="0.8"></Material>')
                    #correct color with depthmode (ALSO FOR LAST LAYER?)
                    # if i != len(isolevels)-1:
                    self.file_x3d.write('\n'+misc.tabs(6)+'<DepthMode readOnly="true"></DepthMode>')
                    self.file_x3d.write('\n'+misc.tabs(5)+'</Appearance>')
                    #define the layer object
                    if verts is not None:
                        if add_normals:
                            self.file_x3d.write('\n'+misc.tabs(5)+'<IndexedFaceSet solid="false" '\
                            +'colorPerVertex="false" normalPerVertex="true" coordIndex="\n\t\t\t\t\t\t')
                        else:
                            self.file_x3d.write('\n'+misc.tabs(5)+'<IndexedFaceSet solid="false" colorPerVertex="false" normalPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
                        #write indices
                        np.savetxt(self.file_x3d, faces, fmt='%i', newline=' -1\n\t\t\t\t\t\t')
                        self.file_x3d.write('">')
                        self.file_x3d.write(f'\n\t\t\t\t\t\t<Coordinate DEF="{nc}Coordinates{i}_sp{sp}" point="\n\t\t\t\t\t\t')
                        #write coordinates
                        np.savetxt(self.file_x3d, verts,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
                        self.file_x3d.write('"></Coordinate>')
                        if add_normals:
                            self.file_x3d.write(f'\n\t\t\t\t\t\t<Normal DEF="{nc}Normals{i}_sp{sp}" vector="\n\t\t\t\t\t\t')
                            #write normals
                            np.savetxt(self.file_x3d, normals,fmt='%.5f', newline=',\n\t\t\t\t\t\t')
                            self.file_x3d.write('"></Normal>')
                        self.visfile.write('\n'+misc.tabs(5)+'</IndexedFaceSet>\n')
                    self.visfile.write(misc.tabs(4)+'</Shape>\n')
                    self.visfile.write(misc.tabs(3)+'</Transform>')

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
        self.file_x3d.write('\n\t\t\t\t\t\t<Material DEF="outline" '\
                            +f'emissiveColor="{col}" diffuseColor="0 0 0"></Material>')
        self.file_x3d.write('\n\t\t\t\t\t</Appearance>')
        self.file_x3d.write('\n\t\t\t\t\t<IndexedLineSet colorPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
        #write indices
        np.savetxt(self.file_x3d, misc.outlineindex, fmt='%i', newline='\n\t\t\t\t\t\t')
        self.file_x3d.write('">')
        self.file_x3d.write('\n\t\t\t\t\t\t<Coordinate DEF="OutlineCoords" point="\n\t\t\t\t\t\t')
        #write coordinates
        np.savetxt(self.file_x3d, outlinecoords,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
        self.file_x3d.write('"></Coordinate>')
        self.file_x3d.write('\n\t\t\t\t\t</IndexedLineSet>\n\t\t\t\t</Shape>\n\t\t\t</Transform>\n')

    def make_galaxies(self):
        """
        Creates spheres and labels in the model at the location of galaxies given as input.
        """        
        gals = self.cube.galaxies

        sphereradius = 2000/45
        crosslen = 2000/20
        #create galaxy crosses and spheres
        for i, gal in enumerate(gals.keys()):
            vec = gals[gal]['coord']
            # #galaxy crosses
            # self.file_x3d.write(misc.tabs(3)+f'<Transform DEF="{gal}_cross_tra" ' \
            #                     +'translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n')
            # self.file_x3d.write(misc.tabs(4)+'<Shape ispickable="false">\n')
            # self.file_x3d.write(misc.tabs(5)+'<Appearance>\n')
            # col = '0 0 0'
            # self.file_x3d.write(misc.tabs(6)+f'<Material DEF="{gal}_cross" emissiveColor="{col}" ' \
            #                     + 'diffuseColor="0 0 0"></Material>\n')
            # self.file_x3d.write(misc.tabs(5)+'</Appearance>\n')
            # #cross indices
            # self.file_x3d.write(misc.tabs(5)+'<IndexedLineSet colorPerVertex="true" coordIndex="\n' \
            #             +misc.tabs(6)+'0 1 -1\n'+misc.tabs(6)+'2 3 -1\n'+misc.tabs(6)+'4 5 -1\n'+misc.tabs(6)+'">\n')
            # self.file_x3d.write(f'{misc.tabs(5)}<Coordinate DEF="CrossCoords{i}" point="\n{misc.tabs(6)}')
            # crosscoords = np.array([[vec[0]-crosslen,vec[1],vec[2]],
            #                   [vec[0]+crosslen,vec[1],vec[2]],
            #                   [vec[0],vec[1]-crosslen, vec[2]],
            #                   [vec[0],vec[1]+crosslen, vec[2]],
            #                   [vec[0],vec[1],vec[2]-crosslen],
            #                   [vec[0],vec[1],vec[2]+crosslen]])
            # #cross coordinates
            # np.savetxt(self.file_x3d, crosscoords, fmt='%.3f', newline='\n\t\t\t\t\t\t')
            # self.file_x3d.write(misc.tabs(6)+'"></Coordinate>\n')
            # self.file_x3d.write(misc.tabs(3)+'</IndexedLineSet>\n\t\t\t\t</Shape>\n\t\t\t</Transform>\n')
            #galaxy spheres (ADD SCALE, ROTATION, ETC.??)
            self.file_x3d.write(f'{misc.tabs(3)}<Transform DEF="{gal}_sphere_tra" translation="{vec[0]} {vec[1]} {vec[2]}">\n')
            self.file_x3d.write(f'{misc.tabs(4)}<Shape ispickable="false">\n')
            self.file_x3d.write(f'{misc.tabs(5)}<Sphere radius="{sphereradius}" solid="false"></Sphere>\n')
            self.file_x3d.write(f'{misc.tabs(5)}<Appearance>\n')
            self.file_x3d.write(f'{misc.tabs(6)}<Material DEF="{gal}" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="{gals[gal]["col"]}" specularColor="0 0 0" shininess="0.0078" transparency="0"></Material>\n')
            self.file_x3d.write(misc.tabs(5)+'</Appearance>\n\t\t\t\t</Shape>\n\t\t\t</Transform>\n')

            self.file_x3d.write(misc.tabs(4)+f'<Transform DEF="glt{i}" translation="{gals[gal]["coord"][0]} {gals[gal]["coord"][1]} {gals[gal]["coord"][2]}" rotation="0 1 0 3.14" scale="20 20 20">\n') # labelscale = 20
            self.file_x3d.write(misc.tabs(5)+'<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">\n')
            self.file_x3d.write(misc.tabs(6)+'<Shape ispickable="false">\n')
            self.file_x3d.write(misc.tabs(6)+'<Appearance>\n')
            self.file_x3d.write(misc.tabs(6)+f'<Material DEF="label_{gal}" diffuseColor="0 0 0" emissiveColor="0 0 0"></Material>\n')
            self.file_x3d.write(misc.tabs(6)+'</Appearance>\n')
            self.file_x3d.write(misc.tabs(6)+f'<Text string="{gal}">\n')
            self.file_x3d.write(misc.tabs(7)+f'<FontStyle DEF="{gal}_fs" family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="5"></FontStyle>\n')
            self.file_x3d.write(misc.tabs(6)+'</Text>\n')
            self.file_x3d.write(misc.tabs(6)+'</Shape>\n')
            self.file_x3d.write(misc.tabs(5)+'</Billboard>\n')
            self.file_x3d.write(misc.tabs(4)+'</Transform>\n')
            
    def make_image2d(self):
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
        imcol, img_shape = self.cube.image2d[1]

        # coordinates of 2d image
        coords2d = np.array([[1000,-1000,1000],
                             [1000,1000,1000],
                             [-1000,-1000,1000],
                             [-1000,1000,1000]])
        
        self.file_x3d.write(misc.tabs(3)+'<Transform DEF="image2d" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n')
        self.file_x3d.write(misc.tabs(4)+'<Shape ispickable="False">\n')
        self.file_x3d.write(misc.tabs(5)+'<Appearance>\n')
        self.file_x3d.write(misc.tabs(6)+'<Material DEF="immat" ambientIntensity="1" emissiveColor="0 0 0" diffuseColor="1 1 1" shininess="0.0078"></Material>\n')
        if imcol is not None and img_shape is not None:
            self.file_x3d.write(f'{misc.tabs(6)}<PixelTexture repeatS="false" repeatT="false" image="{img_shape[0]} {img_shape[1]} 3 \n')
            # write pixel colors
            np.savetxt(self.file_x3d, imcol, fmt='%s', delimiter=' ', newline='\n')
            self.file_x3d.write('"></PixelTexture>\n')
        self.file_x3d.write(misc.tabs(5)+'</Appearance>\n')            
        #SOLID=TRUE makes it transparent from one side
        self.file_x3d.write(misc.tabs(5)+'<IndexedFaceSet solid="false" colorPerVertex="false" normalPerVertex="false" coordIndex="2 3 1 0 -1">\n')
        self.file_x3d.write(misc.tabs(6)+'<Coordinate DEF="imgCoords" point="\n\t\t\t\t\t\t')
        # write coordinates
        np.savetxt(self.file_x3d, coords2d, fmt='%.3f', newline='\n\t\t\t\t\t\t')
        self.file_x3d.write('"></Coordinate>\n')
        self.file_x3d.write(misc.tabs(6)+'<TextureCoordinate DEF="imgTexCoords" point="\n'+misc.tabs(6)+' 0 0, 1 0, 0 1, 1 1"></TextureCoordinate>\n')
        self.file_x3d.write(misc.tabs(5)+'</IndexedFaceSet>\n')
        self.file_x3d.write(misc.tabs(4)+'</Shape>\n')
        self.file_x3d.write(misc.tabs(3)+'</Transform>\n')
            
    def make_ticklines(self):
        """
        Create tickline objects in the X3D model.

        Closes the Transform "ROOT" element. Must be called somewhere after make_layers() and before make_labels().
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
        self.file_x3d.write(misc.tabs(3)+'<Transform DEF="tlt" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n')
        self.file_x3d.write(misc.tabs(4)+'<Shape ispickable="false">\n')
        self.file_x3d.write(misc.tabs(5)+'<Appearance>\n')
        #set color
        col = '0 0 0'
        self.file_x3d.write(f'{misc.tabs(6)}<Material DEF="ticklines" emissiveColor="{col}" diffuseColor="0 0 0"></Material>\n')
        self.file_x3d.write(misc.tabs(5)+'</Appearance>\n')
        self.file_x3d.write(misc.tabs(5)+'<IndexedLineSet colorPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
        #write indices
        np.savetxt(self.file_x3d, misc.ticklineindex, fmt='%i', newline='\n\t\t\t\t\t\t')
        self.file_x3d.write('">\n')
        self.file_x3d.write(misc.tabs(6)+'<Coordinate DEF="ticklineCoords" point="\n\t\t\t\t\t\t')
        #write coordinates
        np.savetxt(self.file_x3d, ticklinecoords,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
        self.file_x3d.write('"></Coordinate>\n')
        self.file_x3d.write(misc.tabs(5)+'</IndexedLineSet>\n')
        self.file_x3d.write(misc.tabs(4)+'</Shape>\n')
        self.file_x3d.write(misc.tabs(3)+'</Transform>\n')

    def make_animation(self, cycleinterval=10, axis=0):
        """
        Create an animation to rotate the X3D model along one axis.
        Must be outside the Transform "ROOT" element. Should be called after make_ticklines(). 
        """
        vec = np.zeros(3,dtype=int)
        vec[axis] = 1
        vec = str(vec)[1:-1]
        self.file_x3d.write(f'\n{misc.tabs(2)}<timeSensor DEF="time" cycleInterval="{cycleinterval}" loop="true" enabled="true" startTime="-1"></timeSensor>')
        self.file_x3d.write(f'\n{misc.tabs(2)}<OrientationInterpolator DEF="move" key="0 0.5 1" keyValue="{vec} 0 {vec} 3.14 {vec} 6.28"></OrientationInterpolator>')
        self.file_x3d.write('\n'+misc.tabs(2)+'<Route fromNode="time" fromField ="fraction_changed" toNode="move" toField="set_fraction"></Route>')
        self.file_x3d.write('\n'+misc.tabs(2)+'<Route fromNode="move" fromField ="value_changed" toNode="ROOT" toField="rotation"></Route>')


    def make_labels(self):
        """
        Create the labels of different elements in the figure.
        """
        self.file_x3d.write('\n\t\t<ProximitySensor DEF="PROX_LABEL" size="1.0e+06 1.0e+06 1.0e+06"></ProximitySensor>')
        self.file_x3d.write('\n\t\t<Collision enabled="false">')

        try:
            ramin1, ramax1 = (self.cube.coords[0]-np.mean(self.cube.coords[0])) \
                    * np.cos(self.cube.coords[1,0]*u.Unit(self.cube.units[2]).to('rad')) \
                    * u.Unit(self.cube.units[1])
            decmin1, decmax1 = (self.cube.coords[1]-np.mean(self.cube.coords[1])) \
                    * u.Unit(self.cube.units[2])
            vmin1, vmax1 = (self.cube.coords[2]-np.mean(self.cube.coords[2])) \
                    * u.Unit(self.cube.units[3])
            
            if ramin1.unit.is_equivalent(u.arcsec) and ramax1 < 1*u.deg:
                ramin1 = ramin1.to('arcsec')
                ramax1 = ramax1.to('arcsec')
            if decmin1.unit.is_equivalent(u.arcsec) and decmax1 < 1*u.deg:
                decmin1 = decmin1.to('arcsec')
                decmax1 = decmax1.to('arcsec')
            if vmin1.unit.is_equivalent(u.km/u.s):
                vmin1 = vmin1.to('km/s')
                vmax1 = vmax1.to('km/s')
            elif vmin1.unit.is_equivalent('m') and vmin1 < 1*u.um:
                vmin1 = vmin1.to(u.Angstrom)
                vmax1 = vmax1.to(u.Angstrom)
            elif vmin1.unit.is_equivalent('m') and vmin1 < 1*u.mm:
                vmin1 = vmin1.to(u.um)
                vmax1 = vmax1.to(u.um)
            elif vmin1.unit.is_equivalent('Hz'):
                vmin1 = vmin1.to(u.GHz)
                vmax1 = vmax1.to(u.GHz)
        except Exception as ex:
            ramin1, ramax1 = (self.cube.coords[0]-np.mean(self.cube.coords[0]))
            decmin1, decmax1 = (self.cube.coords[1]-np.mean(self.cube.coords[1]))
            vmin1, vmax1 = (self.cube.coords[2]-np.mean(self.cube.coords[2]))

        try:
            ramin2, ramax2 = (self.cube.coords[0]) * u.Unit(self.cube.units[1])
            decmin2, decmax2 = (self.cube.coords[1]) * u.Unit(self.cube.units[2])
            vmin2, vmax2 = (self.cube.coords[2]) * u.Unit(self.cube.units[3])
            if vmin2.unit.is_equivalent(u.km/u.s):
                vmin2 = vmin2.to('km/s')
                vmax2 = vmax2.to('km/s')
            elif vmin2.unit.is_equivalent('m') and vmin1 < 1*u.um:
                vmin2 = vmin1.to(u.Angstrom)
                vmax2 = vmax1.to(u.Angstrom)
            elif vmin2.unit.is_equivalent('m') and vmin1 < 1*u.mm:
                vmin2 = vmin1.to(u.um)
                vmax2 = vmax1.to(u.um)
            elif vmin2.unit.is_equivalent('Hz'):
                vmin2 = vmin2.to(u.GHz)
                vmax2 = vmax2.to(u.GHz)
        except:
            ramin2, ramax2 = self.cube.coords[0]
            decmin2, decmax2 = self.cube.coords[1]
            vmin2, vmax2 = self.cube.coords[2]

        # scale of labels
        labelscale = 20

        ax, axtick = misc.labpos

        #Names for the axes tick labels
        axticknames1 = np.array([f'{ramax1:.2f}',f'{ramin1:.2f}',f'{decmax1:.2f}',
                       f'{decmin1:.2f}',f'{vmin1:.2f}',f'{vmax1:.2f}',
                       f'{decmax1:.2f}',f'{decmin1:.2f}',f'{vmin1:.2f}',
                       f'{vmax1:.2f}',f'{ramax1:.2f}',f'{ramin1:.2f}'])

        axticknames2 = np.array([f'{ramax2:.3f}', f'{ramin2:.3f}', f'{decmax2:.3f}',
                       f'{decmin2:.3f}', f'{vmin2:.3f}', f'{vmax2:.3f}',
                       f'{decmax2:.3f}', f'{decmin2:.3f}', f'{vmin2:.3f}',
                       f'{vmax2:.3f}', f'{ramax2:.3f}', f'{ramin2:.3f}'])

        col = '0 0 0'

        axlabnames = misc.get_axlabnames(mags=self.cube.mags)

        #ax labels diff
        for i in range(6):
            self.file_x3d.write(f'\n\t\t\t\t<Transform DEF="alt_diff{i}" translation="{ax[i,0]} {ax[i,1]} {ax[i,2]}" rotation="{misc.axlabrot[i]}" scale="{labelscale} {labelscale} {labelscale}">')
            self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
            self.file_x3d.write(f'\n\t\t\t\t\t\t\t<Material DEF="axlab_diff{i}" diffuseColor="0 0 0" emissiveColor="{col}"></Material>')
            self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
            self.file_x3d.write(f"\n\t\t\t\t\t\t<Text string='{axlabnames[i]}'>")
            self.file_x3d.write(f'\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'{misc.axlabeljustify[i]}\' size="10"></FontStyle>')
            self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')

        #ax tick labels diff
        for i in range(12):
            if i<4:
                rot = misc.axlabrot[0]
            elif i<8:
                rot = misc.axlabrot[2]
            else:
                rot = misc.axlabrot[4]
            self.file_x3d.write(f'\n\t\t\t\t<Transform DEF="att_diff{i}" translation="{axtick[i,0]} {axtick[i,1]} {axtick[i,2]}" rotation="{rot}" scale="{labelscale} {labelscale} {labelscale}">')
            self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
            self.file_x3d.write(f'\n\t\t\t\t\t\t\t<Material DEF="axtick_diff{i}" diffuseColor="0 0 0" emissiveColor="{col}"></Material>')
            self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
            self.file_x3d.write(f'\n\t\t\t\t\t\t<Text string="{axticknames1[i]}">')
            self.file_x3d.write(f'\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'{misc.axticklabjus[i]}\' size="8"></FontStyle>')
            self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
        
        # don't show other labels if overlay
        if isinstance(self.cube.lines, dict) == False:
            #ax labels real
            for i in range(6):
                self.file_x3d.write(f'\n\t\t\t\t<Transform DEF="alt_real{i}" translation="{ax[i,0]} {ax[i,1]} {ax[i,2]}" rotation="{misc.axlabrot[i]}" scale="{labelscale} {labelscale} {labelscale}">')
                self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.file_x3d.write(f'\n\t\t\t\t\t\t\t<Material DEF="axlab_real{i}" diffuseColor="0 0 0" emissiveColor="{col}" transparency="1"></Material>')
                self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
                self.file_x3d.write(f"\n\t\t\t\t\t\t<Text string='{axlabnames[i]}'>")
                self.file_x3d.write(f'\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'{misc.axlabeljustify[i]}\' size="10"></FontStyle>')
                self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
            #ax tick labels
            for i in range(12):
                if i<4:
                    rot = misc.axlabrot[0]
                elif i<8:
                    rot = misc.axlabrot[2]
                else:
                    rot = misc.axlabrot[4]
                self.file_x3d.write(f'\n\t\t\t\t<Transform DEF="att_real{i}" translation="{axtick[i,0]} {axtick[i,1]} {axtick[i,2]}" rotation="{rot}" scale="{labelscale} {labelscale} {labelscale}">')
                self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.file_x3d.write(f'\n\t\t\t\t\t\t\t<Material DEF="axtick_real{i}" diffuseColor="0 0 0" emissiveColor="{col}" transparency="1"></Material>')
                self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
                self.file_x3d.write(f'\n\t\t\t\t\t\t<Text string="{axticknames2[i]}">')
                self.file_x3d.write(f'\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'{misc.axticklabjus[i]}\' size="8"></FontStyle>')
                self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')

        self.file_x3d.write('\n\t\t</Collision>')

    def make_markers(self, geom, points, shape, trans, color, labels=None):
        """
        Create markers in the X3D model.
        Must be called before make_ticklines().

        Parameters
        ----------
        geom: str
            Type of geometry to create. Options are 'tube', 'sphere', 'box', 'cone'.
        points : list
            List of points to create the markers. Each element of the list is a len 3 array with the coordinates of the marker (RA,DEC,Z).
            If geom is 'tube', each element of the list is a list of len 3 arrays that define the tubes. Two points define one tube.
        shape : list
            List of the shape of the markers.
            If geom is 'box', each element of the list is a len 3 array with the size of the box (x,y,z).
            If geom is 'cone', each element of the list is a len 2 array with the radius and height of the cone.
            Else each element of the list is a float with the radius of the marker.
        delta : list
            List with the delta of the cube (header[CDELT]).
        trans : list
            Transformation factor to the cube. It is calculated with (2000/nx, 2000/ny, 2000/nz).
        color : list
            List with the color of the markers. Each element is a string with RGB values.
        """
        delta = np.array(self.cube.delta)
        if geom == 'tube':
            n = 0
            for tube in points:
                # transform points to model coordinates
                tube = (tube - np.mean(self.cube.coords, axis=1))/delta*trans
                # get mean point between consecutive points
                pos = np.array([str(np.mean((tube[i],tube[i+1]), axis=0))[1:-1] for i in range(len(tube)-1)])
                # get distance between consecutive points
                diff = np.diff(tube, axis=0)
                heights = np.linalg.norm(diff,axis=1)
                #get rotation for each tube
                angles = np.arccos(diff[:,1]/heights)

                # create x3d object
                for i in range(len(tube)-1):
                    self.file_x3d.write("\n"+misc.tabs(3)+f'<Transform DEF="pytub{n}tra{i}" translation="{pos[i]}" rotation="{diff[i,2]:.4f} 0 {-diff[i,0]:.4f} {angles[i]:.4f}" scale="1 1 1">\n')
                    self.file_x3d.write(misc.tabs(4)+'<Shape ispickable="false">\n')
                    self.file_x3d.write(misc.tabs(5)+f'<Cylinder DEF="py{n}tub{i}" height="{heights[i]*1.015}" radius="{shape[n]}" solid="false"></Cylinder>\n')
                    self.file_x3d.write(misc.tabs(5)+'<Appearance>\n')
                    self.file_x3d.write(misc.tabs(6)+f'<Material DEF="py{n}tubmat{i}" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="{color[n]}" specularColor="0 0 0" shininess="0.0078" transparency="0"></Material>\n')
                    self.file_x3d.write(misc.tabs(5)+'</Appearance>\n')
                    self.file_x3d.write(misc.tabs(4)+'</Shape>\n')
                    self.file_x3d.write(misc.tabs(3)+'</Transform>\n')
                if labels is not None:
                    self.file_x3d.write(misc.tabs(4)+f'<Transform DEF="label_pytubtra{n}" translation="{pos[0]}" rotation="0 1 0 3.14" scale="20 20 20">\n') # labelscale = 20
                    self.file_x3d.write(misc.tabs(5)+'<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">\n')
                    self.file_x3d.write(misc.tabs(6)+'<Shape ispickable="false">\n')
                    self.file_x3d.write(misc.tabs(6)+'<Appearance>\n')
                    self.file_x3d.write(misc.tabs(6)+f'<Material DEF="label_pytub{n}" diffuseColor="0 0 0" emissiveColor="0 0 0"></Material>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Appearance>\n')
                    self.file_x3d.write(misc.tabs(6)+f'<Text string="{labels[n]}">\n')
                    self.file_x3d.write(misc.tabs(7)+f'<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="5"></FontStyle>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Text>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Shape>\n')
                    self.file_x3d.write(misc.tabs(5)+'</Billboard>\n')
                    self.file_x3d.write(misc.tabs(4)+'</Transform>\n')

                n = n + 1

        if geom == 'sphere':
            n = 0
            for sphere in points:
                sphere = (sphere - np.mean(self.cube.coords, axis=1))/delta*trans
                sphere = str(sphere)[1:-1]
                self.file_x3d.write("\n"+misc.tabs(3)+f'<Transform DEF="pysph{n}tra" translation="{sphere}" scale="1 1 1">\n')
                self.file_x3d.write(misc.tabs(4)+'<Shape ispickable="false">\n')
                self.file_x3d.write(misc.tabs(5)+f'<Sphere DEF="py{n}sph" radius="{shape[n]}" solid="false"></Sphere>\n')
                self.file_x3d.write(misc.tabs(5)+'<Appearance>\n')
                self.file_x3d.write(misc.tabs(6)+f'<Material DEF="py{n}sphmat" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="{color[n]}" specularColor="0 0 0" shininess="0.0078" transparency="0"></Material>\n')
                self.file_x3d.write(misc.tabs(5)+'</Appearance>\n')
                self.file_x3d.write(misc.tabs(4)+'</Shape>\n')
                self.file_x3d.write(misc.tabs(3)+'</Transform>\n')

                if labels is not None:
                    self.file_x3d.write(misc.tabs(4)+f'<Transform DEF="label_pysphtra{n}" translation="{sphere}" rotation="0 1 0 3.14" scale="20 20 20">\n') # labelscale = 20
                    self.file_x3d.write(misc.tabs(5)+'<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">\n')
                    self.file_x3d.write(misc.tabs(6)+'<Shape ispickable="false">\n')
                    self.file_x3d.write(misc.tabs(6)+'<Appearance>\n')
                    self.file_x3d.write(misc.tabs(6)+f'<Material DEF="label_pysph{n}" diffuseColor="0 0 0" emissiveColor="0 0 0"></Material>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Appearance>\n')
                    self.file_x3d.write(misc.tabs(6)+f'<Text string="{labels[n]}">\n')
                    self.file_x3d.write(misc.tabs(7)+f'<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="5"></FontStyle>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Text>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Shape>\n')
                    self.file_x3d.write(misc.tabs(5)+'</Billboard>\n')
                    self.file_x3d.write(misc.tabs(4)+'</Transform>\n')

                n = n + 1
        
        if geom == 'box':
            n = 0
            for box in points:
                box = (box - np.mean(self.cube.coords, axis=1))/delta*trans
                box = str(box)[1:-1]
                self.file_x3d.write("\n"+misc.tabs(3)+f'<Transform DEF="pybox{n}tra" translation="{box}" scale="1 1 1">\n')
                self.file_x3d.write(misc.tabs(4)+'<Shape ispickable="false">\n')
                self.file_x3d.write(misc.tabs(5)+f'<Box DEF="py{n}box" size="{shape[n][0]*2} {shape[n][1]*2} {shape[n][2]*2}" solid="false"></Box>\n')
                self.file_x3d.write(misc.tabs(5)+'<Appearance>\n')
                self.file_x3d.write(misc.tabs(6)+f'<Material DEF="py{n}boxmat" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="{color[n]}" specularColor="0 0 0" shininess="0.0078" transparency="0"></Material>\n')
                self.file_x3d.write(misc.tabs(5)+'</Appearance>\n')
                self.file_x3d.write(misc.tabs(4)+'</Shape>\n')
                self.file_x3d.write(misc.tabs(3)+'</Transform>\n')

                if labels is not None:
                    self.file_x3d.write(misc.tabs(4)+f'<Transform DEF="label_pyboxtra{n}" translation="{box}" rotation="0 1 0 3.14" scale="20 20 20">\n') # labelscale = 20
                    self.file_x3d.write(misc.tabs(5)+'<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">\n')
                    self.file_x3d.write(misc.tabs(6)+'<Shape ispickable="false">\n')
                    self.file_x3d.write(misc.tabs(6)+'<Appearance>\n')
                    self.file_x3d.write(misc.tabs(6)+f'<Material DEF="label_pybox{n}" diffuseColor="0 0 0" emissiveColor="0 0 0"></Material>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Appearance>\n')
                    self.file_x3d.write(misc.tabs(6)+f'<Text string="{labels[n]}">\n')
                    self.file_x3d.write(misc.tabs(7)+f'<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="5"></FontStyle>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Text>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Shape>\n')
                    self.file_x3d.write(misc.tabs(5)+'</Billboard>\n')
                    self.file_x3d.write(misc.tabs(4)+'</Transform>\n')

                n = n + 1
            
        if geom == 'cone':
            n = 0
            for cone in points:
                cone = (cone - np.mean(self.cube.coords, axis=1))/delta*trans
                cone = str(cone)[1:-1]
                self.file_x3d.write("\n"+misc.tabs(3)+f'<Transform DEF="pycon{n}tra" translation="{cone}" scale="1 1 1">\n')
                self.file_x3d.write(misc.tabs(4)+'<Shape ispickable="false">\n')
                self.file_x3d.write(misc.tabs(5)+f'<Cone DEF="py{n}con" height="{shape[n][0]}" bottomRadius="{shape[n][1]}" solid="false"></Cone>\n')
                self.file_x3d.write(misc.tabs(5)+'<Appearance>\n')
                self.file_x3d.write(misc.tabs(6)+f'<Material DEF="py{n}conmat" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="{color[n]}" specularColor="0 0 0" shininess="0.0078" transparency="0"></Material>\n')
                self.file_x3d.write(misc.tabs(5)+'</Appearance>\n')
                self.file_x3d.write(misc.tabs(4)+'</Shape>\n')
                self.file_x3d.write(misc.tabs(3)+'</Transform>\n')

                if labels is not None:
                    self.file_x3d.write(misc.tabs(4)+f'<Transform DEF="label_pycontra{n}" translation="{cone}" rotation="0 1 0 3.14" scale="20 20 20">\n') # labelscale = 20
                    self.file_x3d.write(misc.tabs(5)+'<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">\n')
                    self.file_x3d.write(misc.tabs(6)+'<Shape ispickable="false">\n')
                    self.file_x3d.write(misc.tabs(6)+'<Appearance>\n')
                    self.file_x3d.write(misc.tabs(6)+f'<Material DEF="label_pycon{n}" diffuseColor="0 0 0" emissiveColor="0 0 0"></Material>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Appearance>\n')
                    self.file_x3d.write(misc.tabs(6)+f'<Text string="{labels[n]}">\n')
                    self.file_x3d.write(misc.tabs(7)+f'<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="5"></FontStyle>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Text>\n')
                    self.file_x3d.write(misc.tabs(6)+'</Shape>\n')
                    self.file_x3d.write(misc.tabs(5)+'</Billboard>\n')
                    self.file_x3d.write(misc.tabs(4)+'</Transform>\n')

                n = n + 1
                
    def close(self):
        """
        Closes the X3D file. Not using this function at the end results in
        an error.
        """
        #ending, close all
        self.file_x3d.write('\n\t\t</Transform>')
        self.file_x3d.write('\n\t</Scene>')
        self.file_x3d.write('\n</X3D>')
        self.file_x3d.close()

class WriteHTML:
    """
    Create an HTML file with an embbeded X3D figure and functions to interact with it.

    Parameters
    ----------
    filename : str
        Name of the HTML file including the extension (.html).
    cube : Cube
        Object of the Cube class.
    description : str, optional
        A description for the web page.
    pagetitle : str, optional
        The title of the web page.
    """
    def __init__(self, filename, cube, description=None, pagetitle=None):
        #some attributes to use later
        self.cube = cube
        if pagetitle is None:
            pagetitle = self.cube.name
        self.file_html = open(filename, 'w', encoding="utf-8")
        self.file_html.write('<!DOCTYPE html>\n\t <head>\n')
        self.file_html.write(misc.tabs(2)+"<script type='text/javascript' src='https://ixakalabadie.github.io/x3dom/x3dom.js'></script>\n")
        self.file_html.write(misc.tabs(2)+"<script type='text/javascript'  src='https://www.maths.nottingham.ac.uk/plp/pmadw/LaTeXMathML.js'></script>\n")
        self.file_html.write(misc.tabs(2)+"<link rel='stylesheet' type='text/css' href='https://ixakalabadie.github.io/x3dom/x3dom.css'></link>\n")
        self.file_html.write(misc.tabs(2)+"<script type='text/javascript' src='https://code.jquery.com/jquery-3.6.3.min.js'></script>\n")
        self.file_html.write(misc.tabs(2)+'<script src="https://ixakalabadie.github.io/x3dom/js-colormaps.js"></script> <!-- FOR COLORMAPS IN JS-->\n')
        self.file_html.write(misc.tabs(2)+'<script type="text/javascript" src="https://ixakalabadie.github.io/x3dom/markers.js"></script> <!-- FOR MARKERS IN JS-->\n')
        self.file_html.write(misc.tabs(2)+f'<title> ViSL3D: {self.cube.name}</title>\n')
        if self.cube.interface == 'minimal':
            self.file_html.write("\n\t\t<style>\n"+misc.tabs(3)+"x3d\n"+misc.tabs(4)+"{\n"+misc.tabs(5)+"border:2px solid darkorange;\n"+misc.tabs(5)+"width:100vw;\n"+misc.tabs(5)+"height:100vh;\n"+misc.tabs(3)+"}\n"+misc.tabs(3)+"</style>\n\t</head>\n\t<body>\n")
        else:
            self.file_html.write("\n\t\t<style>\n"+misc.tabs(3)+"x3d\n"+misc.tabs(4)+"{\n"+misc.tabs(5)+"border:2px solid darkorange;\n"+misc.tabs(5)+"width:95vw;\n"+misc.tabs(5)+"height: 75vh;\n"+misc.tabs(3)+"}\n"+misc.tabs(3)+"</style>\n\t</head>\n\t<body>\n")
        self.file_html.write(f'\t<h1 align="middle"> ViSL3D: 3D model of {pagetitle} </h1>\n')
        self.file_html.write('\t<hr/>\n')
        if description is not None:
            self.file_html.write(f"\t<p>\n\t {description}</p> \n")
        
        self.file_html.write(misc.roundto)

        #ANOTHER WAY TO CHANGE TRANSPARENCY instead of loading()
        # self.file_html.write(misc.tabs(3)+"const nl = [%s,%s,%s];"%())
        # self.file_html.write(misc.tabs(3)+"for (let nc = 0; nc < %s; nc++) {\n"%len(l_isolevels))
        # self.file_html.write(misc.tabs(4)+"for (let nl = 0; nl < ; nl++) {\n"%len(l_isolevels[nc]))
        # self.file_html.write(misc.tabs(5)+"if (nl === %s) \n"+misc.tabs(6)+"const op = 0.4;\n"+misc.tabs(5)+"else\n"+misc.tabs(6)+"const op = 0.8;\n")
        # self.file_html.write(misc.tabs(5)+"document.getElementById('cube__'+nc+'layer'+nl).setAttribute('transparency', op);\n")

        # if self.style == 'opaque':
        #     self.file_html.write(misc.tabs(1)+'<script>\n')
        #     self.file_html.write(misc.tabs(2)+'function loading() {\n')
        #     numcubes = len(l_isolevels)
        #     for nc in range(numcubes):
        #         isolevels = l_isolevels[nc]
        #         for nl in range(len(isolevels)):
        #             if nl == len(isolevels)-1:
        #                 op = 0.4
        #             else:
        #                 op = 0.8
        #             for sp in range(self.iso_split[nc][nl]):
        #                 self.file_html.write(misc.tabs(3)+"document.getElementById('cube__%slayer%s_sp%s').setAttribute('transparency', '%s');\n"%(nc,nl,sp,op))    
        #    self.file_html.write(misc.tabs(2)+'}\n')
        #    self.file_html.write(misc.tabs(1)+'</script>\n')

        # setTimeout(loading, 5000); option to execute function after time

    def func_layers(self):
        """
        Make JS funcion to hide/show layers.
        """
        numcubes = len(self.cube.l_isolevels)
        nlayers = [len(l) for l in self.cube.l_isolevels]
        self.file_html.write(misc.tabs(2)+"<script>\n")
        for nc in range(numcubes):
            self.file_html.write(misc.tabs(3)+"function hideall%s() {\n"%nc)
            for i in range(nlayers[nc]):
                self.file_html.write(f'{misc.tabs(4)}setHI{nc}layer{i}();\n')
            self.file_html.write(misc.tabs(3)+"}\n")

        self.file_html.write(misc.tabs(3)+"var showalertopa = true;\n")
        self.file_html.write(misc.tabs(3)+"function changeopa() {\n")
        self.file_html.write(misc.tabs(4)+"if (showalertopa) {\n")
        self.file_html.write(misc.tabs(5)+'alert("The opacity feature does not work alongside the 2D image. Hide the 2D image to change the opacity.")\n')
        self.file_html.write(misc.tabs(5)+"showalertopa = false;\n")
        self.file_html.write(misc.tabs(4)+"}\n")
        self.file_html.write(misc.tabs(4)+f'const nlayers = {nlayers};\n')
        self.file_html.write(misc.tabs(4)+'if (document.getElementById("cube__0layer0_sp0_appe").getAttribute("sortType") == "opaque") {\n')
        self.file_html.write(misc.tabs(5)+'for (let nc = 0; nc < %s; nc++) {\n'%numcubes)
        self.file_html.write(misc.tabs(6)+'for (let i = 0; i < nlayers[nc]; i++) {\n')
        self.file_html.write(misc.tabs(7)+f'document.getElementById("cube__"+nc+"layer"+i+"_sp0_appe").setAttribute("sortType", "transparent");\n')
        self.file_html.write(misc.tabs(6)+"}\n")
        self.file_html.write(misc.tabs(5)+"}\n")
        self.file_html.write(misc.tabs(4)+'} else {\n')
        self.file_html.write(misc.tabs(5)+'for (let nc = 0; nc < %s; nc++) {\n'%numcubes)
        self.file_html.write(misc.tabs(6)+'for (let i = 0; i < nlayers[nc]; i++) {\n')
        self.file_html.write(misc.tabs(7)+f'document.getElementById("cube__"+nc+"layer"+i+"_sp0_appe").setAttribute("sortType", "opaque");\n')
        self.file_html.write(misc.tabs(6)+"}\n")
        self.file_html.write(misc.tabs(5)+"}\n")
        self.file_html.write(misc.tabs(4)+"}\n")
        self.file_html.write(misc.tabs(3)+"}\n")

        self.file_html.write(misc.tabs(2)+"</script>\n")

        for nc in range(numcubes):
            for i in range(nlayers[nc]):
                self.file_html.write("\t <script>\n\t \t function setHI%slayer%s()\n\t \t {\n\t \t if(document.getElementById('cube__%slayer%s_sp0').getAttribute('transparency') != '0.8') {\n"%(nc,i,nc,i))
                self.file_html.write(f"\t\t document.getElementById('{nc}but{i}').style.border = '5px dashed black';\n")
                for sp in range(self.cube.iso_split[nc][i]):
                    self.file_html.write(f"\t\t document.getElementById('cube__{nc}layer{i}_sp{sp}').setAttribute('transparency', '0.8');\n")
                    self.file_html.write(f"\t\t document.getElementById('cube__{nc}layer{i}_sp{sp}_shape').setAttribute('ispickable', 'true');\n")
                self.file_html.write("\t\t } else { \n")
                self.file_html.write(f"\t\t document.getElementById('{nc}but{i}').style.border = 'inset black';\n")
                for sp in range(self.cube.iso_split[nc][i]):
                    self.file_html.write(f"\t\t document.getElementById('cube__{nc}layer{i}_sp{sp}').setAttribute('transparency', '1');\n")
                    self.file_html.write(f"\t\t document.getElementById('cube__{nc}layer{i}_sp{sp}_shape').setAttribute('ispickable', 'false');\n")
                self.file_html.write("\t\t } \n\t\t }\n\t </script>\n")

    def func_galaxies(self):
        """
        Make JS function to hide/show galaxies and their labels.
        The X3D file must have galaxies for this to work.
        Must be after buttons()
        """
        for i,gal in enumerate(self.cube.galaxies):
            if i == 0:
                self.file_html.write("\t \t <script>\n\t \t function setgals()\n\t \t {\n\t \t if(document.getElementById('cube__%s').getAttribute('transparency')!= '0'){\n"%gal)
            # self.file_html.write("\t \t document.getElementById('cube__%s_cross').setAttribute('transparency', '0');\n"%gal)
            self.file_html.write("\t \t document.getElementById('cube__%s').setAttribute('transparency', '0');\n"%gal)
        self.file_html.write("\t \t }\n\t \t else {\n")
        for i,gal in enumerate(self.cube.galaxies):
            # self.file_html.write("\t \t document.getElementById('cube__%s_cross').setAttribute('transparency', '1');\n"%gal)
            self.file_html.write("\t \t document.getElementById('cube__%s').setAttribute('transparency', '1');\n"%gal)
        self.file_html.write("\t\t }\n\t\t }\n\t\t </script>\n")

        for i,gal in enumerate(self.cube.galaxies):
            if i == 0:
                self.file_html.write("\t\t <script>\n\t \t function setgallabels()\n\t \t {\n\t \t if(document.getElementById('cube__label_%s').getAttribute('transparency')!= '0'){\n"%gal)
            self.file_html.write("\t \t document.getElementById('cube__label_%s').setAttribute('transparency', '0');\n"%gal)
        self.file_html.write("\t \t }\n\t \t else {\n")
        for i,gal in enumerate(self.cube.galaxies):
            self.file_html.write("\t \t document.getElementById('cube__label_%s').setAttribute('transparency', '1');\n"%gal)
        self.file_html.write("\t\t }\n\t\t }\n\t\t </script>\n")
            

    def func_grids(self):
        """
        Make JS function to hide/show grids.
        """
        self.file_html.write(misc.tabs(2)+"<script>\n\t\tfunction setgrids()\n\t\t{\n")
        self.file_html.write(misc.tabs(3)+"if(document.getElementById('cube__ticklines').getAttribute('transparency') == '0') {\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('cube__ticklines').setAttribute('transparency', '1');\n")
        self.file_html.write(misc.tabs(3)+"} else if (document.getElementById('cube__outline').getAttribute('transparency') == '0') {\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('cube__outline').setAttribute('transparency', '1');\n")
        self.file_html.write(misc.tabs(3)+"} else {\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('cube__ticklines').setAttribute('transparency', '0');\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('cube__outline').setAttribute('transparency', '0');\n")
        self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(2)+"}\n\t\t </script>\n")

    def func_axes(self):
        """
        Make JS function to hide/show axes labels.
        """
        self.file_html.write(misc.tabs(2)+"<script>\n")
        self.file_html.write(misc.tabs(2)+"function setaxes()\n")
        self.file_html.write(misc.tabs(2)+"{\n")
        self.file_html.write(misc.tabs(2)+"if(document.getElementById('cube__axlab_diff1').getAttribute('transparency') == '0') {\n")
        self.file_html.write(misc.tabs(3)+"for (i=0; i<12; i++) {\n")
        self.file_html.write(misc.tabs(3)+"if (i<6) {\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('cube__axlab_diff'+i).setAttribute('transparency', '1');\n")
        if isinstance(self.cube.lines, dict) == False:
            self.file_html.write(misc.tabs(4)+"document.getElementById('cube__axlab_real'+i).setAttribute('transparency', '0');\n")
        self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('cube__axtick_diff'+i).setAttribute('transparency', '1');\n")
        if isinstance(self.cube.lines, dict) == False:
            self.file_html.write(misc.tabs(4)+"document.getElementById('cube__axtick_real'+i).setAttribute('transparency', '0');\n")
        self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(2)+"}\n")
        if isinstance(self.cube.lines, dict) == False:
            self.file_html.write(misc.tabs(2)+"else if (document.getElementById('cube__axlab_real1').getAttribute('transparency') == '0') {\n")
            self.file_html.write(misc.tabs(3)+"for (i=0; i<12; i++) {\n")
            self.file_html.write(misc.tabs(3)+"if (i<6) {\n")
            self.file_html.write(misc.tabs(4)+"document.getElementById('cube__axlab_real'+i).setAttribute('transparency', '1');\n")
            self.file_html.write(misc.tabs(3)+"}\n")
            self.file_html.write(misc.tabs(4)+"document.getElementById('cube__axtick_real'+i).setAttribute('transparency', '1');\n")
            self.file_html.write(misc.tabs(3)+"}\n")
            self.file_html.write(misc.tabs(2)+"}\n")
        self.file_html.write(misc.tabs(2)+"else {\n")
        self.file_html.write(misc.tabs(3)+"for (i=0; i<12; i++) {\n")
        self.file_html.write(misc.tabs(3)+"if (i<6) {\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('cube__axlab_diff'+i).setAttribute('transparency', '0');\n")
        self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('cube__axtick_diff'+i).setAttribute('transparency', '0');\n")
        self.file_html.write(misc.tabs(3)+"}\n")                
        self.file_html.write("\t\t }\n\t\t }\n\t\t </script>\n")

    def func_pick(self):
        """
        Allows picking the coordinates by clicking in the figure.
        WORKS WITH VIEWPOINT NOT WITH ORTHOVIEWPOINT.
        NOT FINISHED, DON'T USE.

        """
        # self.file_html.write(misc.roundto) #premade string with function to round to two decimals
        # self.file_html.write(misc.tabs(1)+"<script>\n")
        # self.file_html.write(misc.tabs(3)+"const picksca = document.querySelector('#scalev');\n")
        # self.file_html.write(misc.tabs(2)+"function handleClick(event) {\n")
        # self.file_html.write(misc.tabs(3)+"const sca = picksca.value;\n")
        # self.file_html.write(misc.tabs(3)+"var coordinates = event.hitPnt;\n")
        # self.file_html.write(misc.tabs(3)+"$('#coordX').html(roundTo(coordinates[0], 2)+' %s');\n"%self.cube.units[1])
        # self.file_html.write(misc.tabs(3)+"$('#coordY').html(roundTo(coordinates[1], 2)+' %s');\n"%self.cube.units[2])
        # self.file_html.write(misc.tabs(3)+"$('#coordZ').html(roundTo(coordinates[2], 2)/sca+' %s');\n"%self.cube.units[3])
        # self.file_html.write(misc.tabs(2)+"}\n\t </script>\n")
        pass

    def func_animation(self):
        """
        Make JS function to start/stop the animation of the X3D models.
        """
        self.file_html.write('\n'+misc.tabs(1)+"<script>")
        self.file_html.write('\n'+misc.tabs(2)+"var active = false;")
        self.file_html.write('\n'+misc.tabs(2)+"function animation() {")
        self.file_html.write('\n'+misc.tabs(3)+"if (active == false) {")
        self.file_html.write('\n'+misc.tabs(4)+"document.getElementById('cube__time').setAttribute('startTime', document.getElementById('cube__time').getAttribute('time'));")
        self.file_html.write('\n'+misc.tabs(4)+"document.getElementById('cube__time').setAttribute('isPaused', 'false');")
        self.file_html.write('\n'+misc.tabs(4)+"active = true;")
        self.file_html.write('\n'+misc.tabs(3)+"} else if (document.getElementById('cube__time').getAttribute('isPaused') == 'false') {")
        self.file_html.write('\n'+misc.tabs(4)+"document.getElementById('cube__time').setAttribute('loop', 'false');")
        self.file_html.write('\n'+misc.tabs(4)+"document.getElementById('cube__time').setAttribute('isPaused', 'true')")
        self.file_html.write('\n'+misc.tabs(3)+"} else {")
        self.file_html.write('\n'+misc.tabs(4)+"document.getElementById('cube__time').setAttribute('loop', 'true');")
        self.file_html.write('\n'+misc.tabs(4)+"document.getElementById('cube__time').setAttribute('isPaused', 'false');")
        self.file_html.write('\n'+misc.tabs(3)+"}\n"+misc.tabs(2)+'}\n'+misc.tabs(1)+'</script>\n')

    def start_x3d(self):
        """
        Start the X3D part of the HTML. Must go before viewpoints() and close_x3d().
        """
        self.file_html.write(misc.tabs(1)+"<center><x3d id='cubeFixed'>\n")

    def close_x3d(self, filename):
        """
        Insert the X3D file and close the X3D part of the HTML.
        Must go after viewpoints() and start_x3d().

        Parameters
        ----------
        filename : string
            Name of the X3D file to be inserted.
        """
        filename = filename.split('.')[0]+'.x3d'
        # if self.hclick:
        #     self.file_html.write(misc.tabs(3)+'<inline url="%s" nameSpaceName="cube" mapDEFToID="true" onclick="handleClick(event)" onload="loading()"></inline>\n'%x3dname)
        # else:
        urlname = filename.split('/')[-1]
        urlname = urlname.split('\\')[-1]
        self.file_html.write(misc.tabs(3)+f'<inline url="{urlname}" nameSpaceName="cube" mapDEFToID="true" onclick="" onload="loading()"></inline>\n')
        self.file_html.write(misc.tabs(2)+"</scene>\n\t</x3d></center>\n")

    def viewpoints(self, point=None):
        """
        Define viewpoints for the X3D figure. Must go after start_x3d() and
        before close_x3d().

        Parameters
        ----------
        point : list, optional
            List of points to define viewpoints. The point will be used to create a vector from the origin.
            For example, [1,0,0] will create a viewpoint looking at the positive x-axis. The y axis is inverted.
        """
        self.file_html.write(misc.tabs(2)+"<scene>\n")
        #correct camera postition and FoV, not to clip (hide) the figure
        self.file_html.write(misc.tabs(3)+"<OrthoViewpoint id=\"front\" bind='false' centerOfRotation='0,0,0' description='RA-Dec view' fieldOfView='[-1400.0,-1400.0,1400.0,1400.0]' isActive='false' metadata='X3DMetadataObject' orientation='0,1,0,3.141593' position='0,0,-5500' zFar='11000' zNear='0.0001' ></OrthoViewpoint>\n")
        self.file_html.write(misc.tabs(3)+"<OrthoViewpoint id=\"side\" bind='false' centerOfRotation='0,0,0' description='Z - Dec view' fieldOfView='[-1400.0,-1400.0,1400.0,1400.0]' isActive='false' metadata='X3DMetadataObject' orientation='0,-1,0,1.570796' position='-5500,0,0' zFar='11000' zNear='0.0001' ></OrthoViewpoint>\n")
        self.file_html.write(misc.tabs(3)+"<OrthoViewpoint id=\"side2\" bind='false' centerOfRotation='0,0,0' description='Z - RA view' fieldOfView='[-1400.0,-1400.0,1400.0,1400.0]' isActive='false' metadata='X3DMetadataObject' orientation='1,1,1,4.1888' position='0,5500,0' zFar='11000' zNear='0.0001' ></OrthoViewpoint>\n")
        if point is not None:
            for i,po in enumerate(point):
                ori = misc.calc_axis_angle(po)
                pos = misc.calc_camera_position(ori)*5500
                self.file_html.write(misc.tabs(3)+f"<OrthoViewpoint id='vp{i}' bind='false' centerOfRotation='0,0,0' " \
                    +f"description='vp{i}' fieldOfView='[-1400.0,-1400.0,1400.0,1400.0]' isActive='false' " \
                    +f"orientation='{str(ori)[1:-1]}' position='{str(pos)[1:-1]}' zFar='11000' zNear='0.0001' ></OrthoViewpoint>\n")

    def func_background(self):
        """
        Make JS function to change the background color of the X3D figure.
        Must be after buttons()
        """
        self.file_html.write(misc.tabs(3)+"<script>\n")
        self.file_html.write(misc.tabs(4)+"function hex2Rgb(hex) {\n")
        self.file_html.write(misc.tabs(5)+r"var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);\n")
        self.file_html.write(misc.tabs(5)+"var r = parseInt(result[1], 16)/255.;\n")
        self.file_html.write(misc.tabs(5)+"var g = parseInt(result[2], 16)/255.;\n")
        self.file_html.write(misc.tabs(5)+"var b = parseInt(result[3], 16)/255.;\n")
        self.file_html.write(misc.tabs(5)+"return r.toString()+' '+g.toString()+' '+b.toString()\n")
        self.file_html.write(misc.tabs(4)+"}\n")
        self.file_html.write(misc.tabs(4)+"const background = document.querySelector('#back-choice');\n")
        self.file_html.write(misc.tabs(4)+"function change_background() {\n")
        self.file_html.write(misc.tabs(5)+"const backCol = background.value; \n")
        self.file_html.write(misc.tabs(5)+"document.getElementById('cube__back').setAttribute('skyColor', hex2Rgb(backCol));\n")
        self.file_html.write(misc.tabs(4)+"}\n")
        self.file_html.write(misc.tabs(3)+"</script>\n")

    def func_markers(self):
        """
        Create buttons to create markers for the X3D model interactively in the web page.
        Must be after buttons(). It is assumed that scalev is used. It will fail otherwise. 
        The JS functions are included by default in 'markers.js'.
        """
        self.file_html.write(misc.tabs(2) + '\n')
        self.file_html.write(misc.tabs(2)+"<div id=\"divmaster\" style=\"margin-left: 2%\">\n")
        self.file_html.write(misc.tabs(3)+"<br>\n")
        self.file_html.write(misc.tabs(3)+"<!-- BUTTON TO CHANGE LAYOUT -->\n")
        self.file_html.write(misc.tabs(3)+"<label for=\"markers-choice\"><b>Markers:</b> </label>\n")
        self.file_html.write(misc.tabs(3)+"<select id=\"markers-choice\">\n")
        self.file_html.write(misc.tabs(4)+"<option value=\"none\">None</option>\n")
        self.file_html.write(misc.tabs(4)+"<option value=\"sphere\">Sphere</option>\n")
        self.file_html.write(misc.tabs(4)+"<option value=\"box\">Box</option>\n")
        self.file_html.write(misc.tabs(4)+"<option value=\"tub\">Tube</option>\n")
        self.file_html.write(misc.tabs(4)+"<option value=\"con\">Cone</option>\n")
        self.file_html.write(misc.tabs(3)+"</select>\n")
        self.file_html.write(misc.tabs(3)+"<input type=\"color\" id=\"butcol\" value=\"#ff0000\">\n")
        self.file_html.write(misc.tabs(3)+"<button id=\"butnew\" onclick=\"newmarker()\">New</button>\n")
        self.file_html.write(misc.tabs(3)+"<button id=\"butcreate\" onclick=\"createmarker()\">Create</button>\n")
        self.file_html.write(misc.tabs(3)+"<button id=\"butremove\" onclick=\"removemarker()\">Remove</button>\n")
        self.file_html.write(misc.tabs(3)+"<button id=\"buthide\" onclick=\"hidemarker()\">Hide/Show</button> <br><br>\n")
        self.file_html.write(misc.tabs(2)+"</div>\n")
        self.file_html.write(misc.tabs(2)+"<!-- create various layouts for different objects -->\n")
        self.file_html.write(misc.tabs(2)+"<div id=\"spherediv\" style=\"display:none ; margin-left: 2%\">\n")
        self.file_html.write(misc.tabs(3)+"<select id=\"new-sphere\">\n")
        self.file_html.write(misc.tabs(4)+"<option value=\"none\">None</option>\n")
        self.file_html.write(misc.tabs(3)+"</select>\n")
        self.file_html.write(misc.tabs(3)+"<br>\n")
        self.file_html.write(misc.tabs(2)+"</div>\n")
        self.file_html.write(misc.tabs(2)+"<div id=\"boxdiv\" style=\"display:none ; margin-left: 2%\">\n")
        self.file_html.write(misc.tabs(3)+"<select id=\"new-box\">\n")
        self.file_html.write(misc.tabs(4)+"<option value=\"none\">None</option>\n")
        self.file_html.write(misc.tabs(3)+"</select>\n")
        self.file_html.write(misc.tabs(3)+"<br>\n")
        self.file_html.write(misc.tabs(2)+"</div>\n")
        self.file_html.write(misc.tabs(2)+"<div id=\"condiv\" style=\"display:none ; margin-left: 2%\">\n")
        self.file_html.write(misc.tabs(3)+"<select id=\"new-con\">\n")
        self.file_html.write(misc.tabs(4)+"<option value=\"none\">None</option>\n")
        self.file_html.write(misc.tabs(3)+"</select>\n")
        self.file_html.write(misc.tabs(3)+"<br>\n")
        self.file_html.write(misc.tabs(2)+"</div>\n")
        self.file_html.write(misc.tabs(2)+"<div id=\"tubdiv\" style=\"display:none ; margin-left: 2%\">\n")
        self.file_html.write(misc.tabs(3)+"<select id=\"new-tub\">\n")
        self.file_html.write(misc.tabs(4)+"<option value=\"none\">None</option>\n")
        self.file_html.write(misc.tabs(3)+"</select>\n")
        self.file_html.write(misc.tabs(3)+"<button id=\"addpoint\" onclick=\"addpoint(seltub, tubelen)\">Add point</button>\n")
        self.file_html.write(misc.tabs(3)+"<br>\n")
        self.file_html.write(misc.tabs(2)+"</div>\n")

        self.file_html.write(misc.tabs(2)+"<script type=\"text/javascript\">\n")
        self.file_html.write(misc.tabs(3)+"// General parameters\n")
        self.file_html.write(misc.tabs(3)+"const marktype = document.querySelector('#markers-choice');\n")
        self.file_html.write(misc.tabs(3)+"marktype.addEventListener('change', newlayout);\n")
        self.file_html.write(misc.tabs(3)+"const sscasv = document.querySelector('#scalev');\n")
        self.file_html.write(misc.tabs(3)+"const col = document.querySelector('#butcol');\n\n")
        self.file_html.write(misc.tabs(3)+"// Spheres\n")
        self.file_html.write(misc.tabs(3)+"var nspheres = 0; //number of spheres\n")
        self.file_html.write(misc.tabs(3)+"var sph_coords = []; //coordinates of sphere\n\n")
        self.file_html.write(misc.tabs(3)+"const selsph = document.querySelector('#new-sphere');\n")
        self.file_html.write(misc.tabs(3)+"selsph.addEventListener('change', changeSphere);\n\n")
        self.file_html.write(misc.tabs(3)+"// Boxes\n")
        self.file_html.write(misc.tabs(3)+"var nboxes = 0; //number of boxes\n")
        self.file_html.write(misc.tabs(3)+"var box_coords = []; //coordinates of boxes\n\n")
        self.file_html.write(misc.tabs(3)+"const selbox = document.querySelector('#new-box');\n")
        self.file_html.write(misc.tabs(3)+"selbox.addEventListener('change', changeBox);\n\n")
        self.file_html.write(misc.tabs(3)+"// Cones\n")
        self.file_html.write(misc.tabs(3)+"var ncones = 0; //number of cones\n")
        self.file_html.write(misc.tabs(3)+"var con_coords = []; //coordinates of cones\n\n")
        self.file_html.write(misc.tabs(3)+"const selcon = document.querySelector('#new-con');\n")
        self.file_html.write(misc.tabs(3)+"selcon.addEventListener('change', changeCon);\n\n")
        self.file_html.write(misc.tabs(3)+"// Tubes\n")
        self.file_html.write(misc.tabs(3)+"var ntubes = 0; //number of tubes\n")
        self.file_html.write(misc.tabs(3)+"var tub_coords = []; //coordinates of tubes\n")
        self.file_html.write(misc.tabs(3)+"var tubelen = []; //lengths of tubes in number of cylinders\n\n")
        self.file_html.write(misc.tabs(3)+"const seltub = document.querySelector('#new-tub');\n")
        self.file_html.write(misc.tabs(3)+"seltub.addEventListener('change', changeTub);\n\n")
        self.file_html.write(misc.tabs(3)+"// General\n")
        self.file_html.write(misc.tabs(3)+"function newmarker() {\n")
        self.file_html.write(misc.tabs(4)+"if (marktype.value == 'sphere') {\n")
        self.file_html.write(misc.tabs(5)+"nspheres = nspheres += 1;\n")
        self.file_html.write(misc.tabs(5)+"nspheres = newSphere(nspheres, selsph);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'box') {\n")
        self.file_html.write(misc.tabs(5)+"nboxes = nboxes += 1;\n")
        self.file_html.write(misc.tabs(5)+"nboxes = newBox(nboxes, selbox);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'tub') {\n")
        self.file_html.write(misc.tabs(5)+"ntubes = ntubes += 1;\n")
        self.file_html.write(misc.tabs(5)+"ntubes, tubelen = newTub(ntubes, seltub, tubelen);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'con') {\n")
        self.file_html.write(misc.tabs(5)+"ncones = ncones += 1;\n")
        self.file_html.write(misc.tabs(5)+"ncones = newCon(ncones, selcon);\n")
        self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(2)+"}\n\n")
        self.file_html.write(misc.tabs(3)+"function createmarker() {\n")
        self.file_html.write(misc.tabs(4)+"const sca = inpscasv.value;\n\n")
        self.file_html.write(misc.tabs(4)+f"const means = {list(np.mean(self.cube.coords, axis=1))};\n")
        delt = [np.abs(np.diff(self.cube.coords[0])/self.cube.l_cubes[0].shape[0]),
                 np.abs(np.diff(self.cube.coords[1])/self.cube.l_cubes[0].shape[1]),
                 np.abs(np.diff(self.cube.coords[2])/self.cube.l_cubes[0].shape[2])]
        delt = [factor[0] for factor in delt]
        self.file_html.write(misc.tabs(4)+f"const delt = {list(delt)};\n")
        trans = [2000/self.cube.l_cubes[0].shape[0], 2000/self.cube.l_cubes[0].shape[1], 2000/self.cube.l_cubes[0].shape[2]]
        self.file_html.write(misc.tabs(4)+f"const trans = {trans};\n")
        self.file_html.write(misc.tabs(4)+"if (marktype.value == 'sphere') {\n")
        self.file_html.write(misc.tabs(5)+f"sph_coords = createSphere(sca, selsph, col, sph_coords, means, delt, trans, html=false);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'box') {\n")
        self.file_html.write(misc.tabs(5)+"box_coords = createBox(sca, selbox, col, box_coords, means, delt, trans, html=false);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'tub') {\n")
        self.file_html.write(misc.tabs(5)+"tub_coords = createTub(sca, seltub, col, tub_coords, tubelen, means, delt, trans, html=false);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'con') {\n")
        self.file_html.write(misc.tabs(5)+"con_coords = createCon(sca, selcon, col, con_coords, means, delt, trans, html=false);\n")
        self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(2)+"}\n\n")
        self.file_html.write(misc.tabs(3)+"function removemarker() {\n")
        self.file_html.write(misc.tabs(4)+"if (marktype.value == 'sphere') {\n")
        self.file_html.write(misc.tabs(5)+"removeSphere(selsph);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'box') {\n")
        self.file_html.write(misc.tabs(5)+"removeBox(selbox);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'tub') {\n")
        self.file_html.write(misc.tabs(5)+"removeTub(seltub, tubelen);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'con') {\n")
        self.file_html.write(misc.tabs(5)+"removeCon(selcon);\n")
        self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(2)+"}\n")
        # hidemarker
        self.file_html.write(misc.tabs(3)+"function hidemarker() {\n")
        self.file_html.write(misc.tabs(4)+"if (marktype.value == 'sphere') {\n")
        self.file_html.write(misc.tabs(5)+"hideSphere(selsph);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'box') {\n")
        self.file_html.write(misc.tabs(5)+"hideBox(selbox);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'tub') {\n")
        self.file_html.write(misc.tabs(5)+"hideTub(seltub, tubelen);\n")
        self.file_html.write(misc.tabs(4)+"} else if (marktype.value == 'con') {\n")
        self.file_html.write(misc.tabs(5)+"hideCon(selcon);\n")
        self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(2)+"}\n")
        self.file_html.write(misc.tabs(2)+"</script>\n")

    def func_pymarkers(self, tube=None, sphere=None, box=None, cone=None):
        """
        JS function to hide/show markers created with make_markers() directly to the X3D model.

        Parameters
        ----------
        tube : list
            List with the points parameter included in make_markers() for tubes.
        sphere : list
            List with the points parameter included in make_markers() for spheres.
        box : list
            List with the points parameter included in make_markers() for boxes.
        cone : list
            List with the points parameter included in make_markers() for cones.
        """
        self.file_html.write(misc.tabs(2)+"<script>\n")
        if tube is not None:
            self.file_html.write(misc.tabs(3)+"function settube() {\n")
            self.file_html.write(misc.tabs(4)+"if(document.getElementById('cube__py0tubmat0').getAttribute('transparency') != '0') {\n")
            for t in range(len(tube)):
                for i in range(len(tube[t])-1):
                    self.file_html.write(misc.tabs(5)+"document.getElementById('cube__py%stubmat%s').setAttribute('transparency', '0');\n"%(t,i))
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__label_pytub%s').setAttribute('transparency', '0');\n"%t)
            self.file_html.write(misc.tabs(4)+"} else {\n")
            for t in range(len(tube)):
                for i in range(len(tube[t])-1):
                    self.file_html.write(misc.tabs(5)+"document.getElementById('cube__py%stubmat%s').setAttribute('transparency', '1');\n"%(t,i))
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__label_pytub%s').setAttribute('transparency', '1');\n"%t)
            self.file_html.write(misc.tabs(4)+"}\n")
            self.file_html.write(misc.tabs(3)+"}\n")
        if sphere is not None:
            self.file_html.write(misc.tabs(3)+"function setsphere() {\n")
            self.file_html.write(misc.tabs(4)+"if(document.getElementById('cube__py0sphmat').getAttribute('transparency') != '0') {\n")
            for i in range(len(sphere)):
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__py%ssphmat').setAttribute('transparency', '0');\n"%i)
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__label_pysph%s').setAttribute('transparency', '0');\n"%i)
            self.file_html.write(misc.tabs(4)+"} else {\n")
            for i in range(len(sphere)):
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__py%ssphmat').setAttribute('transparency', '1');\n"%i)
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__label_pysph%s').setAttribute('transparency', '1');\n"%i)
            self.file_html.write(misc.tabs(4)+"}\n")
            self.file_html.write(misc.tabs(3)+"}\n")
        if box is not None:
            self.file_html.write(misc.tabs(3)+"function setbox() {\n")
            self.file_html.write(misc.tabs(4)+"if(document.getElementById('cube__py0boxmat').getAttribute('transparency') != '0') {\n")
            for i in range(len(box)):
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__py%sboxmat').setAttribute('transparency', '0');\n"%i)
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__label_pybox%s').setAttribute('transparency', '0');\n"%i)
            self.file_html.write(misc.tabs(4)+"} else {\n")
            for i in range(len(box)):
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__py%sboxmat').setAttribute('transparency', '1');\n"%i)
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__label_pybox%s').setAttribute('transparency', '1');\n"%i)
            self.file_html.write(misc.tabs(4)+"}\n")
            self.file_html.write(misc.tabs(3)+"}\n")
        if cone is not None:
            self.file_html.write(misc.tabs(3)+"function setcone() {\n")
            self.file_html.write(misc.tabs(4)+"if(document.getElementById('cube__py0conmat').getAttribute('transparency') != '0') {\n")
            for i in range(len(cone)):
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__py%sconmat').setAttribute('transparency', '0');\n"%i)
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__label_pycon%s').setAttribute('transparency', '0');\n"%i)
            self.file_html.write(misc.tabs(4)+"} else {\n")
            for i in range(len(cone)):
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__py%sconmat').setAttribute('transparency', '1');\n"%i)
                self.file_html.write(misc.tabs(5)+"document.getElementById('cube__label_pycon%s').setAttribute('transparency', '1');\n"%i)
            self.file_html.write(misc.tabs(4)+"}\n")
            self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(2)+"</script>\n")


    def buttons(self, viewpoint=None, tube=None, sphere=None, box=None, cone=None, centrot=False):
        """
        Makes the buttons to apply different functions in the web page.

        Parameters
        ----------
        viewpoint : list
            List of names for the viewpoints. Viewpoints must be created in "viewpoints()".
        tube : list
            List with the points parameter included in make_markers() for tubes.
        sphere : list
            List with the points parameter included in make_markers() for spheres.
        box : list
            List with the points parameter included in make_markers() for boxes.
        cone : list
            List with the points parameter included in make_markers() for cones.
        centrot : bool
            Create button to change the center of rotation.
        """
        self.file_html.write(misc.tabs(1)+'<div style="width:90%">\n')
        self.file_html.write(misc.tabs(2)+'<br/>\n')
        # Viewpoint buttons
        self.file_html.write(misc.tabs(2)+"&nbsp <b>Viewpoints:</b>\n")
        self.file_html.write(misc.tabs(3)+"<button onclick=\"document.getElementById('cubeFixed').runtime.resetView();\">Reset View</button>\n")
        for i in range(3): # 3 w/o perspective, 4 with
            self.file_html.write(misc.tabs(3)+"<button onclick=\"document.getElementById('%s').setAttribute('set_bind','true');\"> %s </button>\n"%(misc.side[i], misc.nam[i]))
        if viewpoint is not None:
            for i,name in enumerate(viewpoint):
                self.file_html.write(misc.tabs(3)+f"<button onclick=\"document.getElementById('vp{i}').setAttribute('set_bind','true');\"> {name} </button>\n")

        #unccoment next line for next view button
        #self.file_html.write('\t\t   <button onclick="document.getElementById(\'cubeFixed\').runtime.nextView();">Next View</button>\n')
        
        # Grids and ax labels
        self.file_html.write('\n'+misc.tabs(2)+'<br><br>\n')
        self.file_html.write('\n'+misc.tabs(2)+'&nbsp <b>Labels:</b>\n')
        self.file_html.write(misc.tabs(3)+'<button onclick="setgrids();" >Grids</button>\n')
        self.file_html.write(misc.tabs(3)+'<button onclick="setaxes();" >Axes labels</button>\n')

        if self.cube.galaxies is not None:
            self.file_html.write(misc.tabs(3)+'<button onclick="setgals();" >Galaxies</button>\n')
            self.file_html.write(misc.tabs(3)+'<button onclick="setgallabels();" >Galaxy Labels</button>\n')

        if self.cube.image2d[1] is not None:
            self.file_html.write(misc.tabs(3)+'<button onclick="setimage2d();" >2D image</button>\n')
        
        if centrot:
            self.file_html.write(misc.tabs(2)+'&nbsp <label for="rotationCenter"><b>Center of Rotation</b> </label>\n')
            self.file_html.write(misc.tabs(3)+'<select id="rotationCenter">\n')
            self.file_html.write(misc.tabs(4)+'<option value="Origin">Origin</option>\n')
            for nc in range(len(self.cube.l_isolevels)):
                if isinstance(self.cube.lines, dict) or isinstance(self.cube.lines, list):
                    llab = list(self.cube.lines)[nc]
                    self.file_html.write(misc.tabs(4)+'<option value="op%s">%s</option>\n'%(nc,llab))
                else:
                    self.file_html.write(misc.tabs(4)+'<option value="op%s">Center%s</option>\n'%(nc,nc))
            self.file_html.write(misc.tabs(3)+'</select>\n')
            
        self.file_html.write(misc.tabs(3)+'<button id="anim" onclick="animation()">Animation</button>')

        if tube is not None or sphere is not None or box is not None or cone is not None:
            self.file_html.write(misc.tabs(2)+'<br><br>\n')
            self.file_html.write(misc.tabs(2)+'&nbsp <b>Markers:</b>\n')
            if tube is not None:
                self.file_html.write(misc.tabs(3)+'<button onclick="settube();">Tubes</button>\n')
            if sphere is not None:
                self.file_html.write(misc.tabs(3)+'<button onclick="setsphere();">Spheres</button>\n')
            if box is not None:
                self.file_html.write(misc.tabs(3)+'<button onclick="setbox();">Boxes</button>\n')
            if cone is not None:
                self.file_html.write(misc.tabs(3)+'<button onclick="setcone();">Cones</button>\n')
        
        nlayers = [len(l) for l in self.cube.l_isolevels]
        numcubes = len(nlayers)

        for nc in range(numcubes):
            self.file_html.write(misc.tabs(2)+'<br><br>\n')
            rgbcolors = misc.create_colormap(self.cube.cmaps[nc], self.cube.l_isolevels[nc])
            if isinstance(self.cube.lines, dict):
                llab = list(self.cube.lines)[nc]
                llab = f'{llab} ({self.cube.lines[llab][0]})'
                self.file_html.write(misc.tabs(2)+'&nbsp <b>%s (%s):</b>\n'%(llab,self.cube.units[0]))
            elif isinstance(self.cube.lines, list):
                llab = list(self.cube.lines)[nc]
                self.file_html.write(misc.tabs(2)+'&nbsp <b>%s (%s):</b>\n'%(llab,self.cube.units[0]))
            else:
                self.file_html.write(misc.tabs(2)+'&nbsp <b>Cube %s (%s):</b>\n'%(nc,self.cube.units[0]))
            for i in range(nlayers[nc]):
                ca = np.array(rgbcolors[i].split(' ')).astype(float)*255
                c = 'rgb('+str(ca.astype(int))[1:-1]+')'
                butlabel = self.cube.l_isolevels[nc][i]
                if self.cube.l_isolevels[nc][i] > 0.01:
                    butlabel = f'{butlabel:0.3f}'
                else:
                    butlabel = f'{butlabel:0.3e}'
                if (ca[0]*0.299 + ca[1]*0.587 + ca[2]*0.114) > 130:
                    self.file_html.write(misc.tabs(3)+'<button id="%sbut%s" onclick="setHI%slayer%s();" style="font-size:20px ; border:5px dashed black ; background:%s ; color:black"><b>%s</b></button>\n'%(nc,i,nc,i,c,butlabel))
                else:
                    self.file_html.write(misc.tabs(3)+'<button id="%sbut%s" onclick="setHI%slayer%s();" style="font-size:20px ; border:5px dashed black ; background:%s ; color:white"><b>%s</b></button>\n'%(nc,i,nc,i,c,butlabel))
            self.file_html.write(misc.tabs(3)+f'&nbsp <button id="all" onclick="hideall{nc}()"><b>Invert</b></button>')
                    
        # to separate buttons in two parts
        #if self.grids or self.gals or self.gallabs or self.axes or self.hclick or colormaps is not None:
            #self.file_html.write('\n\t <div style="position:absolute;left:800px;top:140px;width:600px">\n')
        
        # Colormaps
        for nc in range(numcubes):
            self.file_html.write(misc.tabs(2)+'<br><br>\n')
            if isinstance(self.cube.lines, dict) or isinstance(self.cube.lines, list):
                llab = list(self.cube.lines)[nc]
                self.file_html.write(misc.tabs(2)+'&nbsp <label for="cmaps-choice%s"><b>Cmap %s</b> </label>\n'%(nc,llab))
            else:
                self.file_html.write(misc.tabs(2)+'&nbsp <label for="cmaps-choice%s"><b>Cmap %s</b> </label>\n'%(nc,nc))
            self.file_html.write(misc.tabs(2)+'<select id="cmaps-choice%s">\n'%nc)
            self.file_html.write(misc.tabs(3)+f'<option value="{self.cube.cmaps[nc]}">{self.cube.cmaps[nc]}</option>\n')
            for c in misc.default_cmaps:
                if c != self.cube.cmaps[nc]:
                    self.file_html.write(misc.tabs(3)+'<option value="%s">%s</option>\n'%(c,c))
            self.file_html.write(misc.tabs(2)+'</select>\n')
            self.file_html.write(misc.tabs(2) + '<label for="cmaps-min%s"><b>Min %s:</b> </label>\n'%(nc,nc))
            self.file_html.write(misc.tabs(2) + '<input id="cmaps-min%s" type="number" value="%s">\n'%(nc, np.min(self.cube.l_isolevels[nc])))
            self.file_html.write(misc.tabs(2) + '<label for="cmaps-max%s"><b>Max %s:</b> </label>\n'%(nc,nc))
            self.file_html.write(misc.tabs(2) + '<input id="cmaps-max%s" type="number" value="%s">\n'%(nc, np.max(self.cube.l_isolevels[nc])))
            self.file_html.write(misc.tabs(2) + '<label for="cmaps-min%s"><b>Scale %s:</b> </label>\n'%(nc,nc))
            self.file_html.write(misc.tabs(2) + '<select id="cmaps-scale%s">\n'%nc)
            self.file_html.write(misc.tabs(3) + '<option value="linear" selected="selected">linear</option>\n')
            self.file_html.write(misc.tabs(3) + '<option value="log">log</option>\n')
            self.file_html.write(misc.tabs(3) + '<option value="sqrt">sqrt</option>\n')
            self.file_html.write(misc.tabs(3) + '<option value="power">power</option>\n')
            self.file_html.write(misc.tabs(3) + '<option value="asinh">asinh</option>\n')
            self.file_html.write(misc.tabs(2) + '</select>\n')

        # Opacity
        self.file_html.write('\n'+misc.tabs(2)+'<br><br>\n')
        self.file_html.write(misc.tabs(3)+f'&nbsp <button id="all" onclick="changeopa()"><b>Opacity</b></button>')

        # Background 
        self.file_html.write(misc.tabs(3)+'&nbsp <label for="back-choice"><b>Background:</b> </label>\n')
        self.file_html.write(misc.tabs(3)+'<input oninput="change_background()" id="back-choice" type="color" value="#999999">\n')

        # SCALEV
        #self.file_html.write(misc.tabs(2)+'<br><br>\n')
        self.file_html.write(misc.tabs(2)+'&nbsp <label for="scalev"><b>Z scale:</b> </label>\n')
        self.file_html.write(misc.tabs(2)+'<input oninput="changescalev()" id="scalev" type="range" list="marker" min="0" max="10" step="0.001" value="1"></input>\n')
        self.file_html.write(misc.tabs(2)+'<datalist id="marker">\n')
        self.file_html.write(misc.tabs(3)+'<option value="1"></option>\n')
        self.file_html.write(misc.tabs(2)+'</datalist>\n')
        
        # Galaxies Font Size
        if self.cube.galaxies is not None:
            self.file_html.write(misc.tabs(3)+'&nbsp <label for="galsize-choice"><b>Galaxy size: </b></label>\n')
            self.file_html.write(misc.tabs(3)+'<input oninput="change_galsize()" id="galsize-choice" type="number" min="2" max="50" value="5", step="1">\n')
            
        if self.cube.image2d[1] is not None:
            #self.file_html.write('\t\t <br><br>\n')
            self.file_html.write(misc.tabs(2)+'&nbsp <label for="move2dimg"><b>2D image:</b> </label>\n')
            self.file_html.write(misc.tabs(2)+'<input oninput="move2d()" id="move2dimg" type="range" min="-1" max="1" step="0.0001" value="1"></input>\n')
            self.file_html.write(misc.tabs(2)+f'<b>$Z=$</b> <output id="showvalue"></output> {self.cube.units[3]}\n')
            # display chosen velocity of bar too

        # PICKING
        # self.file_html.write(misc.tabs(2)+'<br><br>\n')
        # self.file_html.write(misc.tabs(2)+'&nbsp <label for="clickcoords"><b>Click coordinates:</b></label>\n')
        # self.file_html.write(misc.tabs(2)+'<table id="clickcoords">\n\t\t <tbody>\n')
        # self.file_html.write(misc.tabs(3)+'<tr><td>&nbsp RA: </td><td id="coordX">--</td></tr>\n')
        # self.file_html.write(misc.tabs(3)+'<tr><td>&nbsp Dec: </td><td id="coordY">--</td></tr>\n')
        # self.file_html.write(misc.tabs(3)+'<tr><td>&nbsp V: </td><td id="coordZ">--</td></tr>\n')
        # self.file_html.write(misc.tabs(2)+'</tbody></table>\n')
            
        self.file_html.write(misc.tabs(1)+'</div>\n')
        
    def func_move2dimage(self):
        """
        Make JS function to move the 2D image along the spectral axis.
        The X3D file must have a 2D image for this to work.
        Must be after buttons()
        """
        self.file_html.write(misc.tabs(2)+"<script>\n")
        self.file_html.write(misc.tabs(2)+"const inpscam2 = document.querySelector('#scalev');\n")
        self.file_html.write(misc.tabs(2)+"const inpmovem2 = document.querySelector('#move2dimg');\n")
        self.file_html.write(misc.tabs(2)+"const showval = document.querySelector('#showvalue');\n")
        self.file_html.write(misc.tabs(2)+"function move2d()\n\t\t{\n")
        self.file_html.write(misc.tabs(3)+"const sca = inpscam2.value;\n")
        self.file_html.write(misc.tabs(3)+"const move = inpmovem2.value;\n")
        self.file_html.write(misc.tabs(3)+f"showval.textContent = roundTo({self.cube.coords[2][1]}"+\
                             f"+(move-1)*{self.cube.l_cubes[0].shape[2]}*{np.abs(self.cube.delta[2])}/2, 3);\n")
        self.file_html.write(misc.tabs(3)+"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(sca*move-1)*1000);\n")
        self.file_html.write(misc.tabs(2)+"}\n")
        self.file_html.write(misc.tabs(2)+"</script>\n")

    def func_galsize(self):
        """
        Make JS function to change the size of the galaxy markers and their labels.
        The X3D file must have galaxies for this to work.
        Must be after buttons()
        """
        self.file_html.write(misc.tabs(2)+"<script>\n")
        self.file_html.write(misc.tabs(3)+"const galsize = document.querySelector('#galsize-choice');\n")
        self.file_html.write(misc.tabs(3)+"function change_galsize() {\n")
        self.file_html.write(misc.tabs(4)+"let gals = %s;\n"%str(list(self.cube.galaxies.keys())))
        self.file_html.write(misc.tabs(4)+"const gsval = galsize.value/8.;\n")
        self.file_html.write(misc.tabs(4)+"for (const gal of gals) {\n")
        self.file_html.write(misc.tabs(5)+"document.getElementById('cube__'+gal+'_sphere_tra').setAttribute('scale', gsval.toString()+' '+gsval.toString()+' '+gsval.toString());\n")
        self.file_html.write(misc.tabs(5)+"document.getElementById('cube__'+gal+'_fs').setAttribute('size', gsval*8);\n")
        self.file_html.write(misc.tabs(5)+"}\n")
        self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(2)+"</script>\n")

    def func_setCenterOfRotation(self, centers):
        """
        Make JS function to change the center of rotation from given options.
        Must be after buttons()

        Parameters
        ----------
        centers : list
            List of strings with the coordinates (as different from the centre) of the centers of rotation to be added. E.g. ["0 10 0", "10 0 10"]
        """
        self.file_html.write(misc.tabs(2)+"<script>\n")
        self.file_html.write(misc.tabs(3)+"cor = document.querySelector('#rotationCenter');\n")
        self.file_html.write(misc.tabs(3)+"cor.addEventListener('change', setCenterOfRotation);\n")
        self.file_html.write(misc.tabs(2)+"function setCenterOfRotation() {\n")
        self.file_html.write(misc.tabs(3)+"const cen = cor.value;\n")
        self.file_html.write(misc.tabs(3)+"if (cen === 'Origin') {\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('front').setAttribute('centerOfRotation','0 0 0');\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('side').setAttribute('centerOfRotation','0 0 0');\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('side2').setAttribute('centerOfRotation','0 0 0');\n")
        self.file_html.write(misc.tabs(3)+"}\n")
        for nc in range(len(centers)):
            self.file_html.write(misc.tabs(3)+"else if (cen === 'op%s') {\n"%nc)
            self.file_html.write(misc.tabs(4)+"document.getElementById('front').setAttribute('centerOfRotation', '%s');\n"%centers[nc])
            self.file_html.write(misc.tabs(4)+"document.getElementById('side').setAttribute('centerOfRotation', '%s');\n"%centers[nc])
            self.file_html.write(misc.tabs(4)+"document.getElementById('side2').setAttribute('centerOfRotation', '%s');\n"%centers[nc])
            self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(2)+"}\n"+misc.tabs(2)+"</script>\n")
        
    def func_colormaps(self):
        """
        Make JS function to change the colormap of the layers.
        Must be after buttons()
        """
        self.file_html.write("\t\t <!--MUST BE BELOW THE <select> ELEMENT-->\n")
        numcubes = len(self.cube.l_isolevels)
        for nc in range(numcubes):
            self.file_html.write(misc.tabs(2)+"<script>\n")
            self.file_html.write(misc.tabs(3)+"const cc%s = document.querySelector('#cmaps-choice%s');\n"%(nc,nc))
            self.file_html.write(misc.tabs(3)+"cc%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.file_html.write(misc.tabs(3)+"const cmapmin%s = document.querySelector('#cmaps-min%s');\n"%(nc,nc))
            self.file_html.write(misc.tabs(3)+"cmapmin%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.file_html.write(misc.tabs(3)+"const cmapmax%s = document.querySelector('#cmaps-max%s');\n"%(nc,nc))
            self.file_html.write(misc.tabs(3)+"cmapmax%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.file_html.write(misc.tabs(3)+"const cmapscale%s = document.querySelector('#cmaps-scale%s');\n"%(nc,nc))
            self.file_html.write(misc.tabs(3)+"cmapscale%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.file_html.write(misc.tabs(3)+"const isolevels%s = %s;\n"%(nc,repr(self.cube.l_isolevels[nc]).replace('array(','').replace(')','')))

            self.file_html.write(misc.tabs(3)+"function change_colormap%s() {\n"%nc)
            self.file_html.write(misc.tabs(4)+"var cmap%s = cc%s.value;\n"%(nc,nc))
            self.file_html.write(misc.tabs(4)+"var min%s = cmapmin%s.value;\n"%(nc,nc))
            self.file_html.write(misc.tabs(4)+"var max%s = cmapmax%s.value;\n"%(nc,nc))
            self.file_html.write(misc.tabs(4)+"const scale%s = cmapscale%s.value;\n"%(nc,nc))
            self.file_html.write(misc.tabs(4)+"var reverse%s = false;\n"%(nc))
            self.file_html.write(misc.tabs(4)+"var collevs%s = [];\n"%(nc))

            self.file_html.write(misc.tabs(4)+"if (scale%s === 'linear') {\n"%(nc))
            self.file_html.write(misc.tabs(5)+"for (const level of isolevels%s) {\n"%(nc))
            self.file_html.write(misc.tabs(6)+"collevs%s.push(level);\n"%(nc))
            self.file_html.write(misc.tabs(5)+"}\n")
            self.file_html.write(misc.tabs(4)+"}\n")
            self.file_html.write(misc.tabs(4)+"else if (scale%s === 'log') {\n"%(nc))
            self.file_html.write(misc.tabs(5)+"min%s = Math.log(min%s);\n"%(nc,nc))
            self.file_html.write(misc.tabs(5)+"max%s = Math.log(max%s);\n"%(nc,nc))
            self.file_html.write(misc.tabs(5)+"for (const level of isolevels%s) {\n"%(nc))
            self.file_html.write(misc.tabs(6)+"collevs%s.push(Math.log(level));\n"%nc)
            self.file_html.write(misc.tabs(5)+"}\n")
            self.file_html.write(misc.tabs(4)+"}\n")
            self.file_html.write(misc.tabs(4)+"else if (scale%s === 'sqrt') {\n"%(nc))
            self.file_html.write(misc.tabs(5)+"min%s = Math.sqrt(min%s);\n"%(nc,nc))
            self.file_html.write(misc.tabs(5)+"max%s = Math.sqrt(max%s);\n"%(nc,nc))
            self.file_html.write(misc.tabs(5)+"for (const level of isolevels%s) {\n"%(nc))
            self.file_html.write(misc.tabs(6)+"collevs%s.push(Math.sqrt(level));\n"%(nc))
            self.file_html.write(misc.tabs(5)+"}\n")
            self.file_html.write(misc.tabs(4)+"}\n")
            self.file_html.write(misc.tabs(4)+"else if (scale%s === 'power') {\n"%nc)
            self.file_html.write(misc.tabs(5)+"min%s = Math.pow(min%s, 2);\n"%(nc,nc))
            self.file_html.write(misc.tabs(5)+"max%s = Math.pow(max%s, 2);\n"%(nc,nc))
            self.file_html.write(misc.tabs(5)+"for (const level of isolevels%s) {\n"%nc)
            self.file_html.write(misc.tabs(6)+"collevs%s.push(Math.pow(level, 2));\n"%nc)
            self.file_html.write(misc.tabs(5)+"}\n")
            self.file_html.write(misc.tabs(4)+"}\n")
            self.file_html.write(misc.tabs(4)+"else if (scale%s === 'asinh') {\n"%nc)
            self.file_html.write(misc.tabs(5)+"min%s = Math.asinh(min%s);\n"%(nc,nc))
            self.file_html.write(misc.tabs(5)+"max%s = Math.asinh(max%s);\n"%(nc,nc))
            self.file_html.write(misc.tabs(5)+"for (const level of isolevels%s) {\n"%nc)
            self.file_html.write(misc.tabs(6)+"collevs%s.push(Math.asinh(level));\n"%nc)
            self.file_html.write(misc.tabs(5)+"}\n")
            self.file_html.write(misc.tabs(4)+"}\n")
            self.file_html.write(misc.tabs(4)+"for (let lev = 0; lev < isolevels%s.length; lev++) {\n"%nc)
            self.file_html.write(misc.tabs(5)+"if (min%s >= collevs%s[lev]) {\n"%(nc,nc))
            self.file_html.write(misc.tabs(6)+"collevs%s[lev] = 0;\n"%nc)
            self.file_html.write(misc.tabs(5)+"} else if (max%s <= collevs%s[lev]) {\n"%(nc,nc))
            self.file_html.write(misc.tabs(6)+"collevs%s[lev] = 1;\n"%nc)
            self.file_html.write(misc.tabs(5)+"} else {\n")
            self.file_html.write(misc.tabs(6)+"collevs%s[lev] = (collevs%s[lev] - min%s) / (max%s - min%s);\n"%(nc,nc,nc,nc,nc))
            self.file_html.write(misc.tabs(5)+"}\n")
            self.file_html.write(misc.tabs(5)+"if (cmap%s.endsWith('_r')) {\n"%nc)
            self.file_html.write(misc.tabs(6)+"reverse%s = true\n"%nc)
            self.file_html.write(misc.tabs(5)+"}\n")
            self.file_html.write(misc.tabs(5)+"if (cmap%s.includes('gist_rainbow')) {\n"%nc)
            self.file_html.write(misc.tabs(6)+"var color%s = evaluate_cmap(collevs%s[lev], 'gist_rainbow', reverse%s);\n"%(nc,nc,nc))
            self.file_html.write(misc.tabs(5)+"} else {\n")
            self.file_html.write(misc.tabs(6)+"var color%s = evaluate_cmap(collevs%s[lev], cmap%s.replace('_r', ''), reverse%s);\n"%(nc,nc,nc,nc))
            self.file_html.write(misc.tabs(5)+"}\n")
            self.file_html.write(misc.tabs(5)+"var split%s = %s;\n"%(nc,repr(self.cube.iso_split[nc]).replace('array(','').replace(')','')))
            self.file_html.write(misc.tabs(5)+"for (let sp = 0; sp < split%s[lev]; sp++) {\n"%nc)
            self.file_html.write(misc.tabs(6)+"document.getElementById('cube__%slayer'+lev+'_sp'+sp).setAttribute('diffuseColor', color%s[0]/255+' '+color%s[1]/255+' '+color%s[2]/255);\n"%(nc,nc,nc,nc))
            self.file_html.write(misc.tabs(5)+"}\n")
            self.file_html.write(misc.tabs(5)+"document.getElementById('%sbut'+lev).style.background = 'rgb('+color%s[0]+' '+color%s[1]+' '+color%s[2]+')';\n"%(nc,nc,nc,nc))
            self.file_html.write(misc.tabs(5)+"if ((color%s[0]*0.299 + color%s[1]*0.587 + color%s[2]*0.114) > 130) {\n"%(nc,nc,nc))
            self.file_html.write(misc.tabs(6)+"document.getElementById('%sbut'+lev).style.color = 'black';\n"%nc)
            self.file_html.write(misc.tabs(5)+"} else {\n")
            self.file_html.write(misc.tabs(6)+"document.getElementById('%sbut'+lev).style.color = 'white';\n"%nc)
            self.file_html.write(misc.tabs(5)+"}\n")
            self.file_html.write(misc.tabs(4)+"}\n")
            self.file_html.write(misc.tabs(3)+"}\n")
            self.file_html.write(misc.tabs(2)+"</script>\n")
            
            
    def func_scalev(self, sphere=None, box=None, cone=None, tube=None, trans=None):
        """
        Make JS funtion to change the scale of the spectral axis.
        Must be after buttons().

        Parameters
        ----------
        tube : list
            List with the points parameter included in make_markers() for tubes.
        sphere : list
            List with the points parameter included in make_markers() for spheres.
        box : list
            List with the points parameter included in make_markers() for boxes.
        cone : list
            List with the points parameter included in make_markers() for cones.
        delta : list
            List with the delta of the cube (header[CDELT]).
        trans : list
            Transformation factor to the cube. It is calculated with (2000/nx, 2000/ny, 2000/nz).
        """
        self.file_html.write(misc.tabs(2)+"<script>\n")
        self.file_html.write(misc.tabs(2)+"const inpscasv = document.querySelector('#scalev');\n")
        if self.cube.image2d[1] is not None:
            self.file_html.write(misc.tabs(2)+"const inpmovesv = document.querySelector('#move2dimg');\n")
        #self.file_html.write(misc.tabs(2)+"inpscasv.addEventListener('change', changescalev);\n")
        self.file_html.write(misc.tabs(2)+"function changescalev()\n\t\t {\n")
        self.file_html.write(misc.tabs(3)+"const sca = inpscasv.value;\n")
        if self.cube.image2d[1] is not None:
            self.file_html.write(misc.tabs(3)+"const move = inpmovesv.value;\n")
            self.file_html.write(misc.tabs(4)+f"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(sca*move-1)*1000);\n")
        #scale layers
        nlayers = [len(l) for l in self.cube.l_isolevels]
        numcubes = len(nlayers)
        for nc in range(numcubes):
            for nlays in range(nlayers[nc]):
                for sp in range(self.cube.iso_split[nc][nlays]):
                    self.file_html.write(misc.tabs(3)+"document.getElementById('cube__%slt%s_sp%s').setAttribute('scale', '1 1 '+sca);\n"%(nc,nlays,sp))
        #scale outline
        self.file_html.write(misc.tabs(3)+"document.getElementById('cube__ot').setAttribute('scale', '1 1 '+sca);\n")
        #move galaxies
        if self.cube.galaxies is not None:
            for (n,gal) in enumerate(self.cube.galaxies):
                v = self.cube.galaxies[gal]['coord'][2]
                a = self.cube.galaxies[gal]['coord'][0]
                b = self.cube.galaxies[gal]['coord'][1]
                # self.file_html.write(misc.tabs(3)+"document.getElementById('cube__%s_cross_tra').setAttribute('translation', '0 0 '+(sca-1)*%s);\n"%(gal,v))
                self.file_html.write(misc.tabs(3)+"document.getElementById('cube__%s_sphere_tra').setAttribute('translation', '%s %s '+sca*%s);\n"%(gal,a,b,v))
                self.file_html.write(misc.tabs(3)+"document.getElementById('cube__glt%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(n,a,b,v))
        #scale grids
        self.file_html.write("\t\t\t document.getElementById('cube__tlt').setAttribute('scale', '1 1 '+sca);\n")

        #scale markers
        # spheres
        self.file_html.write(misc.tabs(1)+"if (document.getElementById('new-sphere')[0].value != 'none') {\n")
        self.file_html.write(misc.tabs(2)+"for (s=0; s<document.getElementById('new-sphere').length; s++) {\n")
        self.file_html.write(misc.tabs(3)+"const sphInd = document.getElementById('new-sphere')[s].value.slice(3);\n")
        self.file_html.write(misc.tabs(3)+"document.getElementById('sphtra'+sphInd).setAttribute('translation', sph_coords[sphInd-1][0]+' '+sph_coords[sphInd-1][1]+' '+sca*sph_coords[sphInd-1][2]);\n")
        self.file_html.write(misc.tabs(3)+"document.getElementById('sphlabtra'+sphInd).setAttribute('translation', sph_coords[sphInd-1][0]+' '+sph_coords[sphInd-1][1]+' '+sca*sph_coords[sphInd-1][2]);\n")
        self.file_html.write(misc.tabs(2)+"}\n")
        self.file_html.write(misc.tabs(1)+"}\n")
        # boxes
        self.file_html.write(misc.tabs(1)+"if (document.getElementById('new-box')[0].value != 'none') {\n")
        self.file_html.write(misc.tabs(2)+"for (s=0; s<document.getElementById('new-box').length; s++) {\n")
        self.file_html.write(misc.tabs(3)+"const boxInd = document.getElementById('new-box')[s].value.slice(3);\n")
        self.file_html.write(misc.tabs(3)+"document.getElementById('boxtra'+boxInd).setAttribute('translation', box_coords[boxInd-1][0]+' '+box_coords[boxInd-1][1]+' '+sca*box_coords[boxInd-1][2]);\n")
        self.file_html.write(misc.tabs(3)+"document.getElementById('boxlabtra'+boxInd).setAttribute('translation', box_coords[boxInd-1][0]+' '+box_coords[boxInd-1][1]+' '+sca*box_coords[boxInd-1][2]);\n")
        self.file_html.write(misc.tabs(2)+"}\n")
        self.file_html.write(misc.tabs(1)+"}\n")
        # cones
        self.file_html.write(misc.tabs(1)+"if (document.getElementById('new-con')[0].value != 'none') {\n")
        self.file_html.write(misc.tabs(2)+"for (s=0; s<document.getElementById('new-con').length; s++) {\n")
        self.file_html.write(misc.tabs(3)+"const conInd = document.getElementById('new-con')[s].value.slice(3);\n")
        self.file_html.write(misc.tabs(3)+"document.getElementById('contra'+conInd).setAttribute('translation', con_coords[conInd-1][0]+' '+con_coords[conInd-1][1]+' '+sca*con_coords[conInd-1][2]);\n")
        self.file_html.write(misc.tabs(3)+"document.getElementById('conlabtra'+conInd).setAttribute('translation', con_coords[conInd-1][0]+' '+con_coords[conInd-1][1]+' '+sca*con_coords[conInd-1][2]);\n")
        self.file_html.write(misc.tabs(3)+"// missing orientation change (don't implement?)\n")
        self.file_html.write(misc.tabs(2)+"}\n")
        self.file_html.write(misc.tabs(1)+"}\n")
        # tubes
        self.file_html.write(misc.tabs(1)+"if (document.getElementById('new-tub')[0].value != 'none') {\n")
        self.file_html.write(misc.tabs(2)+"for (s=0; s<document.getElementById('new-tub').length; s++) {\n")
        self.file_html.write(misc.tabs(3)+"const tubInd = document.getElementById('new-tub')[s].value.slice(3);\n")
        self.file_html.write(misc.tabs(3)+"for (i=1; i<tubelen[tubInd-1]; i++) {\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('tubtra'+tubInd+'_'+i).setAttribute('translation', tub_coords[tubInd-1][i-1][0][0]+' '+tub_coords[tubInd-1][i-1][0][1]+' '+sca*tub_coords[tubInd-1][i-1][0][2]);\n")
        self.file_html.write(misc.tabs(4)+"const norm = Math.sqrt(tub_coords[tubInd-1][i-1][1][0]**2+tub_coords[tubInd-1][i-1][1][1]**2+(tub_coords[tubInd-1][i-1][1][2]*sca)**2)*1.015;\n")
        self.file_html.write(misc.tabs(4)+"const angle = Math.acos(tub_coords[tubInd-1][i-1][1][1]/norm);\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('tubtra'+tubInd+'_'+i).setAttribute('rotation', tub_coords[tubInd-1][i-1][1][2]*sca+' 0 '+(-tub_coords[tubInd-1][i-1][1][0])+' '+angle);\n")
        self.file_html.write(misc.tabs(4)+"document.getElementById('tub'+tubInd+'_'+i).setAttribute('height', norm);\n")
        self.file_html.write(misc.tabs(3)+"}\n")
        self.file_html.write(misc.tabs(3)+"document.getElementById('tublabtra'+tubInd).setAttribute('translation', tub_coords[tubInd-1][0][0][0]+' '+tub_coords[tubInd-1][0][0][1]+' '+sca*tub_coords[tubInd-1][0][0][2]);\n")
        self.file_html.write(misc.tabs(2)+"}\n")
        self.file_html.write(misc.tabs(1)+"}\n")

        # scale python markers
        delta = np.array(self.cube.delta)
        if sphere is not None:
            n = 0
            for s in range(len(sphere)):
                sphere[s] = (sphere[s] - np.mean(self.cube.coords, axis=1))/delta*trans
                self.file_html.write(misc.tabs(2)+f"document.getElementById('cube__pysph{n}tra').setAttribute('translation', '{sphere[s][0]} {sphere[s][1]} '+sca*{sphere[s][2]});\n")
                self.file_html.write(misc.tabs(2)+f"document.getElementById('cube__label_pysphtra{n}').setAttribute('translation', '{sphere[s][0]} {sphere[s][1]} '+sca*{sphere[s][2]});\n")
                n = n + 1
        if box is not None:
            n = 0
            for b in range(len(box)):
                box[b] = (box[b] - np.mean(self.cube.coords, axis=1))/delta*trans
                self.file_html.write(misc.tabs(2)+f"document.getElementById('cube__pybox{n}tra').setAttribute('translation', '{box[b][0]} {box[b][1]} '+sca*{box[b][2]});\n")
                self.file_html.write(misc.tabs(2)+f"document.getElementById('cube__label_pyboxtra{n}').setAttribute('translation', '{box[b][0]} {box[b][1]} '+sca*{box[b][2]});\n")
                n = n + 1
        if cone is not None:
            n = 0
            for c in range(len(cone)):
                cone[c] = (cone[c] - np.mean(self.cube.coords, axis=1))/delta*trans
                self.file_html.write(misc.tabs(2)+f"document.getElementById('cube__pycon{n}tra').setAttribute('translation', '{cone[c][0]} {cone[c][1]} '+sca*{cone[c][2]});\n")
                self.file_html.write(misc.tabs(2)+f"document.getElementById('cube__label_pycontra{n}').setAttribute('translation', '{cone[c][0]} {cone[c][1]} '+sca*{cone[c][2]});\n")
                n = n + 1
        # missing orientation change for cones
        if tube is not None:
            n = 0
            for t in tube:
                t = (t - np.mean(self.cube.coords, axis=1))/delta*trans
                pos = np.array([np.mean((t[i],t[i+1]), axis=0) for i in range(len(t)-1)])
                diff = np.diff(t, axis=0)

                for i in range(len(t)-1):
                    self.file_html.write(misc.tabs(2)+f"const diff{n}_{i} = [{diff[i][0]},{diff[i][1]},{diff[i][2]}];\n")
                    self.file_html.write(misc.tabs(2)+f"const height{n}_{i} = Math.sqrt(diff{n}_{i}[0]**2+diff{n}_{i}[1]**2+(sca*diff{n}_{i}[2])**2)*1.015;\n")
                    self.file_html.write(misc.tabs(2)+f"const angle{n}_{i} = Math.acos(diff{n}_{i}[1]/height{n}_{i});\n")
                    self.file_html.write(misc.tabs(2)+f"document.getElementById('cube__pytub{n}tra{i}').setAttribute('translation', '{pos[i][0]} {pos[i][1]} '+sca*{pos[i][2]});\n")
                    self.file_html.write(misc.tabs(2)+f"document.getElementById('cube__py{n}tub{i}').setAttribute('height', height{n}_{i}.toString());\n")
                    self.file_html.write(misc.tabs(2)+f"document.getElementById('cube__pytub{n}tra{i}').setAttribute('rotation', sca*diff{n}_{i}[2]+' 0 '+(-diff{n}_{i}[0])+' '+angle{n}_{i});\n")
                self.file_html.write(misc.tabs(2)+f"document.getElementById('cube__label_pytubtra{n}').setAttribute('translation', '{pos[0][0]} {pos[0][1]} '+sca*{pos[0][2]});\n")
                n = n +1

        #scale axes
        ax, axtick = misc.labpos
        
        for i in range(12):
            if i < 6:
                self.file_html.write(misc.tabs(3)+"document.getElementById('cube__alt_diff%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, ax[i][0], ax[i][1], ax[i][2])) #str(ax[i])[1:-1]
                if isinstance(self.cube.lines, dict) == False:
                    self.file_html.write(misc.tabs(3)+"document.getElementById('cube__alt_real%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, ax[i][0], ax[i][1], ax[i][2]))
            self.file_html.write(misc.tabs(3)+"document.getElementById('cube__att_diff%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, axtick[i][0], axtick[i][1], axtick[i][2]))
            if isinstance(self.cube.lines, dict) == False:
                self.file_html.write(misc.tabs(3)+"document.getElementById('cube__att_real%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, axtick[i][0], axtick[i][1], axtick[i][2]))
        
        self.file_html.write(misc.tabs(2)+"}\n")
        self.file_html.write(misc.tabs(2)+"</script>\n")
    
    def func_image2d(self):
        """
        Make JS function to show/hide the 2D image.
        The X3D file must have a 2D image for this to work.
        Must be after buttons().
        """
        # PICKING
        # self.file_html.write(roundTo) #premade string with function to round to two decimals

        self.file_html.write(misc.tabs(2)+"<script>\n")
        self.file_html.write(misc.tabs(3)+"var showalertimage2d = true;\n")
        self.file_html.write(misc.tabs(3)+"function setimage2d()\n\t\t{\n")
        self.file_html.write(misc.tabs(4)+"if (showalertimage2d) {\n")
        self.file_html.write(misc.tabs(5)+'alert("The opacity feature does not work alongside the 2D image. Hide the 2D image to change the opacity.")\n')
        self.file_html.write(misc.tabs(5)+"showalertimage2d = false;\n")
        self.file_html.write(misc.tabs(4)+"}\n")
        self.file_html.write(misc.tabs(4)+"if(document.getElementById('cube__image2d').getAttribute('scale') != '1 1 1') {\n")
        self.file_html.write(misc.tabs(5)+"document.getElementById('cube__image2d').setAttribute('scale', '1 1 1');\n")
        self.file_html.write(misc.tabs(4)+"} else {\n")
        self.file_html.write(misc.tabs(5)+"document.getElementById('cube__image2d').setAttribute('scale', '0 0 0');\n")
        self.file_html.write(misc.tabs(4)+"}\n")
        self.file_html.write(misc.tabs(3)+"}\n\t\t</script>\n")

            
    def close_html(self):
        """
        Must be used to finish and close the HTML file. Not using this function results
        in an error.
        """
        if self.cube.interface != 'minimal':
            self.file_html.write(misc.tablehtml)
        self.file_html.write('\n\t</body>\n</html>')
        self.file_html.close()

class WriteVis:
    """
    Create an HTML file to visualise a 3D model of spectral line data (without an X3D file).

    Parameters
    ----------
    filename : str
        Name of the HTML file to be created (without extension).
    cube : Cube
        Cube object with the data to be visualised.
    pagetitle : str, optional
        The title of the web page.
    description : str
        A description for the web page.
    """
    def __init__(self, filename, cube, description=None, pagetitle=None):
        #some attributes to use later
        self.cube = cube
        if pagetitle is None:
            pagetitle = self.cube.name
        self.visfile = open(filename+'.html', 'w', encoding="utf-8")
        self.visfile.write('<!DOCTYPE html>\n\t <head>\n')
        # pa = misc.__file__[:-7]+'x3dom'
        # if '\\' in pa:
        #     pa = pa + '\\'
        # else:
        #     pa = pa + '/'
        # self.visfile.write(misc.tabs(2)+f"<script type='text/javascript' src='{pa}x3dom.js'></script>\n")
        self.visfile.write(misc.tabs(2)+f"<script type='text/javascript' src='https://ixakalabadie.github.io/x3dom/x3dom.js'></script>\n")
        self.visfile.write(misc.tabs(2)+"<script type='text/javascript'  src='https://www.maths.nottingham.ac.uk/plp/pmadw/LaTeXMathML.js'></script>\n")
        self.visfile.write(misc.tabs(2)+f"<link rel='stylesheet' type='text/css' href='https://ixakalabadie.github.io/x3dom/x3dom.css'></link>\n")
        self.visfile.write(misc.tabs(2)+"<script type='text/javascript' src='https://code.jquery.com/jquery-3.6.3.min.js'></script>\n")
        self.visfile.write(misc.tabs(2)+f'<script src="https://ixakalabadie.github.io/x3dom/js-colormaps.js"></script> <!-- FOR COLORMAPS IN JS-->\n')
        self.visfile.write(misc.tabs(2)+f'<script type="text/javascript" src="https://ixakalabadie.github.io/x3dom/markers.js"></script> <!-- FOR MARKERS IN JS-->\n')
        self.visfile.write(misc.tabs(2)+f'<title> ViSL3D: {self.cube.name}</title>\n')
        if self.cube.interface == 'minimal':
            self.visfile.write("\n\t\t<style>\n"+misc.tabs(3)+"x3d\n"+misc.tabs(4)+"{\n"+misc.tabs(5)+"border:2px solid darkorange;\n"+misc.tabs(5)+"width:100vw;\n"+misc.tabs(5)+"height:100vh;\n"+misc.tabs(3)+"}\n"+misc.tabs(3)+"</style>\n\t</head>\n\t<body>\n")
        else:
            self.visfile.write("\n\t\t<style>\n"+misc.tabs(3)+"x3d\n"+misc.tabs(4)+"{\n"+misc.tabs(5)+"border:2px solid darkorange;\n"+misc.tabs(5)+"width:95vw;\n"+misc.tabs(5)+"height: 75vh;\n"+misc.tabs(3)+"}\n"+misc.tabs(3)+"</style>\n\t</head>\n\t<body>\n")
        self.visfile.write(f'\t<h1 align="middle"> ViSL3D: 3D model of {pagetitle} </h1>\n')
        self.visfile.write('\t<hr/>\n')
        if description is not None:
            self.visfile.write(f"\t<p>\n\t {description}</p> \n")
        self.visfile.write(misc.roundto)

    def func_layers(self):
        """
        Make JS funcion to hide/show layers.
        """
        numcubes = len(self.cube.l_isolevels)
        nlayers = [len(l) for l in self.cube.l_isolevels]
        self.visfile.write(misc.tabs(2)+"<script>\n")
        for nc in range(numcubes):
            self.visfile.write(misc.tabs(3)+"function hideall%s() {\n"%nc)
            for i in range(nlayers[nc]):
                self.visfile.write(f'{misc.tabs(4)}setHI{nc}layer{i}();\n')
            self.visfile.write(misc.tabs(3)+"}\n")

        self.visfile.write(misc.tabs(3)+"var showalertopa = true;\n")
        self.visfile.write(misc.tabs(3)+"function changeopa() {\n")
        self.visfile.write(misc.tabs(4)+"if (showalertopa) {\n")
        self.visfile.write(misc.tabs(5)+'alert("The opacity feature does not work alongside the 2D image. Hide the 2D image to change the opacity.")\n')
        self.visfile.write(misc.tabs(5)+"showalertopa = false;\n")
        self.visfile.write(misc.tabs(4)+"}\n")
        self.visfile.write(misc.tabs(4)+f'const nlayers = {nlayers};\n')
        self.visfile.write(misc.tabs(4)+'if (document.getElementById("0layer0_sp0_appe").getAttribute("sortType") == "opaque") {\n')
        self.visfile.write(misc.tabs(5)+'for (let nc = 0; nc < %s; nc++) {\n'%numcubes)
        self.visfile.write(misc.tabs(6)+'for (let i = 0; i < nlayers[nc]; i++) {\n')
        self.visfile.write(misc.tabs(7)+f'document.getElementById(nc+"layer"+i+"_sp0_appe").setAttribute("sortType", "transparent");\n')
        self.visfile.write(misc.tabs(6)+"}\n")
        self.visfile.write(misc.tabs(5)+"}\n")
        self.visfile.write(misc.tabs(4)+'} else {\n')
        self.visfile.write(misc.tabs(5)+'for (let nc = 0; nc < %s; nc++) {\n'%numcubes)
        self.visfile.write(misc.tabs(6)+'for (let i = 0; i < nlayers[nc]; i++) {\n')
        self.visfile.write(misc.tabs(7)+f'document.getElementById(nc+"layer"+i+"_sp0_appe").setAttribute("sortType", "opaque");\n')
        self.visfile.write(misc.tabs(6)+"}\n")
        self.visfile.write(misc.tabs(5)+"}\n")
        self.visfile.write(misc.tabs(4)+"}\n")
        self.visfile.write(misc.tabs(3)+"}\n")

        self.visfile.write(misc.tabs(2)+"</script>\n")

        for nc in range(numcubes):
            for i in range(nlayers[nc]):
                self.visfile.write("\t <script>\n\t \t function setHI%slayer%s()\n\t \t {\n\t \t if(document.getElementById('%slayer%s_sp0').getAttribute('transparency') != '0.8') {\n"%(nc,i,nc,i))
                self.visfile.write(f"\t\t document.getElementById('{nc}but{i}').style.border = '5px dashed black';\n")
                for sp in range(self.cube.iso_split[nc][i]):
                    self.visfile.write(f"\t\t document.getElementById('{nc}layer{i}_sp{sp}').setAttribute('transparency', '0.8');\n")
                    self.visfile.write(f"\t\t document.getElementById('{nc}layer{i}_sp{sp}_shape').setAttribute('ispickable', 'true');\n")
                self.visfile.write("\t\t } else { \n")
                self.visfile.write(f"\t\t document.getElementById('{nc}but{i}').style.border = 'inset black';\n")
                for sp in range(self.cube.iso_split[nc][i]):
                    self.visfile.write(f"\t\t document.getElementById('{nc}layer{i}_sp{sp}').setAttribute('transparency', '1');\n")
                    self.visfile.write(f"\t\t document.getElementById('{nc}layer{i}_sp{sp}_shape').setAttribute('ispickable', 'false');\n")
                self.visfile.write("\t\t } \n\t\t }\n\t </script>\n")

    def func_galaxies(self):
        """
        Make JS function to hide/show galaxies and their labels.
        The X3D file must have galaxies for this to work.
        Must be after buttons()
        """
        for i,gal in enumerate(self.cube.galaxies):
            if i == 0:
                self.visfile.write("\t \t <script>\n\t \t function setgals()\n\t \t {\n\t \t if(document.getElementById('%s').getAttribute('transparency')!= '0'){\n"%gal)
            # self.visfile.write("\t \t document.getElementById('%s_cross').setAttribute('transparency', '0');\n"%gal)
            self.visfile.write("\t \t document.getElementById('%s').setAttribute('transparency', '0');\n"%gal)
        self.visfile.write("\t \t }\n\t \t else {\n")
        for i,gal in enumerate(self.cube.galaxies):
            # self.visfile.write("\t \t document.getElementById('%s_cross').setAttribute('transparency', '1');\n"%gal)
            self.visfile.write("\t \t document.getElementById('%s').setAttribute('transparency', '1');\n"%gal)
        self.visfile.write("\t\t }\n\t\t }\n\t\t </script>\n")

        for i,gal in enumerate(self.cube.galaxies):
            if i == 0:
                self.visfile.write("\t\t <script>\n\t \t function setgallabels()\n\t \t {\n\t \t if(document.getElementById('label_%s').getAttribute('transparency')!= '0'){\n"%gal)
            self.visfile.write("\t \t document.getElementById('label_%s').setAttribute('transparency', '0');\n"%gal)
        self.visfile.write("\t \t }\n\t \t else {\n")
        for i,gal in enumerate(self.cube.galaxies):
            self.visfile.write("\t \t document.getElementById('label_%s').setAttribute('transparency', '1');\n"%gal)
        self.visfile.write("\t\t }\n\t\t }\n\t\t </script>\n")
            

    def func_grids(self):
        """
        Make JS function to hide/show grids.
        """
        self.visfile.write(misc.tabs(2)+"<script>\n\t\tfunction setgrids()\n\t\t{\n")
        self.visfile.write(misc.tabs(3)+"if(document.getElementById('ticklines').getAttribute('transparency') == '0') {\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('ticklines').setAttribute('transparency', '1');\n")
        self.visfile.write(misc.tabs(3)+"} else if (document.getElementById('outline').getAttribute('transparency') == '0') {\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('outline').setAttribute('transparency', '1');\n")
        self.visfile.write(misc.tabs(3)+"} else {\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('ticklines').setAttribute('transparency', '0');\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('outline').setAttribute('transparency', '0');\n")
        self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(2)+"}\n\t\t </script>\n")

    def func_axes(self):
        """
        Make JS function to hide/show axes labels.
        """
        self.visfile.write(misc.tabs(2)+"<script>\n")
        self.visfile.write(misc.tabs(2)+"function setaxes()\n")
        self.visfile.write(misc.tabs(2)+"{\n")
        self.visfile.write(misc.tabs(2)+"if(document.getElementById('axlab_diff1').getAttribute('transparency') == '0') {\n")
        self.visfile.write(misc.tabs(3)+"for (i=0; i<12; i++) {\n")
        self.visfile.write(misc.tabs(3)+"if (i<6) {\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('axlab_diff'+i).setAttribute('transparency', '1');\n")
        if isinstance(self.cube.lines, dict) == False:
            self.visfile.write(misc.tabs(4)+"document.getElementById('axlab_real'+i).setAttribute('transparency', '0');\n")
        self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('axtick_diff'+i).setAttribute('transparency', '1');\n")
        if isinstance(self.cube.lines, dict) == False:
            self.visfile.write(misc.tabs(4)+"document.getElementById('axtick_real'+i).setAttribute('transparency', '0');\n")
        self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(2)+"}\n")
        if isinstance(self.cube.lines, dict) == False:
            self.visfile.write(misc.tabs(2)+"else if (document.getElementById('axlab_real1').getAttribute('transparency') == '0') {\n")
            self.visfile.write(misc.tabs(3)+"for (i=0; i<12; i++) {\n")
            self.visfile.write(misc.tabs(3)+"if (i<6) {\n")
            self.visfile.write(misc.tabs(4)+"document.getElementById('axlab_real'+i).setAttribute('transparency', '1');\n")
            self.visfile.write(misc.tabs(3)+"}\n")
            self.visfile.write(misc.tabs(4)+"document.getElementById('axtick_real'+i).setAttribute('transparency', '1');\n")
            self.visfile.write(misc.tabs(3)+"}\n")
            self.visfile.write(misc.tabs(2)+"}\n")
        self.visfile.write(misc.tabs(2)+"else {\n")
        self.visfile.write(misc.tabs(3)+"for (i=0; i<12; i++) {\n")
        self.visfile.write(misc.tabs(3)+"if (i<6) {\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('axlab_diff'+i).setAttribute('transparency', '0');\n")
        self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('axtick_diff'+i).setAttribute('transparency', '0');\n")
        self.visfile.write(misc.tabs(3)+"}\n")                
        self.visfile.write("\t\t }\n\t\t }\n\t\t </script>\n")

    def func_pick(self):
        """
        Allows picking the coordinates by clicking in the figure.
        WORKS WITH VIEWPOINT NOT WITH ORTHOVIEWPOINT.
        NOT FINISHED, DON'T USE.

        """
        # self.visfile.write(misc.roundto) #premade string with function to round to two decimals
        # self.visfile.write(misc.tabs(1)+"<script>\n")
        # self.visfile.write(misc.tabs(3)+"const picksca = document.querySelector('#scalev');\n")
        # self.visfile.write(misc.tabs(2)+"function handleClick(event) {\n")
        # self.visfile.write(misc.tabs(3)+"const sca = picksca.value;\n")
        # self.visfile.write(misc.tabs(3)+"var coordinates = event.hitPnt;\n")
        # self.visfile.write(misc.tabs(3)+"$('#coordX').html(roundTo(coordinates[0], 2)+' %s');\n"%self.cube.units[1])
        # self.visfile.write(misc.tabs(3)+"$('#coordY').html(roundTo(coordinates[1], 2)+' %s');\n"%self.cube.units[2])
        # self.visfile.write(misc.tabs(3)+"$('#coordZ').html(roundTo(coordinates[2], 2)/sca+' %s');\n"%self.cube.units[3])
        # self.visfile.write(misc.tabs(2)+"}\n\t </script>\n")
        pass

    def func_animation(self):
        """
        Make JS function to start/stop the animation of the X3D models.
        """
        self.visfile.write('\n'+misc.tabs(1)+"<script>")
        self.visfile.write('\n'+misc.tabs(2)+"var active = false;")
        self.visfile.write('\n'+misc.tabs(2)+"function animation() {")
        self.visfile.write('\n'+misc.tabs(3)+"if (active == false) {")
        self.visfile.write('\n'+misc.tabs(4)+"document.getElementById('time').setAttribute('startTime', document.getElementById('time').getAttribute('time'));")
        self.visfile.write('\n'+misc.tabs(4)+"document.getElementById('time').setAttribute('isPaused', 'false');")
        self.visfile.write('\n'+misc.tabs(4)+"active = true;")
        self.visfile.write('\n'+misc.tabs(3)+"} else if (document.getElementById('time').getAttribute('isPaused') == 'false') {")
        self.visfile.write('\n'+misc.tabs(4)+"document.getElementById('time').setAttribute('loop', 'false');")
        self.visfile.write('\n'+misc.tabs(4)+"document.getElementById('time').setAttribute('isPaused', 'true')")
        self.visfile.write('\n'+misc.tabs(3)+"} else {")
        self.visfile.write('\n'+misc.tabs(4)+"document.getElementById('time').setAttribute('loop', 'true');")
        self.visfile.write('\n'+misc.tabs(4)+"document.getElementById('time').setAttribute('isPaused', 'false');")
        self.visfile.write('\n'+misc.tabs(3)+"}\n"+misc.tabs(2)+'}\n'+misc.tabs(1)+'</script>\n')

    def start_x3d(self, point=None):
        """
        Start the X3D section of the HTML.

        Parameters
        ----------
        point : list, optional
            List of points to define viewpoints. The point will be used to create a vector from the origin.
            For example, [1,0,0] will create a viewpoint looking at the positive x-axis. The y axis is inverted.
        """
        self.visfile.write(misc.tabs(1)+"<center><x3d id='cubeFixed'>\n")
        self.visfile.write('\t<Scene doPickPass="False">\n')
        self.visfile.write('\t\t<Background id="back" skyColor="0.6 0.6 0.6"></Background>\n')
        self.visfile.write(misc.tabs(2)+
            '<NavigationInfo type=\'"EXAMINE" "ANY"\' speed="4" headlight="true"></NavigationInfo>\n')
        self.visfile.write(
            misc.tabs(2)+'<DirectionalLight ambientIntensity="1" intensity="0" color="1 1 1"></DirectionalLight>\n')

        self.visfile.write(misc.tabs(3)+"<OrthoViewpoint id=\"front\" bind='false' centerOfRotation='0,0,0' description='RA-Dec view' fieldOfView='[-1400.0,-1400.0,1400.0,1400.0]' isActive='false' metadata='X3DMetadataObject' orientation='0,1,0,3.141593' position='0,0,-5500' zFar='11000' zNear='0.0001' ></OrthoViewpoint>\n")
        self.visfile.write(misc.tabs(3)+"<OrthoViewpoint id=\"side\" bind='false' centerOfRotation='0,0,0' description='Z - Dec view' fieldOfView='[-1400.0,-1400.0,1400.0,1400.0]' isActive='false' metadata='X3DMetadataObject' orientation='0,-1,0,1.570796' position='-5500,0,0' zFar='11000' zNear='0.0001' ></OrthoViewpoint>\n")
        self.visfile.write(misc.tabs(3)+"<OrthoViewpoint id=\"side2\" bind='false' centerOfRotation='0,0,0' description='Z - RA view' fieldOfView='[-1400.0,-1400.0,1400.0,1400.0]' isActive='false' metadata='X3DMetadataObject' orientation='1,1,1,4.1888' position='0,5500,0' zFar='11000' zNear='0.0001' ></OrthoViewpoint>\n")
        if point is not None:
            for i,po in enumerate(point):
                ori = misc.calc_axis_angle(po)
                pos = misc.calc_camera_position(ori)*5500
                self.visfile.write(misc.tabs(3)+f"<OrthoViewpoint id='vp{i}' bind='false' centerOfRotation='0,0,0' " \
                    +f"description='vp{i}' fieldOfView='[-1400.0,-1400.0,1400.0,1400.0]' isActive='false' " \
                    +f"orientation='{str(ori)[1:-1]}' position='{str(pos)[1:-1]}' zFar='11000' zNear='0.0001' ></OrthoViewpoint>\n")

        self.visfile.write(misc.tabs(2)+'<Transform id="ROOT" translation="0 0 0">\n')

    def close_x3d(self):
        """
        Close the X3D section of the HTML.
        Must go after start_x3d().
        """
        self.visfile.write('\n')
        self.visfile.write(misc.tabs(2)+'</Transform>')
        self.visfile.write(misc.tabs(2)+"</scene>\n")
        self.visfile.write("\t</x3d></center>\n")

    def func_background(self):
        """
        Make JS function to change the background color of the X3D figure.
        Must be after buttons()
        """
        self.visfile.write(misc.tabs(3)+"<script>\n")
        self.visfile.write(misc.tabs(4)+"function hex2Rgb(hex) {\n")
        self.visfile.write(misc.tabs(5)+r"var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);")
        self.visfile.write(misc.tabs(5)+"\n")
        self.visfile.write(misc.tabs(5)+"var r = parseInt(result[1], 16)/255.;\n")
        self.visfile.write(misc.tabs(5)+"var g = parseInt(result[2], 16)/255.;\n")
        self.visfile.write(misc.tabs(5)+"var b = parseInt(result[3], 16)/255.;\n")
        self.visfile.write(misc.tabs(5)+"return r.toString()+' '+g.toString()+' '+b.toString()\n")
        self.visfile.write(misc.tabs(4)+"}\n")
        self.visfile.write(misc.tabs(4)+"const background = document.querySelector('#back-choice');\n")
        self.visfile.write(misc.tabs(4)+"function change_background() {\n")
        self.visfile.write(misc.tabs(5)+"const backCol = background.value; \n")
        self.visfile.write(misc.tabs(5)+"document.getElementById('back').setAttribute('skyColor', hex2Rgb(backCol));\n")
        self.visfile.write(misc.tabs(4)+"}\n")
        self.visfile.write(misc.tabs(3)+"</script>\n")

    def func_markers(self):
        """
        Create buttons to create markers for the X3D model interactively in the web page.
        Must be after buttons(). It is assumed that scalev is used. It will fail otherwise. 
        The JS functions are included by default in 'markers.js'.
        """
        self.visfile.write(misc.tabs(2) + '\n')
        self.visfile.write(misc.tabs(2)+"<div id=\"divmaster\" style=\"margin-left: 2%\">\n")
        self.visfile.write(misc.tabs(3)+"<br>\n")
        self.visfile.write(misc.tabs(3)+"<!-- BUTTON TO CHANGE LAYOUT -->\n")
        self.visfile.write(misc.tabs(3)+"<label for=\"markers-choice\"><b>Markers:</b> </label>\n")
        self.visfile.write(misc.tabs(3)+"<select id=\"markers-choice\">\n")
        self.visfile.write(misc.tabs(4)+"<option value=\"none\">None</option>\n")
        self.visfile.write(misc.tabs(4)+"<option value=\"sphere\">Sphere</option>\n")
        self.visfile.write(misc.tabs(4)+"<option value=\"box\">Box</option>\n")
        self.visfile.write(misc.tabs(4)+"<option value=\"tub\">Tube</option>\n")
        self.visfile.write(misc.tabs(4)+"<option value=\"con\">Cone</option>\n")
        self.visfile.write(misc.tabs(3)+"</select>\n")
        self.visfile.write(misc.tabs(3)+"<input type=\"color\" id=\"butcol\" value=\"#ff0000\">\n")
        self.visfile.write(misc.tabs(3)+"<button id=\"butnew\" onclick=\"newmarker()\">New</button>\n")
        self.visfile.write(misc.tabs(3)+"<button id=\"butcreate\" onclick=\"createmarker()\">Create</button>\n")
        self.visfile.write(misc.tabs(3)+"<button id=\"butremove\" onclick=\"removemarker()\">Remove</button>\n")
        self.visfile.write(misc.tabs(3)+"<button id=\"buthide\" onclick=\"hidemarker()\">Hide/Show</button> <br><br>\n")
        self.visfile.write(misc.tabs(2)+"</div>\n")
        self.visfile.write(misc.tabs(2)+"<!-- create various layouts for different objects -->\n")
        self.visfile.write(misc.tabs(2)+"<div id=\"spherediv\" style=\"display:none ; margin-left: 2%\">\n")
        self.visfile.write(misc.tabs(3)+"<select id=\"new-sphere\">\n")
        self.visfile.write(misc.tabs(4)+"<option value=\"none\">None</option>\n")
        self.visfile.write(misc.tabs(3)+"</select>\n")
        self.visfile.write(misc.tabs(3)+"<br>\n")
        self.visfile.write(misc.tabs(2)+"</div>\n")
        self.visfile.write(misc.tabs(2)+"<div id=\"boxdiv\" style=\"display:none ; margin-left: 2%\">\n")
        self.visfile.write(misc.tabs(3)+"<select id=\"new-box\">\n")
        self.visfile.write(misc.tabs(4)+"<option value=\"none\">None</option>\n")
        self.visfile.write(misc.tabs(3)+"</select>\n")
        self.visfile.write(misc.tabs(3)+"<br>\n")
        self.visfile.write(misc.tabs(2)+"</div>\n")
        self.visfile.write(misc.tabs(2)+"<div id=\"condiv\" style=\"display:none ; margin-left: 2%\">\n")
        self.visfile.write(misc.tabs(3)+"<select id=\"new-con\">\n")
        self.visfile.write(misc.tabs(4)+"<option value=\"none\">None</option>\n")
        self.visfile.write(misc.tabs(3)+"</select>\n")
        self.visfile.write(misc.tabs(3)+"<br>\n")
        self.visfile.write(misc.tabs(2)+"</div>\n")
        self.visfile.write(misc.tabs(2)+"<div id=\"tubdiv\" style=\"display:none ; margin-left: 2%\">\n")
        self.visfile.write(misc.tabs(3)+"<select id=\"new-tub\">\n")
        self.visfile.write(misc.tabs(4)+"<option value=\"none\">None</option>\n")
        self.visfile.write(misc.tabs(3)+"</select>\n")
        self.visfile.write(misc.tabs(3)+"<button id=\"addpoint\" onclick=\"addpoint(seltub, tubelen)\">Add point</button>\n")
        self.visfile.write(misc.tabs(3)+"<br>\n")
        self.visfile.write(misc.tabs(2)+"</div>\n")

        self.visfile.write(misc.tabs(2)+"<script type=\"text/javascript\">\n")
        self.visfile.write(misc.tabs(3)+"// General parameters\n")
        self.visfile.write(misc.tabs(3)+"const marktype = document.querySelector('#markers-choice');\n")
        self.visfile.write(misc.tabs(3)+"marktype.addEventListener('change', newlayout);\n")
        self.visfile.write(misc.tabs(3)+"const sscasv = document.querySelector('#scalev');\n")
        self.visfile.write(misc.tabs(3)+"const col = document.querySelector('#butcol');\n\n")
        self.visfile.write(misc.tabs(3)+"// Spheres\n")
        self.visfile.write(misc.tabs(3)+"var nspheres = 0; //number of spheres\n")
        self.visfile.write(misc.tabs(3)+"var sph_coords = []; //coordinates of sphere\n\n")
        self.visfile.write(misc.tabs(3)+"const selsph = document.querySelector('#new-sphere');\n")
        self.visfile.write(misc.tabs(3)+"selsph.addEventListener('change', changeSphere);\n\n")
        self.visfile.write(misc.tabs(3)+"// Boxes\n")
        self.visfile.write(misc.tabs(3)+"var nboxes = 0; //number of boxes\n")
        self.visfile.write(misc.tabs(3)+"var box_coords = []; //coordinates of boxes\n\n")
        self.visfile.write(misc.tabs(3)+"const selbox = document.querySelector('#new-box');\n")
        self.visfile.write(misc.tabs(3)+"selbox.addEventListener('change', changeBox);\n\n")
        self.visfile.write(misc.tabs(3)+"// Cones\n")
        self.visfile.write(misc.tabs(3)+"var ncones = 0; //number of cones\n")
        self.visfile.write(misc.tabs(3)+"var con_coords = []; //coordinates of cones\n\n")
        self.visfile.write(misc.tabs(3)+"const selcon = document.querySelector('#new-con');\n")
        self.visfile.write(misc.tabs(3)+"selcon.addEventListener('change', changeCon);\n\n")
        self.visfile.write(misc.tabs(3)+"// Tubes\n")
        self.visfile.write(misc.tabs(3)+"var ntubes = 0; //number of tubes\n")
        self.visfile.write(misc.tabs(3)+"var tub_coords = []; //coordinates of tubes\n")
        self.visfile.write(misc.tabs(3)+"var tubelen = []; //lengths of tubes in number of cylinders\n\n")
        self.visfile.write(misc.tabs(3)+"const seltub = document.querySelector('#new-tub');\n")
        self.visfile.write(misc.tabs(3)+"seltub.addEventListener('change', changeTub);\n\n")
        self.visfile.write(misc.tabs(3)+"// General\n")
        self.visfile.write(misc.tabs(3)+"function newmarker() {\n")
        self.visfile.write(misc.tabs(4)+"if (marktype.value == 'sphere') {\n")
        self.visfile.write(misc.tabs(5)+"nspheres = nspheres += 1;\n")
        self.visfile.write(misc.tabs(5)+"nspheres = newSphere(nspheres, selsph);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'box') {\n")
        self.visfile.write(misc.tabs(5)+"nboxes = nboxes += 1;\n")
        self.visfile.write(misc.tabs(5)+"nboxes = newBox(nboxes, selbox);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'tub') {\n")
        self.visfile.write(misc.tabs(5)+"ntubes = ntubes += 1;\n")
        self.visfile.write(misc.tabs(5)+"ntubes, tubelen = newTub(ntubes, seltub, tubelen);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'con') {\n")
        self.visfile.write(misc.tabs(5)+"ncones = ncones += 1;\n")
        self.visfile.write(misc.tabs(5)+"ncones = newCon(ncones, selcon);\n")
        self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(2)+"}\n\n")
        self.visfile.write(misc.tabs(3)+"function createmarker() {\n")
        self.visfile.write(misc.tabs(4)+"const sca = inpscasv.value;\n\n")
        self.visfile.write(misc.tabs(4)+f"const means = {list(np.mean(self.cube.coords, axis=1))};\n")
        delt = [np.abs(np.diff(self.cube.coords[0])/self.cube.l_cubes[0].shape[0]),
                 np.abs(np.diff(self.cube.coords[1])/self.cube.l_cubes[0].shape[1]),
                 np.abs(np.diff(self.cube.coords[2])/self.cube.l_cubes[0].shape[2])]
        delt = [factor[0] for factor in delt]
        self.visfile.write(misc.tabs(4)+f"const delt = {list(delt)};\n")
        trans = [2000/self.cube.l_cubes[0].shape[0], 2000/self.cube.l_cubes[0].shape[1], 2000/self.cube.l_cubes[0].shape[2]]
        self.visfile.write(misc.tabs(4)+f"const trans = {trans};\n")
        self.visfile.write(misc.tabs(4)+"if (marktype.value == 'sphere') {\n")
        self.visfile.write(misc.tabs(5)+f"sph_coords = createSphere(sca, selsph, col, sph_coords, means, delt, trans, html=true);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'box') {\n")
        self.visfile.write(misc.tabs(5)+"box_coords = createBox(sca, selbox, col, box_coords, means, delt, trans, html=true);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'tub') {\n")
        self.visfile.write(misc.tabs(5)+"tub_coords = createTub(sca, seltub, col, tub_coords, tubelen, means, delt, trans, html=true);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'con') {\n")
        self.visfile.write(misc.tabs(5)+"con_coords = createCon(sca, selcon, col, con_coords, means, delt, trans, html=true);\n")
        self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(2)+"}\n\n")
        self.visfile.write(misc.tabs(3)+"function removemarker() {\n")
        self.visfile.write(misc.tabs(4)+"if (marktype.value == 'sphere') {\n")
        self.visfile.write(misc.tabs(5)+"removeSphere(selsph);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'box') {\n")
        self.visfile.write(misc.tabs(5)+"removeBox(selbox);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'tub') {\n")
        self.visfile.write(misc.tabs(5)+"removeTub(seltub, tubelen);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'con') {\n")
        self.visfile.write(misc.tabs(5)+"removeCon(selcon);\n")
        self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(2)+"}\n")
        # hidemarker
        self.visfile.write(misc.tabs(3)+"function hidemarker() {\n")
        self.visfile.write(misc.tabs(4)+"if (marktype.value == 'sphere') {\n")
        self.visfile.write(misc.tabs(5)+"hideSphere(selsph);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'box') {\n")
        self.visfile.write(misc.tabs(5)+"hideBox(selbox);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'tub') {\n")
        self.visfile.write(misc.tabs(5)+"hideTub(seltub, tubelen);\n")
        self.visfile.write(misc.tabs(4)+"} else if (marktype.value == 'con') {\n")
        self.visfile.write(misc.tabs(5)+"hideCon(selcon);\n")
        self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(2)+"}\n")
        self.visfile.write(misc.tabs(2)+"</script>\n")

    def func_pymarkers(self, tube=None, sphere=None, box=None, cone=None):
        """
        JS function to hide/show markers created with make_markers() directly to the X3D model.

        Parameters
        ----------
        tube : list
            List with the points parameter included in make_markers() for tubes.
        sphere : list
            List with the points parameter included in make_markers() for spheres.
        box : list
            List with the points parameter included in make_markers() for boxes.
        cone : list
            List with the points parameter included in make_markers() for cones.
        """
        self.visfile.write(misc.tabs(2)+"<script>\n")
        if tube is not None:
            self.visfile.write(misc.tabs(3)+"function settube() {\n")
            self.visfile.write(misc.tabs(4)+"if(document.getElementById('py0tubmat0').getAttribute('transparency') != '0') {\n")
            for t in range(len(tube)):
                for i in range(len(tube[t])-1):
                    self.visfile.write(misc.tabs(5)+"document.getElementById('py%stubmat%s').setAttribute('transparency', '0');\n"%(t,i))
                self.visfile.write(misc.tabs(5)+"document.getElementById('label_pytub%s').setAttribute('transparency', '0');\n"%t)
            self.visfile.write(misc.tabs(4)+"} else {\n")
            for t in range(len(tube)):
                for i in range(len(tube[t])-1):
                    self.visfile.write(misc.tabs(5)+"document.getElementById('py%stubmat%s').setAttribute('transparency', '1');\n"%(t,i))
                self.visfile.write(misc.tabs(5)+"document.getElementById('label_pytub%s').setAttribute('transparency', '1');\n"%t)
            self.visfile.write(misc.tabs(4)+"}\n")
            self.visfile.write(misc.tabs(3)+"}\n")
        if sphere is not None:
            self.visfile.write(misc.tabs(3)+"function setsphere() {\n")
            self.visfile.write(misc.tabs(4)+"if(document.getElementById('py0sphmat').getAttribute('transparency') != '0') {\n")
            for i in range(len(sphere)):
                self.visfile.write(misc.tabs(5)+"document.getElementById('py%ssphmat').setAttribute('transparency', '0');\n"%i)
                self.visfile.write(misc.tabs(5)+"document.getElementById('label_pysph%s').setAttribute('transparency', '0');\n"%i)
            self.visfile.write(misc.tabs(4)+"} else {\n")
            for i in range(len(sphere)):
                self.visfile.write(misc.tabs(5)+"document.getElementById('py%ssphmat').setAttribute('transparency', '1');\n"%i)
                self.visfile.write(misc.tabs(5)+"document.getElementById('label_pysph%s').setAttribute('transparency', '1');\n"%i)
            self.visfile.write(misc.tabs(4)+"}\n")
            self.visfile.write(misc.tabs(3)+"}\n")
        if box is not None:
            self.visfile.write(misc.tabs(3)+"function setbox() {\n")
            self.visfile.write(misc.tabs(4)+"if(document.getElementById('py0boxmat').getAttribute('transparency') != '0') {\n")
            for i in range(len(box)):
                self.visfile.write(misc.tabs(5)+"document.getElementById('py%sboxmat').setAttribute('transparency', '0');\n"%i)
                self.visfile.write(misc.tabs(5)+"document.getElementById('label_pybox%s').setAttribute('transparency', '0');\n"%i)
            self.visfile.write(misc.tabs(4)+"} else {\n")
            for i in range(len(box)):
                self.visfile.write(misc.tabs(5)+"document.getElementById('py%sboxmat').setAttribute('transparency', '1');\n"%i)
                self.visfile.write(misc.tabs(5)+"document.getElementById('label_pybox%s').setAttribute('transparency', '1');\n"%i)
            self.visfile.write(misc.tabs(4)+"}\n")
            self.visfile.write(misc.tabs(3)+"}\n")
        if cone is not None:
            self.visfile.write(misc.tabs(3)+"function setcone() {\n")
            self.visfile.write(misc.tabs(4)+"if(document.getElementById('py0conmat').getAttribute('transparency') != '0') {\n")
            for i in range(len(cone)):
                self.visfile.write(misc.tabs(5)+"document.getElementById('py%sconmat').setAttribute('transparency', '0');\n"%i)
                self.visfile.write(misc.tabs(5)+"document.getElementById('label_pycon%s').setAttribute('transparency', '0');\n"%i)
            self.visfile.write(misc.tabs(4)+"} else {\n")
            for i in range(len(cone)):
                self.visfile.write(misc.tabs(5)+"document.getElementById('py%sconmat').setAttribute('transparency', '1');\n"%i)
                self.visfile.write(misc.tabs(5)+"document.getElementById('label_pycon%s').setAttribute('transparency', '1');\n"%i)
            self.visfile.write(misc.tabs(4)+"}\n")
            self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(2)+"</script>\n")


    def buttons(self, viewpoint=None, tube=None, sphere=None, box=None, cone=None, centrot=False):
        """
        Makes the buttons to apply different functions in the web page.

        Parameters
        ----------
        viewpoint : list
            List of names for the viewpoints. Viewpoints must be created in "viewpoints()".
        tube : list
            List with the points parameter included in make_markers() for tubes.
        sphere : list
            List with the points parameter included in make_markers() for spheres.
        box : list
            List with the points parameter included in make_markers() for boxes.
        cone : list
            List with the points parameter included in make_markers() for cones.
        centrot : bool
            Create button to change the center of rotation.
        """
        self.visfile.write(misc.tabs(1)+'<div style="width:90%">\n')
        self.visfile.write(misc.tabs(2)+'<br/>\n')
        # Viewpoint buttons
        self.visfile.write(misc.tabs(2)+"&nbsp <b>Viewpoints:</b>\n")
        self.visfile.write(misc.tabs(3)+"<button onclick=\"document.getElementById('cubeFixed').runtime.resetView();\">Reset View</button>\n")
        for i in range(3): # 3 w/o perspective, 4 with
            self.visfile.write(misc.tabs(3)+"<button onclick=\"document.getElementById('%s').setAttribute('set_bind','true');\"> %s </button>\n"%(misc.side[i], misc.nam[i]))
        if viewpoint is not None:
            for i,name in enumerate(viewpoint):
                self.visfile.write(misc.tabs(3)+f"<button onclick=\"document.getElementById('vp{i}').setAttribute('set_bind','true');\"> {name} </button>\n")

        #unccoment next line for next view button
        #self.visfile.write('\t\t   <button onclick="document.getElementById('\'cubeFixed\').runtime.nextView();">Next View</button>\n')
        
        # Grids and ax labels
        self.visfile.write('\n'+misc.tabs(2)+'<br><br>\n')
        self.visfile.write('\n'+misc.tabs(2)+'&nbsp <b>Labels:</b>\n')
        self.visfile.write(misc.tabs(3)+'<button onclick="setgrids();" >Grids</button>\n')
        self.visfile.write(misc.tabs(3)+'<button onclick="setaxes();" >Axes labels</button>\n')

        if self.cube.galaxies is not None:
            self.visfile.write(misc.tabs(3)+'<button onclick="setgals();" >Galaxies</button>\n')
            self.visfile.write(misc.tabs(3)+'<button onclick="setgallabels();" >Galaxy Labels</button>\n')

        if self.cube.image2d[1] is not None:
            self.visfile.write(misc.tabs(3)+'<button onclick="setimage2d();" >2D image</button>\n')
        
        if centrot:
            self.visfile.write(misc.tabs(2)+'&nbsp <label for="rotationCenter"><b>Center of Rotation</b> </label>\n')
            self.visfile.write(misc.tabs(3)+'<select id="rotationCenter">\n')
            self.visfile.write(misc.tabs(4)+'<option value="Origin">Origin</option>\n')
            for nc in range(len(self.cube.l_isolevels)):
                if isinstance(self.cube.lines, dict) or isinstance(self.cube.lines, list):
                    llab = list(self.cube.lines)[nc]
                    self.visfile.write(misc.tabs(4)+'<option value="op%s">%s</option>\n'%(nc,llab))
                else:
                    self.visfile.write(misc.tabs(4)+'<option value="op%s">Center%s</option>\n'%(nc,nc))
            self.visfile.write(misc.tabs(3)+'</select>\n')
            
        self.visfile.write(misc.tabs(3)+'<button id="anim" onclick="animation()">Animation</button>')

        if tube is not None or sphere is not None or box is not None or cone is not None:
            self.visfile.write(misc.tabs(2)+'<br><br>\n')
            self.visfile.write(misc.tabs(2)+'&nbsp <b>Markers:</b>\n')
            if tube is not None:
                self.visfile.write(misc.tabs(3)+'<button onclick="settube();">Tubes</button>\n')
            if sphere is not None:
                self.visfile.write(misc.tabs(3)+'<button onclick="setsphere();">Spheres</button>\n')
            if box is not None:
                self.visfile.write(misc.tabs(3)+'<button onclick="setbox();">Boxes</button>\n')
            if cone is not None:
                self.visfile.write(misc.tabs(3)+'<button onclick="setcone();">Cones</button>\n')
        
        nlayers = [len(l) for l in self.cube.l_isolevels]
        numcubes = len(nlayers)

        for nc in range(numcubes):
            self.visfile.write(misc.tabs(2)+'<br><br>\n')
            rgbcolors = misc.create_colormap(self.cube.cmaps[nc], self.cube.l_isolevels[nc])
            if isinstance(self.cube.lines, dict):
                llab = list(self.cube.lines)[nc]
                llab = f'{llab} ({self.cube.lines[llab][0]})'
                self.visfile.write(misc.tabs(2)+'&nbsp <b>%s (%s):</b>\n'%(llab,self.cube.units[0]))
            elif isinstance(self.cube.lines, list):
                llab = list(self.cube.lines)[nc]
                self.visfile.write(misc.tabs(2)+'&nbsp <b>%s (%s):</b>\n'%(llab,self.cube.units[0]))
            else:
                self.visfile.write(misc.tabs(2)+'&nbsp <b>Cube %s (%s):</b>\n'%(nc,self.cube.units[0]))
            for i in range(nlayers[nc]):
                ca = np.array(rgbcolors[i].split(' ')).astype(float)*255
                c = 'rgb('+str(ca.astype(int))[1:-1]+')'
                butlabel = self.cube.l_isolevels[nc][i]
                if self.cube.l_isolevels[nc][i] > 0.01:
                    butlabel = f'{butlabel:0.3f}'
                else:
                    butlabel = f'{butlabel:0.3e}'
                if (ca[0]*0.299 + ca[1]*0.587 + ca[2]*0.114) > 130:
                    self.visfile.write(misc.tabs(3)+'<button id="%sbut%s" onclick="setHI%slayer%s();" style="font-size:20px ; border:5px dashed black ; background:%s ; color:black"><b>%s</b></button>\n'%(nc,i,nc,i,c,butlabel))
                else:
                    self.visfile.write(misc.tabs(3)+'<button id="%sbut%s" onclick="setHI%slayer%s();" style="font-size:20px ; border:5px dashed black ; background:%s ; color:white"><b>%s</b></button>\n'%(nc,i,nc,i,c,butlabel))
            self.visfile.write(misc.tabs(3)+f'&nbsp <button id="all" onclick="hideall{nc}()"><b>Invert</b></button>')
                    
        # to separate buttons in two parts
        #if self.grids or self.gals or self.gallabs or self.axes or self.hclick or colormaps is not None:
            #self.visfile.write('\n\t <div style="position:absolute;left:800px;top:140px;width:600px">\n')
        
        # Colormaps
        for nc in range(numcubes):
            self.visfile.write(misc.tabs(2)+'<br><br>\n')
            if isinstance(self.cube.lines, dict) or isinstance(self.cube.lines, list):
                llab = list(self.cube.lines)[nc]
                self.visfile.write(misc.tabs(2)+'&nbsp <label for="cmaps-choice%s"><b>Cmap %s</b> </label>\n'%(nc,llab))
            else:
                self.visfile.write(misc.tabs(2)+'&nbsp <label for="cmaps-choice%s"><b>Cmap %s</b> </label>\n'%(nc,nc))
            self.visfile.write(misc.tabs(2)+'<select id="cmaps-choice%s">\n'%nc)
            self.visfile.write(misc.tabs(3)+f'<option value="{self.cube.cmaps[nc]}">{self.cube.cmaps[nc]}</option>\n')
            for c in misc.default_cmaps:
                if c != self.cube.cmaps[nc]:
                    self.visfile.write(misc.tabs(3)+'<option value="%s">%s</option>\n'%(c,c))
            self.visfile.write(misc.tabs(2)+'</select>\n')
            self.visfile.write(misc.tabs(2) + '<label for="cmaps-min%s"><b>Min %s:</b> </label>\n'%(nc,nc))
            self.visfile.write(misc.tabs(2) + '<input id="cmaps-min%s" type="number" value="%s">\n'%(nc, np.min(self.cube.l_isolevels[nc])))
            self.visfile.write(misc.tabs(2) + '<label for="cmaps-max%s"><b>Max %s:</b> </label>\n'%(nc,nc))
            self.visfile.write(misc.tabs(2) + '<input id="cmaps-max%s" type="number" value="%s">\n'%(nc, np.max(self.cube.l_isolevels[nc])))
            self.visfile.write(misc.tabs(2) + '<label for="cmaps-min%s"><b>Scale %s:</b> </label>\n'%(nc,nc))
            self.visfile.write(misc.tabs(2) + '<select id="cmaps-scale%s">\n'%nc)
            self.visfile.write(misc.tabs(3) + '<option value="linear" selected="selected">linear</option>\n')
            self.visfile.write(misc.tabs(3) + '<option value="log">log</option>\n')
            self.visfile.write(misc.tabs(3) + '<option value="sqrt">sqrt</option>\n')
            self.visfile.write(misc.tabs(3) + '<option value="power">power</option>\n')
            self.visfile.write(misc.tabs(3) + '<option value="asinh">asinh</option>\n')
            self.visfile.write(misc.tabs(2) + '</select>\n')

        # Opacity
        self.visfile.write('\n'+misc.tabs(2)+'<br><br>\n')
        self.visfile.write(misc.tabs(3)+f'&nbsp <button id="all" onclick="changeopa()"><b>Opacity</b></button>')

        # Background 
        self.visfile.write(misc.tabs(3)+'&nbsp <label for="back-choice"><b>Background:</b> </label>\n')
        self.visfile.write(misc.tabs(3)+'<input oninput="change_background()" id="back-choice" type="color" value="#999999">\n')

        # SCALEV
        #self.visfile.write(misc.tabs(2)+'<br><br>\n')
        self.visfile.write(misc.tabs(2)+f'&nbsp <label for="scalev"><b>{self.cube.mags[3]} scale:</b> </label>\n')
        self.visfile.write(misc.tabs(2)+'<input oninput="changescalev()" id="scalev" type="range" list="marker" min="0" max="10" step="0.001" value="1"></input>\n')
        self.visfile.write(misc.tabs(2)+'<datalist id="marker">\n')
        self.visfile.write(misc.tabs(3)+'<option value="1"></option>\n')
        self.visfile.write(misc.tabs(2)+'</datalist>\n')
        
        # Galaxies Font Size
        if self.cube.galaxies is not None:
            self.visfile.write(misc.tabs(3)+'&nbsp <label for="galsize-choice"><b>Galaxy size: </b></label>\n')
            self.visfile.write(misc.tabs(3)+'<input oninput="change_galsize()" id="galsize-choice" type="number" min="2" max="50" value="5", step="1">\n')
            
        if self.cube.image2d[1] is not None:
            #self.visfile.write('\t\t <br><br>\n')
            self.visfile.write(misc.tabs(2)+'&nbsp <label for="move2dimg"><b>2D image:</b> </label>\n')
            self.visfile.write(misc.tabs(2)+'<input oninput="move2d()" id="move2dimg" type="range" min="-1" max="1" step="0.0001" value="1"></input>\n')
            self.visfile.write(misc.tabs(2)+f'<b>${self.cube.mags[3]}=$</b> <output id="showvalue"></output> {self.cube.units[3]}\n')
            # display chosen velocity of bar too

        # PICKING
        # self.visfile.write(misc.tabs(2)+'<br><br>\n')
        # self.visfile.write(misc.tabs(2)+'&nbsp <label for="clickcoords"><b>Click coordinates:</b></label>\n')
        # self.visfile.write(misc.tabs(2)+'<table id="clickcoords">\n\t\t <tbody>\n')
        # self.visfile.write(misc.tabs(3)+'<tr><td>&nbsp RA: </td><td id="coordX">--</td></tr>\n')
        # self.visfile.write(misc.tabs(3)+'<tr><td>&nbsp Dec: </td><td id="coordY">--</td></tr>\n')
        # self.visfile.write(misc.tabs(3)+'<tr><td>&nbsp V: </td><td id="coordZ">--</td></tr>\n')
        # self.visfile.write(misc.tabs(2)+'</tbody></table>\n')
            
        self.visfile.write(misc.tabs(1)+'</div>\n')
        
    def func_move2dimage(self):
        """
        Make JS function to move the 2D image along the spectral axis.
        The X3D file must have a 2D image for this to work.
        Must be after buttons()
        """
        self.visfile.write(misc.tabs(2)+"<script>\n")
        self.visfile.write(misc.tabs(2)+"const inpscam2 = document.querySelector('#scalev');\n")
        self.visfile.write(misc.tabs(2)+"const inpmovem2 = document.querySelector('#move2dimg');\n")
        self.visfile.write(misc.tabs(2)+"const showval = document.querySelector('#showvalue');\n")
        self.visfile.write(misc.tabs(2)+"function move2d()\n\t\t{\n")
        self.visfile.write(misc.tabs(3)+"const sca = inpscam2.value;\n")
        self.visfile.write(misc.tabs(3)+"const move = inpmovem2.value;\n")
        self.visfile.write(misc.tabs(3)+f"showval.textContent = roundTo({self.cube.coords[2][1]}"+\
                             f"+(move-1)*{self.cube.l_cubes[0].shape[2]}*{np.abs(self.cube.delta[2])}/2, 3);\n")
        self.visfile.write(misc.tabs(3)+"document.getElementById('image2d').setAttribute('translation', '0 0 '+(sca*move-1)*1000);\n")
        self.visfile.write(misc.tabs(2)+"}\n")
        self.visfile.write(misc.tabs(2)+"</script>\n")

    def func_galsize(self):
        """
        Make JS function to change the size of the galaxy markers and their labels.
        The X3D file must have galaxies for this to work.
        Must be after buttons()
        """
        self.visfile.write(misc.tabs(2)+"<script>\n")
        self.visfile.write(misc.tabs(3)+"const galsize = document.querySelector('#galsize-choice');\n")
        self.visfile.write(misc.tabs(3)+"function change_galsize() {\n")
        self.visfile.write(misc.tabs(4)+"let gals = %s;\n"%str(list(self.cube.galaxies.keys())))
        self.visfile.write(misc.tabs(4)+"const gsval = galsize.value/8.;\n")
        self.visfile.write(misc.tabs(4)+"for (const gal of gals) {\n")
        self.visfile.write(misc.tabs(5)+"document.getElementById(gal+'_sphere_tra').setAttribute('scale', gsval.toString()+' '+gsval.toString()+' '+gsval.toString());\n")
        self.visfile.write(misc.tabs(5)+"document.getElementById(gal+'_fs').setAttribute('size', gsval*8);\n")
        self.visfile.write(misc.tabs(5)+"}\n")
        self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(2)+"</script>\n")

    def func_setCenterOfRotation(self, centers):
        """
        Make JS function to change the center of rotation from given options.
        Must be after buttons()

        Parameters
        ----------
        centers : list
            List of strings with the coordinates (as different from the centre) of the centers of rotation to be added. E.g. ["0 10 0", "10 0 10"]
        """
        self.visfile.write(misc.tabs(2)+"<script>\n")
        self.visfile.write(misc.tabs(3)+"cor = document.querySelector('#rotationCenter');\n")
        self.visfile.write(misc.tabs(3)+"cor.addEventListener('change', setCenterOfRotation);\n")
        self.visfile.write(misc.tabs(2)+"function setCenterOfRotation() {\n")
        self.visfile.write(misc.tabs(3)+"const cen = cor.value;\n")
        self.visfile.write(misc.tabs(3)+"if (cen === 'Origin') {\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('front').setAttribute('centerOfRotation','0 0 0');\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('side').setAttribute('centerOfRotation','0 0 0');\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('side2').setAttribute('centerOfRotation','0 0 0');\n")
        self.visfile.write(misc.tabs(3)+"}\n")
        for nc in range(len(centers)):
            self.visfile.write(misc.tabs(3)+"else if (cen === 'op%s') {\n"%nc)
            self.visfile.write(misc.tabs(4)+"document.getElementById('front').setAttribute('centerOfRotation', '%s');\n"%centers[nc])
            self.visfile.write(misc.tabs(4)+"document.getElementById('side').setAttribute('centerOfRotation', '%s');\n"%centers[nc])
            self.visfile.write(misc.tabs(4)+"document.getElementById('side2').setAttribute('centerOfRotation', '%s');\n"%centers[nc])
            self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(2)+"}\n"+misc.tabs(2)+"</script>\n")
        
    def func_colormaps(self):
        """
        Make JS function to change the colormap of the layers.
        Must be after buttons()
        """
        self.visfile.write("\t\t <!--MUST BE BELOW THE <select> ELEMENT-->\n")
        numcubes = len(self.cube.l_isolevels)
        for nc in range(numcubes):
            self.visfile.write(misc.tabs(2)+"<script>\n")
            self.visfile.write(misc.tabs(3)+"const cc%s = document.querySelector('#cmaps-choice%s');\n"%(nc,nc))
            self.visfile.write(misc.tabs(3)+"cc%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.visfile.write(misc.tabs(3)+"const cmapmin%s = document.querySelector('#cmaps-min%s');\n"%(nc,nc))
            self.visfile.write(misc.tabs(3)+"cmapmin%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.visfile.write(misc.tabs(3)+"const cmapmax%s = document.querySelector('#cmaps-max%s');\n"%(nc,nc))
            self.visfile.write(misc.tabs(3)+"cmapmax%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.visfile.write(misc.tabs(3)+"const cmapscale%s = document.querySelector('#cmaps-scale%s');\n"%(nc,nc))
            self.visfile.write(misc.tabs(3)+"cmapscale%s.addEventListener('change', change_colormap%s);\n"%(nc,nc))
            self.visfile.write(misc.tabs(3)+"const isolevels%s = %s;\n"%(nc,repr(self.cube.l_isolevels[nc]).replace('array(','').replace(')','')))

            self.visfile.write(misc.tabs(3)+"function change_colormap%s() {\n"%nc)
            self.visfile.write(misc.tabs(4)+"var cmap%s = cc%s.value;\n"%(nc,nc))
            self.visfile.write(misc.tabs(4)+"var min%s = cmapmin%s.value;\n"%(nc,nc))
            self.visfile.write(misc.tabs(4)+"var max%s = cmapmax%s.value;\n"%(nc,nc))
            self.visfile.write(misc.tabs(4)+"const scale%s = cmapscale%s.value;\n"%(nc,nc))
            self.visfile.write(misc.tabs(4)+"var reverse%s = false;\n"%(nc))
            self.visfile.write(misc.tabs(4)+"var collevs%s = [];\n"%(nc))

            self.visfile.write(misc.tabs(4)+"if (scale%s === 'linear') {\n"%(nc))
            self.visfile.write(misc.tabs(5)+"for (const level of isolevels%s) {\n"%(nc))
            self.visfile.write(misc.tabs(6)+"collevs%s.push(level);\n"%(nc))
            self.visfile.write(misc.tabs(5)+"}\n")
            self.visfile.write(misc.tabs(4)+"}\n")
            self.visfile.write(misc.tabs(4)+"else if (scale%s === 'log') {\n"%(nc))
            self.visfile.write(misc.tabs(5)+"min%s = Math.log(min%s);\n"%(nc,nc))
            self.visfile.write(misc.tabs(5)+"max%s = Math.log(max%s);\n"%(nc,nc))
            self.visfile.write(misc.tabs(5)+"for (const level of isolevels%s) {\n"%(nc))
            self.visfile.write(misc.tabs(6)+"collevs%s.push(Math.log(level));\n"%nc)
            self.visfile.write(misc.tabs(5)+"}\n")
            self.visfile.write(misc.tabs(4)+"}\n")
            self.visfile.write(misc.tabs(4)+"else if (scale%s === 'sqrt') {\n"%(nc))
            self.visfile.write(misc.tabs(5)+"min%s = Math.sqrt(min%s);\n"%(nc,nc))
            self.visfile.write(misc.tabs(5)+"max%s = Math.sqrt(max%s);\n"%(nc,nc))
            self.visfile.write(misc.tabs(5)+"for (const level of isolevels%s) {\n"%(nc))
            self.visfile.write(misc.tabs(6)+"collevs%s.push(Math.sqrt(level));\n"%(nc))
            self.visfile.write(misc.tabs(5)+"}\n")
            self.visfile.write(misc.tabs(4)+"}\n")
            self.visfile.write(misc.tabs(4)+"else if (scale%s === 'power') {\n"%nc)
            self.visfile.write(misc.tabs(5)+"min%s = Math.pow(min%s, 2);\n"%(nc,nc))
            self.visfile.write(misc.tabs(5)+"max%s = Math.pow(max%s, 2);\n"%(nc,nc))
            self.visfile.write(misc.tabs(5)+"for (const level of isolevels%s) {\n"%nc)
            self.visfile.write(misc.tabs(6)+"collevs%s.push(Math.pow(level, 2));\n"%nc)
            self.visfile.write(misc.tabs(5)+"}\n")
            self.visfile.write(misc.tabs(4)+"}\n")
            self.visfile.write(misc.tabs(4)+"else if (scale%s === 'asinh') {\n"%nc)
            self.visfile.write(misc.tabs(5)+"min%s = Math.asinh(min%s);\n"%(nc,nc))
            self.visfile.write(misc.tabs(5)+"max%s = Math.asinh(max%s);\n"%(nc,nc))
            self.visfile.write(misc.tabs(5)+"for (const level of isolevels%s) {\n"%nc)
            self.visfile.write(misc.tabs(6)+"collevs%s.push(Math.asinh(level));\n"%nc)
            self.visfile.write(misc.tabs(5)+"}\n")
            self.visfile.write(misc.tabs(4)+"}\n")
            self.visfile.write(misc.tabs(4)+"for (let lev = 0; lev < isolevels%s.length; lev++) {\n"%nc)
            self.visfile.write(misc.tabs(5)+"if (min%s >= collevs%s[lev]) {\n"%(nc,nc))
            self.visfile.write(misc.tabs(6)+"collevs%s[lev] = 0;\n"%nc)
            self.visfile.write(misc.tabs(5)+"} else if (max%s <= collevs%s[lev]) {\n"%(nc,nc))
            self.visfile.write(misc.tabs(6)+"collevs%s[lev] = 1;\n"%nc)
            self.visfile.write(misc.tabs(5)+"} else {\n")
            self.visfile.write(misc.tabs(6)+"collevs%s[lev] = (collevs%s[lev] - min%s) / (max%s - min%s);\n"%(nc,nc,nc,nc,nc))
            self.visfile.write(misc.tabs(5)+"}\n")
            self.visfile.write(misc.tabs(5)+"if (cmap%s.endsWith('_r')) {\n"%nc)
            self.visfile.write(misc.tabs(6)+"reverse%s = true\n"%nc)
            self.visfile.write(misc.tabs(5)+"}\n")
            self.visfile.write(misc.tabs(5)+"if (cmap%s.includes('gist_rainbow')) {\n"%nc)
            self.visfile.write(misc.tabs(6)+"var color%s = evaluate_cmap(collevs%s[lev], 'gist_rainbow', reverse%s);\n"%(nc,nc,nc))
            self.visfile.write(misc.tabs(5)+"} else {\n")
            self.visfile.write(misc.tabs(6)+"var color%s = evaluate_cmap(collevs%s[lev], cmap%s.replace('_r', ''), reverse%s);\n"%(nc,nc,nc,nc))
            self.visfile.write(misc.tabs(5)+"}\n")
            self.visfile.write(misc.tabs(5)+"var split%s = %s;\n"%(nc,repr(self.cube.iso_split[nc]).replace('array(','').replace(')','')))
            self.visfile.write(misc.tabs(5)+"for (let sp = 0; sp < split%s[lev]; sp++) {\n"%nc)
            self.visfile.write(misc.tabs(6)+"document.getElementById('%slayer'+lev+'_sp'+sp).setAttribute('diffuseColor', color%s[0]/255+' '+color%s[1]/255+' '+color%s[2]/255);\n"%(nc,nc,nc,nc))
            self.visfile.write(misc.tabs(5)+"}\n")
            self.visfile.write(misc.tabs(5)+"document.getElementById('%sbut'+lev).style.background = 'rgb('+color%s[0]+' '+color%s[1]+' '+color%s[2]+')';\n"%(nc,nc,nc,nc))
            self.visfile.write(misc.tabs(5)+"if ((color%s[0]*0.299 + color%s[1]*0.587 + color%s[2]*0.114) > 130) {\n"%(nc,nc,nc))
            self.visfile.write(misc.tabs(6)+"document.getElementById('%sbut'+lev).style.color = 'black';\n"%nc)
            self.visfile.write(misc.tabs(5)+"} else {\n")
            self.visfile.write(misc.tabs(6)+"document.getElementById('%sbut'+lev).style.color = 'white';\n"%nc)
            self.visfile.write(misc.tabs(5)+"}\n")
            self.visfile.write(misc.tabs(4)+"}\n")
            self.visfile.write(misc.tabs(3)+"}\n")
            self.visfile.write(misc.tabs(2)+"</script>\n")
            
            
    def func_scalev(self, sphere=None, box=None, cone=None, tube=None, trans=None):
        """
        Make JS funtion to change the scale of the spectral axis.
        Must be after buttons().

        Parameters
        ----------
        tube : list
            List with the points parameter included in make_markers() for tubes.
        sphere : list
            List with the points parameter included in make_markers() for spheres.
        box : list
            List with the points parameter included in make_markers() for boxes.
        cone : list
            List with the points parameter included in make_markers() for cones.
        delta : list
            List with the delta of the cube (header[CDELT]).
        trans : list
            Transformation factor to the cube. It is calculated with (2000/nx, 2000/ny, 2000/nz).
        """
        self.visfile.write(misc.tabs(2)+"<script>\n")
        self.visfile.write(misc.tabs(2)+"const inpscasv = document.querySelector('#scalev');\n")
        if self.cube.image2d[1] is not None:
            self.visfile.write(misc.tabs(2)+"const inpmovesv = document.querySelector('#move2dimg');\n")
        #self.visfile.write(misc.tabs(2)+"inpscasv.addEventListener('change', changescalev);\n")
        self.visfile.write(misc.tabs(2)+"function changescalev()\n\t\t {\n")
        self.visfile.write(misc.tabs(3)+"const sca = inpscasv.value;\n")
        if self.cube.image2d[1] is not None:
            self.visfile.write(misc.tabs(3)+"const move = inpmovesv.value;\n")
            self.visfile.write(misc.tabs(4)+f"document.getElementById('image2d').setAttribute('translation', '0 0 '+(sca*move-1)*1000);\n")
        #scale layers
        nlayers = [len(l) for l in self.cube.l_isolevels]
        numcubes = len(nlayers)
        for nc in range(numcubes):
            for nlays in range(nlayers[nc]):
                for sp in range(self.cube.iso_split[nc][nlays]):
                    self.visfile.write(misc.tabs(3)+"document.getElementById('%slt%s_sp%s').setAttribute('scale', '1 1 '+sca);\n"%(nc,nlays,sp))
        #scale outline
        self.visfile.write(misc.tabs(3)+"document.getElementById('ot').setAttribute('scale', '1 1 '+sca);\n")
        #move galaxies
        if self.cube.galaxies is not None:
            for (n,gal) in enumerate(self.cube.galaxies):
                v = self.cube.galaxies[gal]['coord'][2]
                a = self.cube.galaxies[gal]['coord'][0]
                b = self.cube.galaxies[gal]['coord'][1]
                # self.visfile.write(misc.tabs(3)+"document.getElementById('%s_cross_tra').setAttribute('translation', '0 0 '+(sca-1)*%s);\n"%(gal,v))
                self.visfile.write(misc.tabs(3)+"document.getElementById('%s_sphere_tra').setAttribute('translation', '%s %s '+sca*%s);\n"%(gal,a,b,v))
                self.visfile.write(misc.tabs(3)+"document.getElementById('glt%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(n,a,b,v))
        #scale grids
        self.visfile.write("\t\t\t document.getElementById('tlt').setAttribute('scale', '1 1 '+sca);\n")

        #scale markers
        # spheres
        self.visfile.write(misc.tabs(1)+"if (document.getElementById('new-sphere')[0].value != 'none') {\n")
        self.visfile.write(misc.tabs(2)+"for (s=0; s<document.getElementById('new-sphere').length; s++) {\n")
        self.visfile.write(misc.tabs(3)+"const sphInd = document.getElementById('new-sphere')[s].value.slice(3);\n")
        self.visfile.write(misc.tabs(3)+"document.getElementById('sphtra'+sphInd).setAttribute('translation', sph_coords[sphInd-1][0]+' '+sph_coords[sphInd-1][1]+' '+sca*sph_coords[sphInd-1][2]);\n")
        self.visfile.write(misc.tabs(3)+"document.getElementById('sphlabtra'+sphInd).setAttribute('translation', sph_coords[sphInd-1][0]+' '+sph_coords[sphInd-1][1]+' '+sca*sph_coords[sphInd-1][2]);\n")
        self.visfile.write(misc.tabs(2)+"}\n")
        self.visfile.write(misc.tabs(1)+"}\n")
        # boxes
        self.visfile.write(misc.tabs(1)+"if (document.getElementById('new-box')[0].value != 'none') {\n")
        self.visfile.write(misc.tabs(2)+"for (s=0; s<document.getElementById('new-box').length; s++) {\n")
        self.visfile.write(misc.tabs(3)+"const boxInd = document.getElementById('new-box')[s].value.slice(3);\n")
        self.visfile.write(misc.tabs(3)+"document.getElementById('boxtra'+boxInd).setAttribute('translation', box_coords[boxInd-1][0]+' '+box_coords[boxInd-1][1]+' '+sca*box_coords[boxInd-1][2]);\n")
        self.visfile.write(misc.tabs(3)+"document.getElementById('boxlabtra'+boxInd).setAttribute('translation', box_coords[boxInd-1][0]+' '+box_coords[boxInd-1][1]+' '+sca*box_coords[boxInd-1][2]);\n")
        self.visfile.write(misc.tabs(2)+"}\n")
        self.visfile.write(misc.tabs(1)+"}\n")
        # cones
        self.visfile.write(misc.tabs(1)+"if (document.getElementById('new-con')[0].value != 'none') {\n")
        self.visfile.write(misc.tabs(2)+"for (s=0; s<document.getElementById('new-con').length; s++) {\n")
        self.visfile.write(misc.tabs(3)+"const conInd = document.getElementById('new-con')[s].value.slice(3);\n")
        self.visfile.write(misc.tabs(3)+"document.getElementById('contra'+conInd).setAttribute('translation', con_coords[conInd-1][0]+' '+con_coords[conInd-1][1]+' '+sca*con_coords[conInd-1][2]);\n")
        self.visfile.write(misc.tabs(3)+"document.getElementById('conlabtra'+conInd).setAttribute('translation', con_coords[conInd-1][0]+' '+con_coords[conInd-1][1]+' '+sca*con_coords[conInd-1][2]);\n")
        self.visfile.write(misc.tabs(3)+"// missing orientation change (don't implement?)\n")
        self.visfile.write(misc.tabs(2)+"}\n")
        self.visfile.write(misc.tabs(1)+"}\n")
        # tubes
        self.visfile.write(misc.tabs(1)+"if (document.getElementById('new-tub')[0].value != 'none') {\n")
        self.visfile.write(misc.tabs(2)+"for (s=0; s<document.getElementById('new-tub').length; s++) {\n")
        self.visfile.write(misc.tabs(3)+"const tubInd = document.getElementById('new-tub')[s].value.slice(3);\n")
        self.visfile.write(misc.tabs(3)+"for (i=1; i<tubelen[tubInd-1]; i++) {\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('tubtra'+tubInd+'_'+i).setAttribute('translation', tub_coords[tubInd-1][i-1][0][0]+' '+tub_coords[tubInd-1][i-1][0][1]+' '+sca*tub_coords[tubInd-1][i-1][0][2]);\n")
        self.visfile.write(misc.tabs(4)+"const norm = Math.sqrt(tub_coords[tubInd-1][i-1][1][0]**2+tub_coords[tubInd-1][i-1][1][1]**2+(tub_coords[tubInd-1][i-1][1][2]*sca)**2)*1.015;\n")
        self.visfile.write(misc.tabs(4)+"const angle = Math.acos(tub_coords[tubInd-1][i-1][1][1]/norm);\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('tubtra'+tubInd+'_'+i).setAttribute('rotation', tub_coords[tubInd-1][i-1][1][2]*sca+' 0 '+(-tub_coords[tubInd-1][i-1][1][0])+' '+angle);\n")
        self.visfile.write(misc.tabs(4)+"document.getElementById('tub'+tubInd+'_'+i).setAttribute('height', norm);\n")
        self.visfile.write(misc.tabs(3)+"}\n")
        self.visfile.write(misc.tabs(3)+"document.getElementById('tublabtra'+tubInd).setAttribute('translation', tub_coords[tubInd-1][0][0][0]+' '+tub_coords[tubInd-1][0][0][1]+' '+sca*tub_coords[tubInd-1][0][0][2]);\n")
        self.visfile.write(misc.tabs(2)+"}\n")
        self.visfile.write(misc.tabs(1)+"}\n")

        # scale python markers
        delta = np.array(self.cube.delta)
        if sphere is not None:
            n = 0
            for s in range(len(sphere)):
                sphere[s] = (sphere[s] - np.mean(self.cube.coords, axis=1))/delta*trans
                self.visfile.write(misc.tabs(2)+f"document.getElementById('pysph{n}tra').setAttribute('translation', '{sphere[s][0]} {sphere[s][1]} '+sca*{sphere[s][2]});\n")
                self.visfile.write(misc.tabs(2)+f"document.getElementById('label_pysphtra{n}').setAttribute('translation', '{sphere[s][0]} {sphere[s][1]} '+sca*{sphere[s][2]});\n")
                n = n + 1
        if box is not None:
            n = 0
            for b in range(len(box)):
                box[b] = (box[b] - np.mean(self.cube.coords, axis=1))/delta*trans
                self.visfile.write(misc.tabs(2)+f"document.getElementById('pybox{n}tra').setAttribute('translation', '{box[b][0]} {box[b][1]} '+sca*{box[b][2]});\n")
                self.visfile.write(misc.tabs(2)+f"document.getElementById('label_pyboxtra{n}').setAttribute('translation', '{box[b][0]} {box[b][1]} '+sca*{box[b][2]});\n")
                n = n + 1
        if cone is not None:
            n = 0
            for c in range(len(cone)):
                cone[c] = (cone[c] - np.mean(self.cube.coords, axis=1))/delta*trans
                self.visfile.write(misc.tabs(2)+f"document.getElementById('pycon{n}tra').setAttribute('translation', '{cone[c][0]} {cone[c][1]} '+sca*{cone[c][2]});\n")
                self.visfile.write(misc.tabs(2)+f"document.getElementById('label_pycontra{n}').setAttribute('translation', '{cone[c][0]} {cone[c][1]} '+sca*{cone[c][2]});\n")
                n = n + 1
        # missing orientation change for cones
        if tube is not None:
            n = 0
            for t in tube:
                t = (t - np.mean(self.cube.coords, axis=1))/delta*trans
                pos = np.array([np.mean((t[i],t[i+1]), axis=0) for i in range(len(t)-1)])
                diff = np.diff(t, axis=0)

                for i in range(len(t)-1):
                    self.visfile.write(misc.tabs(2)+f"const diff{n}_{i} = [{diff[i][0]},{diff[i][1]},{diff[i][2]}];\n")
                    self.visfile.write(misc.tabs(2)+f"const height{n}_{i} = Math.sqrt(diff{n}_{i}[0]**2+diff{n}_{i}[1]**2+(sca*diff{n}_{i}[2])**2)*1.015;\n")
                    self.visfile.write(misc.tabs(2)+f"const angle{n}_{i} = Math.acos(diff{n}_{i}[1]/height{n}_{i});\n")
                    self.visfile.write(misc.tabs(2)+f"document.getElementById('pytub{n}tra{i}').setAttribute('translation', '{pos[i][0]} {pos[i][1]} '+sca*{pos[i][2]});\n")
                    self.visfile.write(misc.tabs(2)+f"document.getElementById('py{n}tub{i}').setAttribute('height', height{n}_{i}.toString());\n")
                    self.visfile.write(misc.tabs(2)+f"document.getElementById('pytub{n}tra{i}').setAttribute('rotation', sca*diff{n}_{i}[2]+' 0 '+(-diff{n}_{i}[0])+' '+angle{n}_{i});\n")
                self.visfile.write(misc.tabs(2)+f"document.getElementById('label_pytubtra{n}').setAttribute('translation', '{pos[0][0]} {pos[0][1]} '+sca*{pos[0][2]});\n")
                n = n +1

        #scale axes
        ax, axtick = misc.labpos
        
        for i in range(12):
            if i < 6:
                self.visfile.write(misc.tabs(3)+"document.getElementById('alt_diff%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, ax[i][0], ax[i][1], ax[i][2])) #str(ax[i])[1:-1]
                if isinstance(self.cube.lines, dict) == False:
                    self.visfile.write(misc.tabs(3)+"document.getElementById('alt_real%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, ax[i][0], ax[i][1], ax[i][2]))
            self.visfile.write(misc.tabs(3)+"document.getElementById('att_diff%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, axtick[i][0], axtick[i][1], axtick[i][2]))
            if isinstance(self.cube.lines, dict) == False:
                self.visfile.write(misc.tabs(3)+"document.getElementById('att_real%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, axtick[i][0], axtick[i][1], axtick[i][2]))
        
        self.visfile.write(misc.tabs(2)+"}\n")
        self.visfile.write(misc.tabs(2)+"</script>\n")
    
    def func_image2d(self):
        """
        Make JS function to show/hide the 2D image.
        The X3D file must have a 2D image for this to work.
        Must be after buttons().
        """
        # PICKING
        # self.visfile.write(roundTo) #premade string with function to round to two decimals

        self.visfile.write(misc.tabs(2)+"<script>\n")
        self.visfile.write(misc.tabs(3)+"var showalertimage2d = true;\n")
        self.visfile.write(misc.tabs(3)+"function setimage2d() {\n")
        self.visfile.write(misc.tabs(4)+"if (showalertimage2d) {\n")
        self.visfile.write(misc.tabs(5)+'alert("The opacity feature does not work alongside the 2D image. Hide the 2D image to change the opacity.")\n')
        self.visfile.write(misc.tabs(5)+"showalertimage2d = false;\n")
        self.visfile.write(misc.tabs(4)+"}\n")
        self.visfile.write(misc.tabs(4)+"if(document.getElementById('image2d').getAttribute('scale') != '1 1 1') {\n")
        self.visfile.write(misc.tabs(5)+"document.getElementById('image2d').setAttribute('scale', '1 1 1');\n")
        self.visfile.write(misc.tabs(4)+"} else {\n")
        self.visfile.write(misc.tabs(5)+"document.getElementById('image2d').setAttribute('scale', '0 0 0');\n")
        self.visfile.write(misc.tabs(4)+"}\n")
        self.visfile.write(misc.tabs(3)+"}\n\t\t</script>\n")

    def close_html(self):
        """
        Must be used to finish and close the HTML file. Not using this function results
        in an error.
        """
        if self.cube.interface != 'minimal':
            self.visfile.write(misc.tablehtml)
        self.visfile.write('\n\t</body>\n</html>')
        self.visfile.close()

    def make_layers(self, shifts=None, add_normals=False):
        """
        Calculate iso-surfaces from the data and write the objects in the X3D file.

        Parameters
        ----------
        shift : list, optional
            A list with a arrays of 3D vectors giving the shift in RA, DEC and spectral axis in
            the same units given to the cube. Similar to l_cube or l_isolevels.
        add_normals : bool, optional
            Whether to add normal vectors in the X3D model. Default is False.
        """
        numcubes = len(self.cube.l_cubes)
        self.cube.iso_split = []

        for nc in range(numcubes):
            cube_full = self.cube.l_cubes[nc]
            isolevels = self.cube.l_isolevels[nc]
            self.cube.iso_split.append(np.zeros((len(isolevels)), dtype=int))
            rgbcolors = misc.create_colormap(self.cube.cmaps[nc], isolevels)
            for (i,lev) in enumerate(isolevels):
                # calculate how many times to split the cube, 1 means the cube stays the same
                split = int(np.sum(cube_full>lev)/700000)+1
                self.cube.iso_split[nc][i] = split
                _, _, nz = cube_full.shape

                for sp in range(split):
                    cube = cube_full[:,:,int(nz/split*sp):int(nz/split*(sp+1))]
                    if lev > np.max(cube) or lev < np.min(cube):
                        print(f'Level {lev} is out of bounds for cube {nc} split {sp}.')
                        verts, faces, normals = None, None, None
                    else:
                        try:
                            if shifts is not None:
                                verts, faces, normals = misc.marching_cubes(cube, level=lev,
                                            shift=shifts[nc], step_size=self.cube.resol)
                            else:
                                verts, faces, normals = misc.marching_cubes(cube, level=lev,
                                                                step_size=self.cube.resol)
                        except Exception as ex:
                            print(ex)
                            continue
                    self.visfile.write('\n'+misc.tabs(3)+f'<Transform id="{nc}lt{i}_sp{sp}" ' \
                                        +' translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">')
                    self.visfile.write('\n'+misc.tabs(4)+f'<Shape id="{nc}layer{i}_sp{sp}_shape">')
                    if self.cube.image2d[1] is not None:
                        sortType = 'transparent'
                    else:
                        sortType = 'opaque'
                    self.visfile.write('\n'+misc.tabs(5)+f'<Appearance id="{nc}layer{i}_sp{sp}_appe" sortType="{sortType}" sortKey="{len(isolevels)-1-i}">')
                    self.visfile.write(f'\n{misc.tabs(6)}<Material id="{nc}layer{i}_sp{sp}" '\
                            + 'ambientIntensity="0" emissiveColor="0 0 0" '\
                            + f'diffuseColor="{rgbcolors[i]}" specularColor=' \
                            +f'"0 0 0" shininess="0.0078" transparency="0.8"></Material>')
                    #correct color with depthmode (ALSO FOR LAST LAYER?)
                    # if i != len(isolevels)-1:
                    self.visfile.write('\n'+misc.tabs(6)+'<DepthMode readOnly="true"></DepthMode>')
                    self.visfile.write('\n'+misc.tabs(5)+'</Appearance>')
                    #define the layer object
                    if verts is not None:
                        if add_normals:
                            self.visfile.write('\n'+misc.tabs(5)+'<IndexedFaceSet solid="false" '\
                            +'colorPerVertex="false" normalPerVertex="true" coordIndex="\n\t\t\t\t\t\t')
                        else:
                            self.visfile.write('\n'+misc.tabs(5)+'<IndexedFaceSet solid="false" colorPerVertex="false" normalPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
                        #write indices
                        np.savetxt(self.visfile, faces, fmt='%i', newline=' -1\n\t\t\t\t\t\t')
                        self.visfile.write('">')
                        self.visfile.write(f'\n\t\t\t\t\t\t<Coordinate id="{nc}Coordinates{i}_sp{sp}" point="\n\t\t\t\t\t\t')
                        #write coordinates
                        np.savetxt(self.visfile, verts,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
                        self.visfile.write('"></Coordinate>')
                        if add_normals:
                            self.visfile.write(f'\n\t\t\t\t\t\t<Normal id="{nc}Normals{i}_sp{sp}" vector="\n\t\t\t\t\t\t')
                            #write normals
                            np.savetxt(self.visfile, normals,fmt='%.5f', newline=',\n\t\t\t\t\t\t')
                            self.visfile.write('"></Normal>')
                        self.visfile.write('\n'+misc.tabs(5)+'</IndexedFaceSet>\n')
                    self.visfile.write(misc.tabs(4)+'</Shape>\n')
                    self.visfile.write(misc.tabs(3)+'</Transform>')

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
        self.visfile.write('\n\t\t\t<Transform id="ot" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">')
        self.visfile.write('\n\t\t\t\t<Shape ispickable="false">')
        self.visfile.write('\n\t\t\t\t\t<Appearance>')
        #define ouline ID
        col = '0 0 0'
        self.visfile.write('\n\t\t\t\t\t\t<Material id="outline" '\
                            +f'emissiveColor="{col}" diffuseColor="0 0 0"></Material>')
        self.visfile.write('\n\t\t\t\t\t</Appearance>')
        self.visfile.write('\n\t\t\t\t\t<IndexedLineSet colorPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
        #write indices
        np.savetxt(self.visfile, misc.outlineindex, fmt='%i', newline='\n\t\t\t\t\t\t')
        self.visfile.write('">')
        self.visfile.write('\n\t\t\t\t\t\t<Coordinate id="OutlineCoords" point="\n\t\t\t\t\t\t')
        #write coordinates
        np.savetxt(self.visfile, outlinecoords,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
        self.visfile.write('"></Coordinate>')
        self.visfile.write('\n\t\t\t\t\t</IndexedLineSet>\n\t\t\t\t</Shape>\n\t\t\t</Transform>\n')

    def make_galaxies(self):
        """
        Creates spheres and labels in the model at the location of galaxies given as input.
        """        
        gals = self.cube.galaxies

        sphereradius = 2000/45
        crosslen = 2000/20
        #create galaxy crosses and spheres
        for i, gal in enumerate(gals.keys()):
            vec = gals[gal]['coord']
            # #galaxy crosses
            # self.visfile.write(misc.tabs(3)+f'<Transform id="{gal}_cross_tra" ' \
            #                     +'translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n')
            # self.visfile.write(misc.tabs(4)+'<Shape ispickable="false">\n')
            # self.visfile.write(misc.tabs(5)+'<Appearance>\n')
            # col = '0 0 0'
            # self.visfile.write(misc.tabs(6)+f'<Material id="{gal}_cross" emissiveColor="{col}" ' \
            #                     + 'diffuseColor="0 0 0"></Material>\n')
            # self.visfile.write(misc.tabs(5)+'</Appearance>\n')
            # #cross indices
            # self.visfile.write(misc.tabs(5)+'<IndexedLineSet colorPerVertex="true" coordIndex="\n' \
            #             +misc.tabs(6)+'0 1 -1\n'+misc.tabs(6)+'2 3 -1\n'+misc.tabs(6)+'4 5 -1\n'+misc.tabs(6)+'">\n')
            # self.visfile.write(f'{misc.tabs(5)}<Coordinate id="CrossCoords{i}" point="\n{misc.tabs(6)}')
            # crosscoords = np.array([[vec[0]-crosslen,vec[1],vec[2]],
            #                   [vec[0]+crosslen,vec[1],vec[2]],
            #                   [vec[0],vec[1]-crosslen, vec[2]],
            #                   [vec[0],vec[1]+crosslen, vec[2]],
            #                   [vec[0],vec[1],vec[2]-crosslen],
            #                   [vec[0],vec[1],vec[2]+crosslen]])
            # #cross coordinates
            # np.savetxt(self.visfile, crosscoords, fmt='%.3f', newline='\n\t\t\t\t\t\t')
            # self.visfile.write(misc.tabs(6)+'"></Coordinate>\n')
            # self.visfile.write(misc.tabs(3)+'</IndexedLineSet>\n\t\t\t\t</Shape>\n\t\t\t</Transform>\n')
            #galaxy spheres (ADD SCALE, ROTATION, ETC.??)
            self.visfile.write(f'{misc.tabs(3)}<Transform id="{gal}_sphere_tra" translation="{vec[0]} {vec[1]} {vec[2]}">\n')
            self.visfile.write(f'{misc.tabs(4)}<Shape ispickable="false">\n')
            self.visfile.write(f'{misc.tabs(5)}<Sphere radius="{sphereradius}" solid="false"></Sphere>\n')
            self.visfile.write(f'{misc.tabs(5)}<Appearance>\n')
            self.visfile.write(f'{misc.tabs(6)}<Material id="{gal}" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="{gals[gal]["col"]}" specularColor="0 0 0" shininess="0.0078" transparency="0"></Material>\n')
            self.visfile.write(misc.tabs(5)+'</Appearance>\n\t\t\t\t</Shape>\n\t\t\t</Transform>\n')

            self.visfile.write(misc.tabs(4)+f'<Transform id="glt{i}" translation="{gals[gal]["coord"][0]} {gals[gal]["coord"][1]} {gals[gal]["coord"][2]}" rotation="0 1 0 3.14" scale="20 20 20">\n') # labelscale = 20
            self.visfile.write(misc.tabs(5)+'<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">\n')
            self.visfile.write(misc.tabs(6)+'<Shape ispickable="false">\n')
            self.visfile.write(misc.tabs(6)+'<Appearance>\n')
            self.visfile.write(misc.tabs(6)+f'<Material id="label_{gal}" diffuseColor="0 0 0" emissiveColor="0 0 0"></Material>\n')
            self.visfile.write(misc.tabs(6)+'</Appearance>\n')
            self.visfile.write(misc.tabs(6)+f'<Text string="{gal}">\n')
            self.visfile.write(misc.tabs(7)+f'<FontStyle id="{gal}_fs" family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="5"></FontStyle>\n')
            self.visfile.write(misc.tabs(6)+'</Text>\n')
            self.visfile.write(misc.tabs(6)+'</Shape>\n')
            self.visfile.write(misc.tabs(5)+'</Billboard>\n')
            self.visfile.write(misc.tabs(4)+'</Transform>\n')
            
    def make_image2d(self):
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
        imcol, img_shape = self.cube.image2d[1]

        # coordinates of 2d image
        coords2d = np.array([[1000,-1000,1000],
                             [1000,1000,1000],
                             [-1000,-1000,1000],
                             [-1000,1000,1000]])
        
        self.visfile.write(misc.tabs(3)+'<Transform id="image2d" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n')
        self.visfile.write(misc.tabs(4)+'<Shape ispickable="False">\n')
        self.visfile.write(misc.tabs(5)+'<Appearance>\n')
        self.visfile.write(misc.tabs(6)+'<Material id="immat" ambientIntensity="1" emissiveColor="0 0 0" diffuseColor="1 1 1" shininess="0.0078"></Material>\n')
        if imcol is not None and img_shape is not None:
            self.visfile.write(f'{misc.tabs(6)}<PixelTexture repeatS="false" repeatT="false" image="{img_shape[0]} {img_shape[1]} 3 \n')
            # write pixel colors
            np.savetxt(self.visfile, imcol, fmt='%s', delimiter=' ', newline='\n')
            self.visfile.write('"></PixelTexture>\n')
        self.visfile.write(misc.tabs(5)+'</Appearance>\n')            
        #SOLID=TRUE makes it transparent from one side
        self.visfile.write(misc.tabs(5)+'<IndexedFaceSet solid="false" colorPerVertex="false" normalPerVertex="false" coordIndex="2 3 1 0 -1">\n')
        self.visfile.write(misc.tabs(6)+'<Coordinate id="imgCoords" point="\n\t\t\t\t\t\t')
        # write coordinates
        np.savetxt(self.visfile, coords2d, fmt='%.3f', newline='\n\t\t\t\t\t\t')
        self.visfile.write('"></Coordinate>\n')
        self.visfile.write(misc.tabs(6)+'<TextureCoordinate id="imgTexCoords" point="\n'+misc.tabs(6)+' 0 0, 1 0, 0 1, 1 1"></TextureCoordinate>\n')
        self.visfile.write(misc.tabs(5)+'</IndexedFaceSet>\n')
        self.visfile.write(misc.tabs(4)+'</Shape>\n')
        self.visfile.write(misc.tabs(3)+'</Transform>\n')
            
    def make_ticklines(self):
        """
        Create tickline objects in the X3D model.

        Closes the Transform "ROOT" element. Must be called somewhere after make_layers() and before make_labels().
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
        self.visfile.write(misc.tabs(3)+'<Transform id="tlt" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n')
        self.visfile.write(misc.tabs(4)+'<Shape ispickable="false">\n')
        self.visfile.write(misc.tabs(5)+'<Appearance>\n')
        #set color
        col = '0 0 0'
        self.visfile.write(f'{misc.tabs(6)}<Material id="ticklines" emissiveColor="{col}" diffuseColor="0 0 0"></Material>\n')
        self.visfile.write(misc.tabs(5)+'</Appearance>\n')
        self.visfile.write(misc.tabs(5)+'<IndexedLineSet colorPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
        #write indices
        np.savetxt(self.visfile, misc.ticklineindex, fmt='%i', newline='\n\t\t\t\t\t\t')
        self.visfile.write('">\n')
        self.visfile.write(misc.tabs(6)+'<Coordinate id="ticklineCoords" point="\n\t\t\t\t\t\t')
        #write coordinates
        np.savetxt(self.visfile, ticklinecoords,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
        self.visfile.write('"></Coordinate>\n')
        self.visfile.write(misc.tabs(5)+'</IndexedLineSet>\n')
        self.visfile.write(misc.tabs(4)+'</Shape>\n')
        self.visfile.write(misc.tabs(3)+'</Transform>\n')

    def make_animation(self, cycleinterval=10, axis=0):
        """
        Create an animation to rotate the X3D model along one axis.
        Must be outside the Transform "ROOT" element. Should be called after make_ticklines(). 
        """
        vec = np.zeros(3,dtype=int)
        vec[axis] = 1
        vec = str(vec)[1:-1]
        self.visfile.write(f'\n{misc.tabs(2)}<timeSensor id="time" cycleInterval="{cycleinterval}" loop="true" enabled="true" startTime="-1"></timeSensor>')
        self.visfile.write(f'\n{misc.tabs(2)}<OrientationInterpolator id="move" key="0 0.5 1" keyValue="{vec} 0 {vec} 3.14 {vec} 6.28"></OrientationInterpolator>')
        self.visfile.write('\n'+misc.tabs(2)+'<Route fromNode="time" fromField ="fraction_changed" toNode="move" toField="set_fraction"></Route>')
        self.visfile.write('\n'+misc.tabs(2)+'<Route fromNode="move" fromField ="value_changed" toNode="ROOT" toField="rotation"></Route>')


    def make_labels(self):
        """
        Create the labels of different elements in the figure.
        """
        self.visfile.write('\n\t\t<ProximitySensor id="PROX_LABEL" size="1.0e+06 1.0e+06 1.0e+06"></ProximitySensor>')
        self.visfile.write('\n\t\t<Collision enabled="false">')

        try:
            ramin1, ramax1 = (self.cube.coords[0]-np.mean(self.cube.coords[0])) \
                    * np.cos(self.cube.coords[1,0]*u.Unit(self.cube.units[2]).to('rad')) \
                    * u.Unit(self.cube.units[1])
            decmin1, decmax1 = (self.cube.coords[1]-np.mean(self.cube.coords[1])) \
                    * u.Unit(self.cube.units[2])
            vmin1, vmax1 = (self.cube.coords[2]-np.mean(self.cube.coords[2])) \
                    * u.Unit(self.cube.units[3])
            
            if ramin1.unit.is_equivalent(u.arcsec) and ramax1 < 1*u.deg:
                ramin1 = ramin1.to('arcsec')
                ramax1 = ramax1.to('arcsec')
            if decmin1.unit.is_equivalent(u.arcsec) and decmax1 < 1*u.deg:
                decmin1 = decmin1.to('arcsec')
                decmax1 = decmax1.to('arcsec')
            if vmin1.unit.is_equivalent(u.km/u.s):
                vmin1 = vmin1.to('km/s')
                vmax1 = vmax1.to('km/s')
            elif vmin1.unit.is_equivalent('m'):
                vmin1 = vmin1.to(u.Angstrom)
                vmax1 = vmax1.to(u.Angstrom)
            elif vmin1.unit.is_equivalent('m') and vmin1 < 1*u.mm:
                vmin1 = vmin1.to(u.um)
                vmax1 = vmax1.to(u.um)
            elif vmin1.unit.is_equivalent('Hz'):
                vmin1 = vmin1.to(u.GHz)
                vmax1 = vmax1.to(u.GHz)
        except Exception as ex:
            ramin1, ramax1 = (self.cube.coords[0]-np.mean(self.cube.coords[0]))
            decmin1, decmax1 = (self.cube.coords[1]-np.mean(self.cube.coords[1]))
            vmin1, vmax1 = (self.cube.coords[2]-np.mean(self.cube.coords[2]))

        try:
            ramin2, ramax2 = (self.cube.coords[0]) * u.Unit(self.cube.units[1])
            decmin2, decmax2 = (self.cube.coords[1]) * u.Unit(self.cube.units[2])
            vmin2, vmax2 = (self.cube.coords[2]) * u.Unit(self.cube.units[3])
            if vmin2.unit.is_equivalent(u.km/u.s):
                vmin2 = vmin2.to('km/s')
                vmax2 = vmax2.to('km/s')
            elif vmin2.unit.is_equivalent('m'):
                vmin2 = vmin2.to(u.Angstrom)
                vmax2 = vmax2.to(u.Angstrom)
            elif vmin2.unit.is_equivalent('Hz'):
                vmin2 = vmin2.to(u.GHz)
                vmax2 = vmax2.to(u.GHz)
        except:
            ramin2, ramax2 = self.cube.coords[0]
            decmin2, decmax2 = self.cube.coords[1]
            vmin2, vmax2 = self.cube.coords[2]

        # scale of labels
        labelscale = 20

        ax, axtick = misc.labpos

        #Names for the axes tick labels
        axticknames1 = np.array([f'{ramax1:.2f}',f'{ramin1:.2f}',f'{decmax1:.2f}',
                       f'{decmin1:.2f}',f'{vmin1:.2f}',f'{vmax1:.2f}',
                       f'{decmax1:.2f}',f'{decmin1:.2f}',f'{vmin1:.2f}',
                       f'{vmax1:.2f}',f'{ramax1:.2f}',f'{ramin1:.2f}'])

        axticknames2 = np.array([f'{ramax2:.3f}', f'{ramin2:.3f}', f'{decmax2:.3f}',
                       f'{decmin2:.3f}', f'{vmin2:.3f}', f'{vmax2:.3f}',
                       f'{decmax2:.3f}', f'{decmin2:.3f}', f'{vmin2:.3f}',
                       f'{vmax2:.3f}', f'{ramax2:.3f}', f'{ramin2:.3f}'])

        col = '0 0 0'

        axlabnames = misc.get_axlabnames(mags=self.cube.mags)

        #ax labels diff
        for i in range(6):
            self.visfile.write(f'\n\t\t\t\t<Transform id="alt_diff{i}" translation="{ax[i,0]} {ax[i,1]} {ax[i,2]}" rotation="{misc.axlabrot[i]}" scale="{labelscale} {labelscale} {labelscale}">')
            self.visfile.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
            self.visfile.write(f'\n\t\t\t\t\t\t\t<Material id="axlab_diff{i}" diffuseColor="0 0 0" emissiveColor="{col}"></Material>')
            self.visfile.write('\n\t\t\t\t\t\t</Appearance>')
            self.visfile.write(f"\n\t\t\t\t\t\t<Text string='{axlabnames[i]}'>")
            self.visfile.write(f'\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'{misc.axlabeljustify[i]}\' size="10"></FontStyle>')
            self.visfile.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')

        #ax tick labels diff
        for i in range(12):
            if i<4:
                rot = misc.axlabrot[0]
            elif i<8:
                rot = misc.axlabrot[2]
            else:
                rot = misc.axlabrot[4]
            self.visfile.write(f'\n\t\t\t\t<Transform id="att_diff{i}" translation="{axtick[i,0]} {axtick[i,1]} {axtick[i,2]}" rotation="{rot}" scale="{labelscale} {labelscale} {labelscale}">')
            self.visfile.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
            self.visfile.write(f'\n\t\t\t\t\t\t\t<Material id="axtick_diff{i}" diffuseColor="0 0 0" emissiveColor="{col}"></Material>')
            self.visfile.write('\n\t\t\t\t\t\t</Appearance>')
            self.visfile.write(f'\n\t\t\t\t\t\t<Text string="{axticknames1[i]}">')
            self.visfile.write(f'\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'{misc.axticklabjus[i]}\' size="8"></FontStyle>')
            self.visfile.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
        
        # don't show other labels if overlay
        if isinstance(self.cube.lines, dict) == False:
            #ax labels real
            for i in range(6):
                self.visfile.write(f'\n\t\t\t\t<Transform id="alt_real{i}" translation="{ax[i,0]} {ax[i,1]} {ax[i,2]}" rotation="{misc.axlabrot[i]}" scale="{labelscale} {labelscale} {labelscale}">')
                self.visfile.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.visfile.write(f'\n\t\t\t\t\t\t\t<Material id="axlab_real{i}" diffuseColor="0 0 0" emissiveColor="{col}" transparency="1"></Material>')
                self.visfile.write('\n\t\t\t\t\t\t</Appearance>')
                self.visfile.write(f"\n\t\t\t\t\t\t<Text string='{axlabnames[i]}'>")
                self.visfile.write(f'\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'{misc.axlabeljustify[i]}\' size="10"></FontStyle>')
                self.visfile.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
            #ax tick labels
            for i in range(12):
                if i<4:
                    rot = misc.axlabrot[0]
                elif i<8:
                    rot = misc.axlabrot[2]
                else:
                    rot = misc.axlabrot[4]
                self.visfile.write(f'\n\t\t\t\t<Transform id="att_real{i}" translation="{axtick[i,0]} {axtick[i,1]} {axtick[i,2]}" rotation="{rot}" scale="{labelscale} {labelscale} {labelscale}">')
                self.visfile.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.visfile.write(f'\n\t\t\t\t\t\t\t<Material id="axtick_real{i}" diffuseColor="0 0 0" emissiveColor="{col}" transparency="1"></Material>')
                self.visfile.write('\n\t\t\t\t\t\t</Appearance>')
                self.visfile.write(f'\n\t\t\t\t\t\t<Text string="{axticknames2[i]}">')
                self.visfile.write(f'\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'{misc.axticklabjus[i]}\' size="8"></FontStyle>')
                self.visfile.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')

        self.visfile.write('\n\t\t</Collision>')

    def make_markers(self, geom, points, shape, trans, color, labels=None):
        """
        Create markers in the X3D model.
        Must be called before make_ticklines().

        Parameters
        ----------
        geom: str
            Type of geometry to create. Options are 'tube', 'sphere', 'box', 'cone'.
        points : list
            List of points to create the markers. Each element of the list is a len 3 array with the coordinates of the marker (RA,DEC,Z).
            If geom is 'tube', each element of the list is a list of len 3 arrays that define the tubes. Two points define one tube.
        shape : list
            List of the shape of the markers.
            If geom is 'box', each element of the list is a len 3 array with the size of the box (x,y,z).
            If geom is 'cone', each element of the list is a len 2 array with the radius and height of the cone.
            Else each element of the list is a float with the radius of the marker.
        delta : list
            List with the delta of the cube (header[CDELT]).
        trans : list
            Transformation factor to the cube. It is calculated with (2000/nx, 2000/ny, 2000/nz).
        color : list
            List with the color of the markers. Each element is a string with RGB values.
        """
        delta = np.array(self.cube.delta)
        if geom == 'tube':
            n = 0
            for tube in points:
                # transform points to model coordinates
                tube = (tube - np.mean(self.cube.coords, axis=1))/delta*trans
                # get mean point between consecutive points
                pos = np.array([str(np.mean((tube[i],tube[i+1]), axis=0))[1:-1] for i in range(len(tube)-1)])
                # get distance between consecutive points
                diff = np.diff(tube, axis=0)
                heights = np.linalg.norm(diff,axis=1)
                #get rotation for each tube
                angles = np.arccos(diff[:,1]/heights)

                # create x3d object
                for i in range(len(tube)-1):
                    self.visfile.write("\n"+misc.tabs(3)+f'<Transform id="pytub{n}tra{i}" translation="{pos[i]}" rotation="{diff[i,2]:.4f} 0 {-diff[i,0]:.4f} {angles[i]:.4f}" scale="1 1 1">\n')
                    self.visfile.write(misc.tabs(4)+'<Shape ispickable="false">\n')
                    self.visfile.write(misc.tabs(5)+f'<Cylinder id="py{n}tub{i}" height="{heights[i]*1.015}" radius="{shape[n]}" solid="false"></Cylinder>\n')
                    self.visfile.write(misc.tabs(5)+'<Appearance>\n')
                    self.visfile.write(misc.tabs(6)+f'<Material id="py{n}tubmat{i}" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="{color[n]}" specularColor="0 0 0" shininess="0.0078" transparency="0"></Material>\n')
                    self.visfile.write(misc.tabs(5)+'</Appearance>\n')
                    self.visfile.write(misc.tabs(4)+'</Shape>\n')
                    self.visfile.write(misc.tabs(3)+'</Transform>\n')
                if labels is not None:
                    self.visfile.write(misc.tabs(4)+f'<Transform id="label_pytubtra{n}" translation="{pos[0]}" rotation="0 1 0 3.14" scale="20 20 20">\n') # labelscale = 20
                    self.visfile.write(misc.tabs(5)+'<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">\n')
                    self.visfile.write(misc.tabs(6)+'<Shape ispickable="false">\n')
                    self.visfile.write(misc.tabs(6)+'<Appearance>\n')
                    self.visfile.write(misc.tabs(6)+f'<Material id="label_pytub{n}" diffuseColor="0 0 0" emissiveColor="0 0 0"></Material>\n')
                    self.visfile.write(misc.tabs(6)+'</Appearance>\n')
                    self.visfile.write(misc.tabs(6)+f'<Text string="{labels[n]}">\n')
                    self.visfile.write(misc.tabs(7)+f'<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="5"></FontStyle>\n')
                    self.visfile.write(misc.tabs(6)+'</Text>\n')
                    self.visfile.write(misc.tabs(6)+'</Shape>\n')
                    self.visfile.write(misc.tabs(5)+'</Billboard>\n')
                    self.visfile.write(misc.tabs(4)+'</Transform>\n')

                n = n + 1

        if geom == 'sphere':
            n = 0
            for sphere in points:
                sphere = (sphere - np.mean(self.cube.coords, axis=1))/delta*trans
                sphere = str(sphere)[1:-1]
                self.visfile.write("\n"+misc.tabs(3)+f'<Transform id="pysph{n}tra" translation="{sphere}" scale="1 1 1">\n')
                self.visfile.write(misc.tabs(4)+'<Shape ispickable="false">\n')
                self.visfile.write(misc.tabs(5)+f'<Sphere id="py{n}sph" radius="{shape[n]}" solid="false"></Sphere>\n')
                self.visfile.write(misc.tabs(5)+'<Appearance>\n')
                self.visfile.write(misc.tabs(6)+f'<Material id="py{n}sphmat" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="{color[n]}" specularColor="0 0 0" shininess="0.0078" transparency="0"></Material>\n')
                self.visfile.write(misc.tabs(5)+'</Appearance>\n')
                self.visfile.write(misc.tabs(4)+'</Shape>\n')
                self.visfile.write(misc.tabs(3)+'</Transform>\n')

                if labels is not None:
                    self.visfile.write(misc.tabs(4)+f'<Transform id="label_pysphtra{n}" translation="{sphere}" rotation="0 1 0 3.14" scale="20 20 20">\n') # labelscale = 20
                    self.visfile.write(misc.tabs(5)+'<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">\n')
                    self.visfile.write(misc.tabs(6)+'<Shape ispickable="false">\n')
                    self.visfile.write(misc.tabs(6)+'<Appearance>\n')
                    self.visfile.write(misc.tabs(6)+f'<Material id="label_pysph{n}" diffuseColor="0 0 0" emissiveColor="0 0 0"></Material>\n')
                    self.visfile.write(misc.tabs(6)+'</Appearance>\n')
                    self.visfile.write(misc.tabs(6)+f'<Text string="{labels[n]}">\n')
                    self.visfile.write(misc.tabs(7)+f'<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="5"></FontStyle>\n')
                    self.visfile.write(misc.tabs(6)+'</Text>\n')
                    self.visfile.write(misc.tabs(6)+'</Shape>\n')
                    self.visfile.write(misc.tabs(5)+'</Billboard>\n')
                    self.visfile.write(misc.tabs(4)+'</Transform>\n')

                n = n + 1
        
        if geom == 'box':
            n = 0
            for box in points:
                box = (box - np.mean(self.cube.coords, axis=1))/delta*trans
                box = str(box)[1:-1]
                self.visfile.write("\n"+misc.tabs(3)+f'<Transform id="pybox{n}tra" translation="{box}" scale="1 1 1">\n')
                self.visfile.write(misc.tabs(4)+'<Shape ispickable="false">\n')
                self.visfile.write(misc.tabs(5)+f'<Box id="py{n}box" size="{shape[n][0]*2} {shape[n][1]*2} {shape[n][2]*2}" solid="false"></Box>\n')
                self.visfile.write(misc.tabs(5)+'<Appearance>\n')
                self.visfile.write(misc.tabs(6)+f'<Material id="py{n}boxmat" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="{color[n]}" specularColor="0 0 0" shininess="0.0078" transparency="0"></Material>\n')
                self.visfile.write(misc.tabs(5)+'</Appearance>\n')
                self.visfile.write(misc.tabs(4)+'</Shape>\n')
                self.visfile.write(misc.tabs(3)+'</Transform>\n')

                if labels is not None:
                    self.visfile.write(misc.tabs(4)+f'<Transform id="label_pyboxtra{n}" translation="{box}" rotation="0 1 0 3.14" scale="20 20 20">\n') # labelscale = 20
                    self.visfile.write(misc.tabs(5)+'<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">\n')
                    self.visfile.write(misc.tabs(6)+'<Shape ispickable="false">\n')
                    self.visfile.write(misc.tabs(6)+'<Appearance>\n')
                    self.visfile.write(misc.tabs(6)+f'<Material id="label_pybox{n}" diffuseColor="0 0 0" emissiveColor="0 0 0"></Material>\n')
                    self.visfile.write(misc.tabs(6)+'</Appearance>\n')
                    self.visfile.write(misc.tabs(6)+f'<Text string="{labels[n]}">\n')
                    self.visfile.write(misc.tabs(7)+f'<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="5"></FontStyle>\n')
                    self.visfile.write(misc.tabs(6)+'</Text>\n')
                    self.visfile.write(misc.tabs(6)+'</Shape>\n')
                    self.visfile.write(misc.tabs(5)+'</Billboard>\n')
                    self.visfile.write(misc.tabs(4)+'</Transform>\n')

                n = n + 1
            
        if geom == 'cone':
            n = 0
            for cone in points:
                cone = (cone - np.mean(self.cube.coords, axis=1))/delta*trans
                cone = str(cone)[1:-1]
                self.visfile.write("\n"+misc.tabs(3)+f'<Transform id="pycon{n}tra" translation="{cone}" scale="1 1 1">\n')
                self.visfile.write(misc.tabs(4)+'<Shape ispickable="false">\n')
                self.visfile.write(misc.tabs(5)+f'<Cone id="py{n}con" height="{shape[n][0]}" bottomRadius="{shape[n][1]}" solid="false"></Cone>\n')
                self.visfile.write(misc.tabs(5)+'<Appearance>\n')
                self.visfile.write(misc.tabs(6)+f'<Material id="py{n}conmat" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="{color[n]}" specularColor="0 0 0" shininess="0.0078" transparency="0"></Material>\n')
                self.visfile.write(misc.tabs(5)+'</Appearance>\n')
                self.visfile.write(misc.tabs(4)+'</Shape>\n')
                self.visfile.write(misc.tabs(3)+'</Transform>\n')

                if labels is not None:
                    self.visfile.write(misc.tabs(4)+f'<Transform id="label_pycontra{n}" translation="{cone}" rotation="0 1 0 3.14" scale="20 20 20">\n') # labelscale = 20
                    self.visfile.write(misc.tabs(5)+'<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">\n')
                    self.visfile.write(misc.tabs(6)+'<Shape ispickable="false">\n')
                    self.visfile.write(misc.tabs(6)+'<Appearance>\n')
                    self.visfile.write(misc.tabs(6)+f'<Material id="label_pycon{n}" diffuseColor="0 0 0" emissiveColor="0 0 0"></Material>\n')
                    self.visfile.write(misc.tabs(6)+'</Appearance>\n')
                    self.visfile.write(misc.tabs(6)+f'<Text string="{labels[n]}">\n')
                    self.visfile.write(misc.tabs(7)+f'<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="5"></FontStyle>\n')
                    self.visfile.write(misc.tabs(6)+'</Text>\n')
                    self.visfile.write(misc.tabs(6)+'</Shape>\n')
                    self.visfile.write(misc.tabs(5)+'</Billboard>\n')
                    self.visfile.write(misc.tabs(4)+'</Transform>\n')

                n = n + 1
                
    def close(self):
        """
        Closes the X3D file. Not using this function at the end results in
        an error.
        """
        #ending, close all
        self.visfile.write('\n\t\t</Transform>')
        self.visfile.write('\n\t</Scene>')
        self.visfile.write('\n</X3D>')
        self.visfile.close()