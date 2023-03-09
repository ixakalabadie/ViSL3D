# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:31:34 2023

@author: ixaka
"""

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
from astropy.coordinates import Angle
import matplotlib.pyplot as plt

from spectral_cube import SpectralCube
from astropy import wcs
from astropy.coordinates import SkyCoord

class write_x3d:
    """
    

    Parameters
    ----------
    filename : string, optinal
        Name of the file to be created. Should have the extension '.x3d'.
    delta : len 3 array
        Array with the step in each direction of the cube.
        Better if they are all around the same order and between 0.1~10.
    coords : astropy.Quantity 2d array
        Array with the minimum and maximum of the RA, DEC and VRAD
        in each row, in that order, of the cube. Must be an astropy quantity.

    Returns
    -------
    None.

    """
    
    def __init__(self, filename, delta, coords, units=['deg','deg','km s-1'], meta=None, picking=False):
        self.delta = delta
        #self.units = units
        self.real_coords, self.diff_coords = get_coords(coords[0], coords[1], coords[2])
        self.diff_coords[0] = self.diff_coords[0] * u.Unit(units[0]).to('arcsec')
        self.diff_coords[1] = self.diff_coords[1] * u.Unit(units[1]).to('arcsec')
        self.real_coords[2] = self.real_coords[2] * u.Unit(units[2]).to('km/s')
        self.diff_coords[2] = self.diff_coords[2] * u.Unit(units[2]).to('km/s')
        if picking:
            picking = 'true'
        else:
            picking = 'false'
        self.file_x3d = open(filename, 'w')
        self.file_x3d.write('<?xml version="1.0" encoding="UTF-8"?>\n <!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN" \n "http://www.web3d.org/specifications/x3d-3.3.dtd">')
        self.file_x3d.write('\n <X3D profile="Immersive" version="3.3" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.3.xsd">')
        self.file_x3d.write('\n <head>\n\t<meta name="file" content="%s"/>'%filename)
        # Additional metadata
        if meta != None:
            for met in meta.keys():
                self.file_x3d.write('\n\t<meta name="%s" content="%s"/>'%(met,meta[met]))
        self.file_x3d.write('\n </head>\n\t<Scene doPickPass="%s">\n\t\t<Background skyColor="0.9 0.9 0.9"/>'%picking)
        
        #make viewpoints
        # ramax, decmax = self.diff_coords[0][2], self.diff_coords[1][2]
        # vmin, vmean, vmax = self.diff_coords[2]
        # ma = np.max([ramax,decmax])
        # self.file_html.write("\t\t <OrthoViewpoint DEF='front' bind='false' centerOfRotation='0,0,%s' description='RA-Dec view' fieldOfView='[-%s,-%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='0,1,0,3.14' position='0,0,%s' zFar='10000' zNear='0.05' ></OrthoViewpoint>\n"%(vmean,ma*1.4,ma*1.4,ma*1.4,ma*1.4,1000))
        # ma = np.max([decmax,(vmax-vmin)/2])
        # self.file_html.write("\t\t <OrthoViewpoint DEF='side' bind='false' centerOfRotation='0,0,%s' description='V - Dec view' fieldOfView='[-%s,-%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='0,-1,0,1.57' position='-%s,0,%s' zFar='10000' zNear='0.05' ></OrthoViewpoint>\n"%(vmean,ma*1.4,ma*1.4,ma*1.4,ma*1.4,ma*1.4,vmean))
        # ma = np.max([ramax,(vmax-vmin)/2])
        # self.file_html.write("\t\t <OrthoViewpoint DEF='side2' bind='false' centerOfRotation='0,0,%s' description='V - RA view' fieldOfView='[-%s,-%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='1,1,1,4.1888' position='0,%s,%s' zFar='10000' zNear='0.05' ></OrthoViewpoint>\n"%(vmean,ma*1.4,ma*1.4,ma*1.4,ma*1.4,ma*1.4*4,vmean))
        
        self.file_x3d.write('\n\t\t<NavigationInfo type=\'"EXAMINE" "ANY"\' speed="4" headlight="true"/>')
        self.file_x3d.write('\n\t\t<DirectionalLight ambientIntensity="1" intensity="0" color="1 1 1"/>')
        self.file_x3d.write('\n\t\t<Transform DEF="ROOT" translation="0 0 0">')
        
    def make_layers(self, cube, isolevels, colors):
        #Change colors for default color using create_colormap()
        """
        Makes layers of the equal intensity and writes an x3d file

        Parameters
        ----------
        cube : 3d array
            The data cube.
        isolevels : list, array
            An array or list with the value of each isosurface layer. 
            E.g.[2,5,9] for three layers at values 2,5 and 9 of the cube.
            Should be in increasing order.
        colors : list of len 3 arrays
            List with RGB colors to be given to the layers, in the same order as isolevels.

        Returns
        -------
        None.

        """
        mins = (self.diff_coords[0][0], self.diff_coords[1][0], self.diff_coords[2][0])
        for i in range(len(isolevels)):
            verts, faces = marching_cubes(cube, level=isolevels[i], delta=self.delta, mins=mins)
            self.file_x3d.write('\n\t\t\t<Transform DEF="lt%s" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">'%i)
            self.file_x3d.write('\n\t\t\t\t<Shape DEF="layer%s_shape">'%i)
            self.file_x3d.write('\n\t\t\t\t\t<Appearance sortKey="%s">'%(len(isolevels)-1-i))
            #set color and transparency of layer
            if i == len(isolevels)-1:
                op = 0.001
            else:
                op = 0.8
            self.file_x3d.write('\n'+tabs(6)+'<Material DEF="layer%s" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="%s" specularColor="0 0 0" shininess="0.0078" transparency="%s"/>'%(i,colors[i],op))
            #correct color with depthmode (ALSO FOR LAST LAYER?)
            # if i != len(isolevels)-1:
            self.file_x3d.write('\n'+tabs(6)+'<DepthMode readOnly="true"></DepthMode>')
            self.file_x3d.write('\n'+tabs(5)+'</Appearance>')
            #define the layer object (normals set to false?)
            self.file_x3d.write('\n'+tabs(5)+'<IndexedFaceSet solid="false" colorPerVertex="false" normalPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
            #write indices
            np.savetxt(self.file_x3d, faces, fmt='%i', newline=' -1\n\t\t\t\t\t\t')
            self.file_x3d.write('">')
            self.file_x3d.write('\n\t\t\t\t\t\t<Coordinate DEF="Coordinates%s" point="\n\t\t\t\t\t\t'%i)
            #write coordinates
            np.savetxt(self.file_x3d, verts,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
            self.file_x3d.write('"/>')
            self.file_x3d.write('\n\t\t\t\t\t</IndexedFaceSet>\n\t\t\t\t</Shape>\n\t\t\t</Transform>')

    def make_outline(self):
        ramin, _, ramax = self.diff_coords[0]
        decmin, _, decmax = self.diff_coords[1]
        vmin, _, vmax = self.diff_coords[2]
        outlinecoords = np.array([[ramin,decmin,vmin],
                                  [ramax,decmin,vmin],
                                  [ramin,decmax,vmin],
                                  [ramax,decmax,vmin],
                                  [ramin,decmin,vmax],
                                  [ramax,decmin,vmax],
                                  [ramin,decmax,vmax],
                                  [ramax,decmax,vmax]])
        # Create outline
        self.file_x3d.write('\n\t\t\t<Transform DEF="ot" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">')
        self.file_x3d.write('\n\t\t\t\t<Shape ispickable="false">')
        self.file_x3d.write('\n\t\t\t\t\t<Appearance>')
        #define ouline ID
        self.file_x3d.write('\n\t\t\t\t\t\t<Material DEF="outline" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="0 0 0" specularColor="0 0 0" shininess="0.0078" transparency="0"/>')
        self.file_x3d.write('\n\t\t\t\t\t</Appearance>')
        self.file_x3d.write('\n\t\t\t\t\t<IndexedLineSet colorPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
        #write indices
        np.savetxt(self.file_x3d, outlineindex, fmt='%i', newline='\n\t\t\t\t\t\t')
        self.file_x3d.write('">')
        self.file_x3d.write('\n\t\t\t\t\t\t<Coordinate DEF="OutlineCoords" point="\n\t\t\t\t\t\t')
        #write coordinates
        np.savetxt(self.file_x3d, outlinecoords,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
        self.file_x3d.write('"/>')
        self.file_x3d.write('\n\t\t\t\t\t</IndexedLineSet>\n\t\t\t\t</Shape>\n\t\t\t</Transform>')
        
    def make_galaxies(self, gals, labels=True):
        """
        

        Parameters
        ----------
        gals : TYPE
            DESCRIPTION.
        labels : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        ramin1, _, ramax1 = self.diff_coords[0]
        decmin1, _, decmax1 = self.diff_coords[1]
        vmin1, _, vmax1 = self.diff_coords[2]
        m = np.min([ramax1-ramin1, decmax1-decmin1, vmax1-vmin1])
        
        sphereradius = m/45 #min(self.delta)*8
        crosslen = m/20 #min(self.delta)*20
        #create galaxy crosses and spheres
        for i, gal in enumerate(gals.keys()):
            #galaxy crosses
            self.file_x3d.write(tabs(3)+'<Transform DEF="gct%s" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n'%i)
            self.file_x3d.write(tabs(4)+'<Shape ispickable="false">\n')
            self.file_x3d.write(tabs(5)+'<Appearance>\n')
            self.file_x3d.write(tabs(6)+'<Material DEF="%s" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="0 0 0" specularColor="0 0 0" shininess="0.0078" transparency="0"/>\n'%(gal+'_cross'))
            self.file_x3d.write(tabs(5)+'</Appearance>\n')
            #cross indices
            self.file_x3d.write(tabs(5)+'<IndexedLineSet colorPerVertex="true" coordIndex="\n\t\t\t\t\t\t0 1 -1\n\t\t\t\t\t\t2 3 -1\n\t\t\t\t\t\t4 5 -1\n\t\t\t\t\t\t">\n')
            self.file_x3d.write(tabs(5)+'<Coordinate DEF="CrossCoords%s" point="\n\t\t\t\t\t\t'%i)
            vec = gals[gal]['coord']
            vec[2] = vec[2]-self.real_coords[2][1]
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
            self.file_x3d.write(tabs(3)+'<Transform DEF="gst%s" translation="%s %s %s">\n'%(i,vec[0],vec[1],vec[2]))
            self.file_x3d.write(tabs(4)+'<Shape ispickable="false">\n')
            self.file_x3d.write(tabs(5)+'<Sphere radius="%s" solid="false"/>\n'%sphereradius)
            self.file_x3d.write(tabs(5)+'<Appearance>\n')
            self.file_x3d.write(tabs(6)+'<Material DEF="%s" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="%s" specularColor="0 0 0" shininess="0.0078" transparency="0"/>\n'%(gal,gals[gal]['col']))
            self.file_x3d.write(tabs(5)+'</Appearance>\n\t\t\t\t</Shape>\n\t\t\t</Transform>\n')
            
    def make_image2d(self, imcol, img_shape):
        """
        Create a 2d image in the X3D figure.

        Parameters
        ----------
        imcol : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        ramin, _, ramax = self.diff_coords[0]
        decmin, _, decmax = self.diff_coords[1]
        _, _, vmax = self.diff_coords[2]
        
        
        # coordinates of 2d image
        coords2d = np.array([[ramax,decmin,vmax],
                             [ramax,decmax,vmax],
                             [ramin,decmin,vmax],
                             [ramin,decmax,vmax]])
        
        self.file_x3d.write(tabs(3)+'<Transform DEF="image2d" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n')
        self.file_x3d.write(tabs(4)+'<Shape ispickable="False">\n')
        self.file_x3d.write(tabs(5)+'<Appearance>\n')
        self.file_x3d.write(tabs(6)+'<Material ambientIntensity="1" emissiveColor="0 0 0" diffuseColor="0 0 0" specularColor="0 0 0" shininess="0.0078" transparency="0"/>\n')
        self.file_x3d.write(tabs(6)+'<PixelTexture repeatS="false" repeatT="false" image=" %s %s 3 \n'%(img_shape[0], img_shape[1]))
        # write pixel colors
        np.savetxt(self.file_x3d, imcol, fmt='%s', delimiter=' ', newline='\n')
        self.file_x3d.write('"/>\n'+tabs(5)+'</Appearance>\n')
        #SOLID=TRUE makes it transparent from one side
        self.file_x3d.write(tabs(5)+'<IndexedFaceSet solid="false" colorPerVertex="false" normalPerVertex="false" coordIndex="2 3 1 0 -1">\n')
        self.file_x3d.write(tabs(6)+'<Coordinate DEF="imgCoords" point="\n\t\t\t\t\t\t')
        # write coordinates
        np.savetxt(self.file_x3d, coords2d, fmt='%.3f', newline='\n\t\t\t\t\t\t')
        self.file_x3d.write('"/>\n')
        self.file_x3d.write(tabs(6)+'<TextureCoordinate DEF="imgTexCoords" point="\n\t\t\t\t\t\t 0 0, 1 0, 0 1, 1 1"/>\n')
        self.file_x3d.write(tabs(5)+'</IndexedFaceSet>\n')
        self.file_x3d.write(tabs(4)+'</Shape>\n')
        self.file_x3d.write(tabs(3)+'</Transform>\n')
            
    def make_ticklines(self):
        ramin, ramean, ramax = self.diff_coords[0]
        decmin, decmean, decmax = self.diff_coords[1]
        vmin, vmean, vmax = self.diff_coords[2]
        # coordinates of tick lines
        ticklinecoords = np.array([[ramin,decmean,vmin],
                                   [ramax,decmean,vmin],
                                   [ramean,decmin,vmin],
                                   [ramean,decmax,vmin],
                                   [ramin,decmean,vmin],
                                   [ramin,decmean,vmax],
                                   [ramin,decmin,vmean],
                                   [ramin,decmax,vmean],
                                   [ramean,decmax,vmin],
                                   [ramean,decmax,vmax],
                                   [ramin,decmax,vmean],
                                   [ramax,decmax,vmean]])
        #Create ticklines
        self.file_x3d.write('\n\t\t\t<Transform DEF="tlt" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">')
        self.file_x3d.write('\n\t\t\t\t<Shape ispickable="false">')
        self.file_x3d.write('\n\t\t\t\t\t<Appearance>')
        #set color and transparency of layer
        self.file_x3d.write('\n\t\t\t\t\t\t<Material DEF="ticklines" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="0 0 0" specularColor="0 0 0" shininess="0.0078" transparency="0"/>')
        self.file_x3d.write('\n\t\t\t\t\t</Appearance>')
        self.file_x3d.write('\n\t\t\t\t\t<IndexedLineSet colorPerVertex="false" coordIndex="\n\t\t\t\t\t\t')
        #write indices
        np.savetxt(self.file_x3d, ticklineindex, fmt='%i', newline='\n\t\t\t\t\t\t')
        self.file_x3d.write('">')
        self.file_x3d.write('\n\t\t\t\t\t\t<Coordinate DEF="ticklineCoords" point="\n\t\t\t\t\t\t')
        #write coordinates
        np.savetxt(self.file_x3d, ticklinecoords,fmt='%.3f', newline=',\n\t\t\t\t\t\t')
        self.file_x3d.write('"/>')
        self.file_x3d.write('\n\t\t\t\t\t</IndexedLineSet>\n\t\t\t\t</Shape>\n\t\t\t</Transform>')
        self.file_x3d.write('\n\t\t</Transform>')
        
    def make_labels(self, gals=None, axlab=None):
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
            A string indicating what ax labels to include. Can be 'real' for the actual
            coordinates; 'diff' for the difference from the center of the cube or
            'both' for both options, in this case ???. Leave None for no ax labels.
            The default is None.

        Returns
        -------
        None.

        """
        self.file_x3d.write('\n\t\t<ProximitySensor DEF="PROX_LABEL" size="1.0e+06 1.0e+06 1.0e+06"/>')
        self.file_x3d.write('\n\t\t<Collision enabled="false">')
        self.file_x3d.write('\n\t\t\t<Transform DEF="TRANS_LABEL">')
        
        ramin1, ramean1, ramax1 = self.diff_coords[0]
        decmin1, decmean1, decmax1 = self.diff_coords[1]
        vmin1, vmean1, vmax1 = self.diff_coords[2]
        ramin2, _, ramax2 = Angle(self.real_coords[0] * u.Unit('deg')).to_string(u.hour, precision=0)
        decmin2, _, decmax2 = Angle(self.real_coords[1] * u.Unit('deg')).to_string(u.degree, precision=0)
        vmin2, _, vmax2 = self.real_coords[2]
        
        # scale of labels
        shape = np.min([ramax1-ramin1, decmax1-decmin1, vmax1-vmin1])
        labelscale = calc_scale(shape)
        
        ax,axtick = labpos(self.diff_coords)
        
        #Names for the axes tick labels
        axticknames1 = np.array([f'{ramax1:.0f}',f'{ramin1:.0f}',f'{decmax1:.0f}',
                       f'{decmin1:.0f}',f'{vmin1:.0f}',f'{vmax1:.0f}',
                       f'{decmax1:.0f}',f'{decmin1:.0f}',f'{vmin1:.0f}',
                       f'{vmax1:.0f}',f'{ramax1:.0f}',f'{ramin1:.0f}'])
        
        axticknames2 = np.array([ramax2, ramin2, decmax2,
                       decmin2, f'{vmin2:.0f}', f'{vmax2:.0f}',
                       decmax2, decmin2, f'{vmin2:.0f}',
                       f'{vmax2:.0f}', ramax2, ramin2])
        
        #galaxy labels
        if gals:
            for (i,gal) in enumerate(gals.keys()):
                self.file_x3d.write('\n\t\t\t\t<Transform DEF="glt%s" translation="%s %s %s" rotation="0 1 0 3.14" scale="%s %s %s">'%(i,gals[gal]['coord'][0],gals[gal]['coord'][1], gals[gal]['coord'][2], labelscale, labelscale, labelscale))
                self.file_x3d.write('\n\t\t\t\t\t<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">')
                self.file_x3d.write('\n\t\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t<Material DEF="%s" diffuseColor="0 0 0" emissiveColor="0 0 0"/>'%('label_'+gal))
                self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t<Text string="%s">'%gal)
                self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="10"/>')
                self.file_x3d.write('\n\t\t\t\t\t\t</Text> \n\t\t\t\t\t\t</Shape>\n\t\t\t\t\t</Billboard>\n\t\t\t\t</Transform>')
        
        
        #CHANGE this, all in same for
        if axlab == 'diff' or axlab == 'both':
            #ax labels
            for i in range(6):
                self.file_x3d.write('\n\t\t\t\t<Transform DEF="alt_diff%s" translation="%s %s %s" rotation="%s" scale="%s %s %s">'%(i,ax[i,0],ax[i,1],ax[i,2],axlabrot[i], labelscale, labelscale, labelscale))
                self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axlab_diff%s" diffuseColor="0 0 0" emissiveColor="0 0 0"/>'%i)
                self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
                self.file_x3d.write("\n\t\t\t\t\t\t<Text string='%s'>"%axlabname1[i])
                self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'%s\' size="14"/>'%axlabeljustify[i])
                self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
            #ax tick labels
            for i in range(12):
                if i<4: rot = axlabrot[0]
                elif i<8: rot = axlabrot[2]
                else: rot = axlabrot[4]
                self.file_x3d.write('\n\t\t\t\t<Transform DEF="att_diff%s" translation="%s %s %s" rotation="%s" scale="%s %s %s">'%(i,axtick[i,0],axtick[i,1],axtick[i,2], rot, labelscale, labelscale, labelscale))
                self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axtick_diff%s" diffuseColor="0 0 0" emissiveColor="0 0 0"/>'%i)
                self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t<Text string="%s">'%axticknames1[i])
                self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'%s\' size="12"/>'%axticklabjus[i])
                self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
        
        if axlab == 'real' or axlab == 'both':
            if axlab == 'real': trans = '0'
            if axlab == 'both': trans = '1'
            #ax labels
            for i in range(6):
                self.file_x3d.write('\n\t\t\t\t<Transform DEF="alt_real%s" translation="%s %s %s" rotation="%s" scale="%s %s %s">'%(i,ax[i,0],ax[i,1],ax[i,2],axlabrot[i], labelscale, labelscale, labelscale))
                self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axlab_real%s" diffuseColor="0 0 0" emissiveColor="0 0 0" transparency="%s"/>'%(i,trans))
                self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
                self.file_x3d.write("\n\t\t\t\t\t\t<Text string='%s'>"%axlabname2[i])
                self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'%s\' size="14"/>'%axlabeljustify[i])
                self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
            #ax tick labels
            for i in range(12):
                if i<4: rot = axlabrot[0]
                elif i<8: rot = axlabrot[2]
                else: rot = axlabrot[4]
                self.file_x3d.write('\n\t\t\t\t<Transform DEF="att_real%s" translation="%s %s %s" rotation="%s" scale="%s %s %s">'%(i,axtick[i,0],axtick[i,1],axtick[i,2], rot, labelscale, labelscale, labelscale))
                self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axtick_real%s" diffuseColor="0 0 0" emissiveColor="0 0 0" transparency="%s"/>'%(i, trans))
                self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t<Text string="%s">'%axticknames2[i])
                self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'%s\' size="12"/>'%axticklabjus[i])
                self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
                
    def close(self):
        """
        Closes the X3D file. Not using this function at the end results in
        an error.

        Returns
        -------
        None.

        """
        #ending, close all
        self.file_x3d.write('\n\t\t\t</Transform>')
        self.file_x3d.write('\n\t\t</Collision>')
        self.file_x3d.write('\n\t\t<ROUTE fromNode="PROX_LABEL" fromField="position_changed" toNode="TRANS_LABEL" toField="set_translation"/>')
        self.file_x3d.write('\n\t\t<ROUTE fromNode="PROX_LABEL" fromField="orientation_changed" toNode="TRANS_LABEL" toField="set_rotation"/>')
        self.file_x3d.write('\n\t</Scene>')
        self.file_x3d.write('\n</X3D>')
        self.file_x3d.close()

class write_html:
    """
    Create an HTML file with an embbeded X3D figure and buttons to interact with it

    Parameters
    ----------
    filename : string
        Name of the file to be created, should have ".html" extension.
    tabtitle : string, optional
        Title of the tab in the web browser. The default is 'new_html_x3d'.
    pagetitle : string, optional
        Title of the web page. The default is None.
    description : string, optional
        Description of the figure or any other text to be included in the web page.
        Should follow HTML format. The default is None.

    Returns
    -------
    None.

    """
    
    def __init__(self, filename, tabtitle='new_html_x3d', pagetitle=None, description=None):
        #some attributes to use later
        self.grids = False
        self.gals = False
        self.gallabs = False
        self.axes = False
        self.hclick = False
        self.viewp = False
            
        self.file_html = open(filename, 'w')
        self.file_html.write('<html>\n\t <head>\n')
        self.file_html.write('\t\t <title> %s </title>\n'%tabtitle)
        self.file_html.write("\t\t <script type='text/javascript' src='http://www.x3dom.org/download/1.8.0/x3dom.js'></script>\n")
        self.file_html.write("\n\t\t <script type='text/javascript'  src='https://www.maths.nottingham.ac.uk/plp/pmadw/LaTeXMathML.js'></script>\n")
        self.file_html.write("\t\t <link rel='stylesheet' type='text/css' href='http://www.x3dom.org/release/x3dom.css'></link>\n")
        self.file_html.write("\t\t <script type='text/javascript' src='https://code.jquery.com/jquery-3.6.3.min.js'></script>\n")
        self.file_html.write("\n\t \t <style>\n\t \t   x3d\n\t \t \t   {\n\t \t \t \t   border:2px solid darkorange;\n\t \t    }\n\t \t    </style>\n\t </head>\n\t <body>\n")
        
        
        if pagetitle != None:
            self.file_html.write('\t <h1 align="middle"> %s </h1>\n'%pagetitle)
            self.file_html.write('\t <hr/>\n')
        if description != None:
            self.file_html.write('\t <p style="width=1000px;"> <font size="3">\n\t %s \n'%description)
        
        self.file_html.write('\t <br/>\n\t </font> </p>\n\n')
        
    def func_layers(self, nlayers):
        """
        Make the funcion to hide/show layers. If this is used, the buttons()
        function with isolevels should also be used to create the buttons.
        The X3D file must have layers for this to work.

        Parameters
        ----------
        nlayers : int
            Number of layers in the X3D file.

        Returns
        -------
        None.

        """
        self.nlayers = nlayers
        for i in range(nlayers):
            if i != nlayers-1:
                self.file_html.write("\t <script>\n\t \t function setHIlayer%s()\n\t \t {\n\t \t if(document.getElementById('cube__layer%s').getAttribute('transparency') != '0.8') {\n"%(i,i))
                self.file_html.write("\t\t document.getElementById('cube__layer%s').setAttribute('transparency', '0.8');\n"%i)
                self.file_html.write("\t\t document.getElementById('cube__layer%s_shape').setAttribute('ispickable', 'true');\n"%i)
                self.file_html.write("\t\t } else { \n\t\t document.getElementById('cube__layer%s').setAttribute('transparency', '1');\n"%i)
                self.file_html.write("\t\t document.getElementById('cube__layer%s_shape').setAttribute('ispickable', 'false');\n"%i)
                self.file_html.write("\t\t } \n\t\t }\n\t </script>\n")
            else:
                self.file_html.write("\t <script>\n\t\t function setHIlayer%s()\n\t \t {\n\t \t if(document.getElementById('cube__layer%s').getAttribute('transparency') != '0.001') {\n"%(i,i))
                self.file_html.write("\t\t document.getElementById('cube__layer%s').setAttribute('transparency', '0.001');\n"%i)
                self.file_html.write("\t\t document.getElementById('cube__layer%s_shape').setAttribute('ispickable', 'true');\n"%i)
                self.file_html.write("\t\t } else { \n\t\t document.getElementById('cube__layer%s').setAttribute('transparency', '1');\n"%i)
                self.file_html.write("\t\t document.getElementById('cube__layer%s_shape').setAttribute('ispickable', 'false');\n"%i)
                self.file_html.write("\t\t } \n\t\t }\n\t </script>\n")

    def func_galaxies(self, gals):
        """
        Make function to hide/show galaxies.
        The X3D file must have galaxies for this to work.

        Parameters
        ----------
        gals : dictionary
            Same dictionary as the one used to create galaxies in the X3D file.

        Returns
        -------
        None.

        """
        self.gals = gals
        for i,gal in enumerate(gals):
            if i == 0:
                self.file_html.write("\t \t <script>\n\t \t function setgals()\n\t \t {\n\t \t if(document.getElementById('cube__%s_cross').getAttribute('transparency')!= '0'){\n"%gal)
            self.file_html.write("\t \t document.getElementById('cube__%s_cross').setAttribute('transparency', '0');\n"%gal)
            self.file_html.write("\t \t document.getElementById('cube__%s').setAttribute('transparency', '0');\n"%gal)
        self.file_html.write("\t \t }\n\t \t else {\n")
        for i,gal in enumerate(gals):
            self.file_html.write("\t \t document.getElementById('cube__%s_cross').setAttribute('transparency', '1');\n"%gal)
            self.file_html.write("\t \t document.getElementById('cube__%s').setAttribute('transparency', '1');\n"%gal)
        self.file_html.write("\t\t }\n\t\t }\n\t\t </script>\n")
    
    def func_gallab(self):
        """
        Make function to hide/show galaxy labels. If this is used, the buttons()
        function with gallabs=True should also be used to create the buttons.
        The X3D file must have galaxy labels for this to work.

        Parameters
        ----------
        gals : dictionary
            Same dictionary as the one used to create galaxies in the X3D file.

        Returns
        -------
        None.

        """
        self.gallabs = True # to use in buttons()
        for i,gal in enumerate(self.gals):
            if i == 0:
                self.file_html.write("\t\t <script>\n\t \t function setgallabels()\n\t \t {\n\t \t if(document.getElementById('cube__label_%s').getAttribute('transparency')!= '0'){\n"%gal)
            self.file_html.write("\t \t document.getElementById('cube__label_%s').setAttribute('transparency', '0');\n"%gal)
        self.file_html.write("\t \t }\n\t \t else {\n")
        for i,gal in enumerate(self.gals):
            self.file_html.write("\t \t document.getElementById('cube__label_%s').setAttribute('transparency', '1');\n"%gal)
        self.file_html.write("\t\t }\n\t\t }\n\t\t </script>\n")
        
    def func_grids(self):
        self.grids = True # to use in buttons()
        self.file_html.write(tabs(2)+"<script>\n\t\tfunction setgrids()\n\t\t{\n")
        self.file_html.write(tabs(3)+"if(document.getElementById('cube__ticklines').getAttribute('transparency')!= '0')\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__ticklines').setAttribute('transparency', '0');\n")
        self.file_html.write(tabs(3)+"else \n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__ticklines').setAttribute('transparency', '1');\n")
        self.file_html.write(tabs(2)+"}\n\t\t </script>\n")
            
    def func_axes(self, axes):
        self.axes = True
        if axes == 'diff' or axes == 'real':
            self.file_html.write("\t \t <script>\n\t\t function setaxes()\n\t \t {\n\t \t if(document.getElementById('cube__axlab_diff1').getAttribute('transparency')!= '0') {\n")
            for i in range(6):
                self.file_html.write("\t\t document.getElementById('cube__axlab_%s%s').setAttribute('transparency', '0');\n"%(axes,i))
            for i in range(12):
                self.file_html.write("\t\t document.getElementById('cube__axtick_%s%s').setAttribute('transparency', '0');\n"%(axes,i))
            self.file_html.write("\t \t }\n\t \t else {\n")
            for i in range(6):
                self.file_html.write("\t\t document.getElementById('cube__axlab_%s%s').setAttribute('transparency', '1');\n"%(axes,i))
            for i in range(12):
                self.file_html.write("\t\t document.getElementById('cube__axtick_%s%s').setAttribute('transparency', '1');\n"%(axes,i))
                
        if axes == 'both':
            self.file_html.write("\t \t <script>\n\t\t function setaxes()\n\t \t {\n\t \t if(document.getElementById('cube__axlab_diff1').getAttribute('transparency') == '0') {\n")
            for i in range(6):
                self.file_html.write("\t\t document.getElementById('cube__axlab_diff%s').setAttribute('transparency', '1');\n"%(i))
                self.file_html.write("\t\t document.getElementById('cube__axlab_real%s').setAttribute('transparency', '0');\n"%(i))
            for i in range(12):
                self.file_html.write("\t\t document.getElementById('cube__axtick_diff%s').setAttribute('transparency', '1');\n"%(i))
                self.file_html.write("\t\t document.getElementById('cube__axtick_real%s').setAttribute('transparency', '0');\n"%(i))
                
            self.file_html.write("\t \t }\n\t \t else if (document.getElementById('cube__axlab_real1').getAttribute('transparency') == '0') {\n")
            for i in range(6):
                self.file_html.write("\t\t document.getElementById('cube__axlab_real%s').setAttribute('transparency', '1');\n"%(i))
            for i in range(12):
                self.file_html.write("\t\t document.getElementById('cube__axtick_real%s').setAttribute('transparency', '1');\n"%(i))
                
            self.file_html.write("\t\t }\n\t\t else {\n")
            for i in range(6):
                self.file_html.write("\t\t document.getElementById('cube__axlab_diff%s').setAttribute('transparency', '0');\n"%(i))
            for i in range(12):
                self.file_html.write("\t\t document.getElementById('cube__axtick_diff%s').setAttribute('transparency', '0');\n"%(i))
            
        self.file_html.write("\t\t }\n\t\t }\n\t\t </script>\n")
        
    def func_pick(self):
        """
        Allows picking the coordinates by clicking in the figure.
        WORKS WITH VIEWPOINT NOT WITH ORTHOVIEWPOINT

        Returns
        -------
        None.

        """
        self.hclick = True
        self.file_html.write(roundTo) #premade string with function to round to two decimals
        self.file_html.write(tabs(1)+"<script>\n")
        self.file_html.write(tabs(3)+"const picksca = document.querySelector('#scalev');\n")
        self.file_html.write(tabs(2)+"function handleClick(event) {\n")
        self.file_html.write(tabs(3)+"const sca = picksca.value;\n")
        self.file_html.write(tabs(3)+"var coordinates = event.hitPnt;\n")
        self.file_html.write(tabs(3)+"$('#coordX').html(roundTo(coordinates[0], 2)+' arcsec');\n")
        self.file_html.write(tabs(3)+"$('#coordY').html(roundTo(coordinates[1], 2)+' arcsec');\n")
        self.file_html.write(tabs(3)+"$('#coordZ').html(roundTo(coordinates[2], 2)/sca+' km/s');\n")
        self.file_html.write(tabs(2)+"}\n\t </script>\n")
        
    def start_x3d(self):
        """
        Start the X3D part of the HTML. Must go before close_x3d().

        """
        self.file_html.write(tabs(1)+"<center><x3d id='cubeFixed' width='100\%' height='500px'>\n")

    def close_x3d(self, x3dname):
        """
        Insert the X3D file and close the X3D part of the HTML. Must go after start_x3d()

        Parameters
        ----------
        x3dname : string
            Name of the X3D file to be inserted.

        """
        if self.hclick:
            self.file_html.write(tabs(3)+'<inline url="%s" nameSpaceName="cube" mapDEFToID="true" onclick="handleClick(event)"/>\n'%x3dname)
        else:
            self.file_html.write(tabs(3)+'<inline url="%s" nameSpaceName="cube" mapDEFToID="true" onclick=""/>\n'%x3dname)
        self.file_html.write(tabs(2)+"</scene>\n\t</x3d></center>\n")
        
    def viewpoints(self, maxco, vrad):
        """
        Define viewpoints for the X3D figure. Must go after start_x3d() and
        before close_x3d(). Mandatory if the X3D does not have a Viewpoint,
        which is the case with the one created with this module.

        Parameters
        ----------
        maxco : len 2 array
            Maximum values of RA and DEC, in that order, for the difference to the center
        vrad : len 3 array
            Minimum, mean and maximum of radio velocity, in that order, for the difference from the center

        Returns
        -------
        None.

        """
        self.viewp = True # to use in buttons()
        ramax, decmax = maxco
        vmin, vmean, vmax = vrad
        self.file_html.write("\t\t <scene>\n")
        ma = np.max([ramax,decmax])
        self.file_html.write("\t\t\t <OrthoViewpoint id=\"front\" bind='false' centerOfRotation='0,0,0' description='RA-Dec view' fieldOfView='[-%s,-%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='0,1,0,3.14' position='0,0,-%s' zFar='10000' zNear='0.05' ></OrthoViewpoint>\n"%(ma*1.4,ma*1.4,ma*1.4,ma*1.4,ma*4))
        ma = np.max([decmax,(vmax-vmin)/2])
        self.file_html.write("\t\t\t <OrthoViewpoint id=\"side\" bind='false' centerOfRotation='0,0,0' description='V - Dec view' fieldOfView='[-%s,-%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='0,-1,0,1.57' position='-%s,0,0' zFar='10000' zNear='0.05' ></OrthoViewpoint>\n"%(ma*1.4,ma*1.4,ma*1.4,ma*1.4,ma*4))
        ma = np.max([ramax,(vmax-vmin)/2])
        self.file_html.write("\t\t\t <OrthoViewpoint id=\"side2\" bind='false' centerOfRotation='0,0,0' description='V - RA view' fieldOfView='[-%s,-%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='1,1,1,4.1888' position='0,%s,0' zFar='10000' zNear='0.05' ></OrthoViewpoint>\n"%(ma*1.4,ma*1.4,ma*1.4,ma*1.4,ma*4))

    def buttons(self, isolevels=None, colormaps=None, hide2d=False, scalev=False, move2d=False):
        """
        Makes the buttons to apply the functions to hide/show elements, if
        funcitons such as func_layers() or func_galaxies() have been created.

        Parameters
        ----------
        isolevels : array, optional
            Array with the values of each isosurface in mJy/beam, to create
            buttons for layers. If None, a default name will be given to the
            buttons. The default is None.

        Returns
        -------
        None.

        """
        self.scalev=scalev
        self.move2d = move2d
        self.file_html.write(tabs(1)+'<div style="width:90%">\n')
        self.file_html.write(tabs(2)+'<br/>\n')
        # Viewpoint buttons
        if self.viewp:
            self.file_html.write(tabs(2)+"&nbsp <b>Viewpoints:</b>\n")
            self.file_html.write(tabs(3)+"<button onclick=\"document.getElementById('cubeFixed').runtime.resetView();\">Reset View</button>\n")
            for i in range(3): # 3 w/o perspective, 4 with
                self.file_html.write(tabs(3)+"<button onclick=\"document.getElementById('%s').setAttribute('set_bind','true');\"> %s </button>\n"%(side[i], nam[i]))
            #unccoment next line for next view button
            #self.file_html.write('\t\t   <button onclick="document.getElementById(\'cubeFixed\').runtime.nextView();">Next View</button>\n')
                
        if self.grids or self.gals or self.gallabs or self.axes or hide2d:
            self.file_html.write('\n'+tabs(2)+'&nbsp <b>Labels:</b>\n')
        if self.grids:
            self.file_html.write(tabs(3)+'<button onclick="setgrids();" >Grids</button>\n')
        if self.gals:
            self.file_html.write(tabs(3)+'<button onclick="setgals();" >Galaxies</button>\n')
        if self.gallabs:
            self.file_html.write(tabs(3)+'<button onclick="setgallabels();" >Galaxy Labels</button>\n')
        if self.axes:
            self.file_html.write(tabs(3)+'<button onclick="setaxes();" >Axes labels</button>\n')
        if hide2d:
            self.file_html.write(tabs(3)+'<button onclick="setimage2d();" >2D image</button>\n')
                
        if isolevels is not None:
            self.file_html.write(tabs(2)+'<br><br>\n')
            self.file_html.write(tabs(2)+'&nbsp <b>HI layers:</b>\n')
            for i in range(len(isolevels)):
                self.file_html.write(tabs(3)+'<button onclick="setHIlayer%s();" style="color:FireBrick">%s mJy/beam</button>\n'%(i,np.round(isolevels[i],1)))
        elif isolevels is None and self.nlayers:
            self.file_html.write(tabs(2)+'<br><br>\n')
            self.file_html.write(tabs(2)+' &nbsp <b>HI layers:</b>\n')
            for i in range(self.nlayers):
                self.file_html.write(tabs(3)+'<button onclick="setHIlayer%s();" style="color:FireBrick">Layer %s</button>\n'%(i,i))
                    
        # to separate buttons in two parts
        #if self.grids or self.gals or self.gallabs or self.axes or self.hclick or colormaps is not None:
            #self.file_html.write('\n\t <div style="position:absolute;left:800px;top:140px;width:600px">\n')
            
        if colormaps is not None:
            self.colormaps = colormaps
            #self.file_html.write('\t\t <br><br>\n')
            self.file_html.write(tabs(2)+'&nbsp <label for="cmaps-choice"><b>Colormaps:</b> </label>\n')
            self.file_html.write(tabs(2)+'<select id="cmaps-choice">\n')
            for c in self.colormaps:
                self.file_html.write(tabs(3)+'<option value="%s">%s</option>\n'%(c,c))
            self.file_html.write(tabs(2)+'</select>\n')     
            
        if scalev:
            #self.file_html.write('\t\t <br><br>\n')
            self.file_html.write(tabs(2)+'&nbsp <label for="scalev"><b>V scale:</b> </label>\n')
            self.file_html.write(tabs(2)+'<input oninput="changescalev()" id="scalev" type="range" list="marker" min="0" max="5" step="0.001" value="1"/>\n')
            self.file_html.write(tabs(2)+'<datalist id="marker">\n')
            self.file_html.write(tabs(3)+'<option value="1"></option>\n')
            self.file_html.write(tabs(2)+'</datalist>\n')
            
        if move2d:
            #self.file_html.write('\t\t <br><br>\n')
            self.file_html.write(tabs(2)+'&nbsp <label for="move2dimg"><b>2D image:</b> </label>\n')
            self.file_html.write(tabs(2)+'<input oninput="move2d()" id="move2dimg" type="range" min="-1" max="1" step="0.0001"/>\n')
            self.file_html.write(tabs(2)+'<b>$\Delta V=$</b> <output id="showvalue"></output> km/s\n')
            # display chosen velocity of bar too

        if self.hclick:
            self.file_html.write(tabs(2)+'<br><br>\n')
            self.file_html.write(tabs(2)+'&nbsp <label for="clickcoords"><b>Click coordinates:</b></label>\n')
            self.file_html.write(tabs(2)+'<table id="clickcoords">\n\t\t <tbody>\n')
            self.file_html.write(tabs(3)+'<tr><td>&nbsp RA: </td><td id="coordX">--</td></tr>\n')
            self.file_html.write(tabs(3)+'<tr><td>&nbsp Dec: </td><td id="coordY">--</td></tr>\n')
            self.file_html.write(tabs(3)+'<tr><td>&nbsp V: </td><td id="coordZ">--</td></tr>\n')
            self.file_html.write(tabs(2)+'</tbody></table>\n')
            
        self.file_html.write(tabs(1)+'</div>\n')
        
    def func_move2dimage(self, vmax):
        if self.scalev:
            self.file_html.write(tabs(2)+"<script>\n")
            self.file_html.write(tabs(2)+"const inpscam2 = document.querySelector('#scalev');\n")
            self.file_html.write(tabs(2)+"const inpmovem2 = document.querySelector('#move2dimg');\n")
            self.file_html.write(tabs(2)+"const showval = document.querySelector('#showvalue');\n")
            self.file_html.write(tabs(2)+"function move2d()\n\t\t{\n")
            self.file_html.write(tabs(3)+"const sca = inpscam2.value;\n")
            self.file_html.write(tabs(3)+"const move = inpmovem2.value;\n")
            self.file_html.write(tabs(3)+"showval.textContent = roundTo(inpmovem2.value*%s, 3);\n"%vmax)
            self.file_html.write(tabs(3)+"if(document.getElementById('cube__image2d').getAttribute('translation') != '9e9 9e9 9e9') {\n")
            self.file_html.write(tabs(4)+"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(sca*move-1)*%s); }\n"%vmax)
            self.file_html.write(tabs(2)+"}\n")
            self.file_html.write(tabs(2)+"</script>\n")
        else:
            self.file_html.write(tabs(2)+"<script>\n")
            self.file_html.write(tabs(2)+"const inpmovem2 = document.querySelector('#move2dimg');\n")
            self.file_html.write(tabs(2)+"const showval = document.querySelector('#showvalue');\n")
            self.file_html.write(tabs(2)+"function move2d()\n\t\t {\n")
            self.file_html.write(tabs(3)+"const move = inpmovem2.value;\n")
            self.file_html.write(tabs(3)+"showval.textContent = roundTo(inpmovem2.value*%s, 3);\n"%vmax)
            self.file_html.write(tabs(3)+"if(document.getElementById('cube__image2d').getAttribute('translation') != '9e9 9e9 9e9') {\n")
            self.file_html.write(tabs(4)+"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(move-1)*%s); }\n"%vmax)
            self.file_html.write(tabs(2)+"}\n")
            self.file_html.write(tabs(2)+"</script>\n")
        
    def func_colormaps(self, isolevels):
        """
        Make functions to change the colormap of the layers. IMPORTANT: Must be called
        after buttons() function.

        Parameters
        ----------
        colormaps : list of strings
            A list of string with names of Matplotlib colormaps.
            Should also include the default one used in make_layers()
            for the X3D file.
        isolevels : list, array
            An array or list with the value of each isosurface layer. 
            See make_layers().

        Returns
        -------
        None.

        """
        self.file_html.write("\t\t <!--MUST BE BELOW THE <select> ELEMENT-->\n")
        self.file_html.write("\t\t <script>\n")
        self.file_html.write("\t\t const select = document.querySelector('select');\n")
        self.file_html.write("\t\t select.addEventListener('change', change_colormap);\n")
        self.file_html.write("\t\t function change_colormap()\n\t\t {\n")
        self.file_html.write("\t\t\t  const cmap = select.value;\n")
        for (i,cmap) in enumerate(self.colormaps):
            diffCol = create_colormap(cmap, isolevels)
            if i == 0:
                self.file_html.write("\t\t\t if (cmap === '%s') {\n"%cmap)
            else:
                self.file_html.write("\t\t\t else if (cmap === '%s') {\n"%cmap)
            for lev in range(len(isolevels)):
                self.file_html.write("\t\t\t document.getElementById('cube__layer%s').setAttribute('diffuseColor','%s')\n"%(lev,diffCol[lev]))
            self.file_html.write("\t\t\t }\n")
        self.file_html.write("\t\t }\n\t\t </script>\n")
      
        
    def func_scalev(self, nlayers, gals=None, axes='both', coords=None, vmax=None):
        self.file_html.write(tabs(2)+"<script>\n")
        self.file_html.write(tabs(2)+"const inpscasv = document.querySelector('#scalev');\n")
        if vmax != None:
            self.file_html.write(tabs(2)+"const inpmovesv = document.querySelector('#move2dimg');\n")
        #self.file_html.write(tabs(2)+"inpscasv.addEventListener('change', changescalev);\n")
        self.file_html.write(tabs(2)+"function changescalev()\n\t\t {\n")
        self.file_html.write(tabs(3)+"const sca = inpscasv.value;\n")
        if vmax != None:
            self.file_html.write(tabs(3)+"const move = inpmovesv.value;\n")
            self.file_html.write(tabs(3)+"if(document.getElementById('cube__image2d').getAttribute('translation') != '9e9 9e9 9e9') {\n")
            self.file_html.write(tabs(4)+"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(sca*move-1)*%s); }\n"%vmax)
        #scale layers
        for nlays in range(nlayers):
            self.file_html.write(tabs(3)+"document.getElementById('cube__lt%s').setAttribute('scale', '1 1 '+sca);\n"%nlays)
        #scale outline
        self.file_html.write(tabs(3)+"document.getElementById('cube__ot').setAttribute('scale', '1 1 '+sca);\n")
        #move galaxies
        if gals != None:
            for (n,gal) in enumerate(gals):
                v = gals[gal]['coord'][2]
                a = gals[gal]['coord'][0]
                b = gals[gal]['coord'][1]
                self.file_html.write(tabs(3)+"document.getElementById('cube__gct%s').setAttribute('translation', '0 0 '+(sca-1)*%s);\n"%(n,v))
                self.file_html.write(tabs(3)+"document.getElementById('cube__gst%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(n,a,b,v))
                self.file_html.write(tabs(3)+"document.getElementById('cube__glt%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(n,a,b,v))
        #scale grids
        self.file_html.write("\t\t\t document.getElementById('cube__tlt').setAttribute('scale', '1 1 '+sca);\n")
        
        ax, axtick = labpos(coords)
        
        for i in range(12):
            if i < 6:
                if axes == 'diff' or axes == 'both':
                    self.file_html.write(tabs(3)+"document.getElementById('cube__alt_diff%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, ax[i][0], ax[i][1], ax[i][2])) #str(ax[i])[1:-1]
                if axes == 'real' or axes == 'both':
                    self.file_html.write(tabs(3)+"document.getElementById('cube__alt_real%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, ax[i][0], ax[i][1], ax[i][2]))
            if axes == 'diff' or axes == 'both':
                self.file_html.write(tabs(3)+"document.getElementById('cube__att_diff%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, axtick[i][0], axtick[i][1], axtick[i][2]))
            if axes == 'real' or axes == 'both':
                self.file_html.write(tabs(3)+"document.getElementById('cube__att_real%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(i, axtick[i][0], axtick[i][1], axtick[i][2]))
        
        # java for loop and get value from getAttribute()
        #self.file_html.write(tabs(3)+"for (let i = 0; i < %s; i++) {\n"%maxit)
        #self.file_html.write(tabs(4)+"if (i < %s) {\n"%len(gals))
        #self.file_html.write(tabs(5)+"const trans = document.getElementByID('cube__glt'+i).getAttribute('translation').split(' ');\n")
        #self.file_html.write(tabs(5)+"var arr = Float64Array.from(trans)")
        #self.file_html.write(tabs(5)+"arr[2] = arr[2] * sca")
        #self.file_html.write(tabs(5)+"document.getElementById('cube__glt'+i).setAttribute('translation', arr.toString()); }\n")
        #self.file_html.write(tabs(3)+"}\n")
        
        self.file_html.write(tabs(2)+"}\n")
        self.file_html.write(tabs(2)+"</script>\n")
    
    def func_image2d(self, vmax=None, scalev=False):
        """
        Below buttons()

        Returns
        -------
        None.

        """
        if self.hclick == False:
            self.file_html.write(roundTo) #premade string with function to round to two decimals
        self.file_html.write(tabs(2)+"<script>\n")
        if vmax != None:
            if scalev:
                self.file_html.write(tabs(3)+"const inpmovehide = document.querySelector('#move2dimg');\n")
                self.file_html.write(tabs(3)+"const inpscahide = document.querySelector('#scalev');\n")
                self.file_html.write(tabs(2)+"function setimage2d()\n\t\t{\n")
                self.file_html.write(tabs(3)+"const sca = inpscahide.value;\n")
                self.file_html.write(tabs(3)+"const move = inpmovehide.value;\n")
                self.file_html.write(tabs(2)+"if(document.getElementById('cube__image2d').getAttribute('translation')!= '0 0 '+(sca*move-1)*%s)\n"%vmax)
                self.file_html.write(tabs(3)+"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(sca*move-1)*%s);\n"%vmax)
            else:
                self.file_html.write(tabs(3)+"const inpmovehide = document.querySelector('#move2dimg');\n")
                self.file_html.write(tabs(2)+"function setimage2d()\n\t\t{\n")
                self.file_html.write(tabs(3)+"const move = inpmovehide.value;\n")
                self.file_html.write(tabs(2)+"if(document.getElementById('cube__image2d').getAttribute('translation')!= '0 0 '+(move-1)*%s)\n"%vmax)
                self.file_html.write(tabs(3)+"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(move-1)*%s);\n"%vmax)
        else:
            self.file_html.write(tabs(2)+"function setimage2d()\n\t\t{\n")
            self.file_html.write(tabs(2)+"if(document.getElementById('cube__image2d').getAttribute('translation')!= '0 0 0')\n")
            self.file_html.write(tabs(3)+"document.getElementById('cube__image2d').setAttribute('translation', '0 0 0';\n")
        self.file_html.write(tabs(2)+"else \n")
        self.file_html.write(tabs(3)+"document.getElementById('cube__image2d').setAttribute('translation', '9e9 9e9 9e9');\n")
        self.file_html.write(tabs(2)+"}\n\t\t </script>\n")

            
    def close_html(self):
        """
        Must be used to finish and close the HTML file. Not using this function results
        in an error.

        """
        self.file_html.write(tablehtml)
        self.file_html.write('\n\t</body>\n</html>')
        self.file_html.close()
    
class make_all():
    
    def __init__(self, fits, isolevels=None, gals=None, image2d=None, lims=None, unit=None):
        """
        

        Parameters
        ----------
        fits : string
            Path to FITS file containing the data cube to be visualized.
        isolevels : list or array-like, optional
            The values of the datacube from which to create isosurfaces, no maximum length. Must be in the same units as unit,
            but the array itself dimensionless. If None, four layers will be created automatically. The default is None.
        gals : dictionary, optional
            DESCRIPTION. The default is None.
        image2d : tuple, optional
            A two-element list . The default is None.
        lims : TYPE, optional
            DESCRIPTION. The default is None.
        unit : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        cube = SpectralCube.read(fits)
        cubehead = cube.header
        
        self.obj = cubehead['OBJECT']
        nz, ny, nx = cube.shape
        cubeunits = [cubehead['BUNIT'], cubehead['CUNIT1'], cubehead['CUNIT2'], cubehead['CUNIT3']]
        delta = np.array([cubehead['CDELT1']*u.Unit(cubeunits[1]).to('arcsec'),
                      cubehead['CDELT2']*u.Unit(cubeunits[2]).to('arcsec'), 
                      cubehead['CDELT3']*u.Unit(cubeunits[3]).to('km/s')])
        
        print("Cube deltas = %s in arcsec,arcsec and km/s"%delta)
        
        if lims is None:
            lims = np.array([[0,nx-1], [0,ny-1], [0, nz-1]])
            
        ralim = cube.spatial_coordinate_map[1][0,:][lims[0]][::int(np.sign(delta[0]))]
        ramean = np.mean(ralim)
        declim = cube.spatial_coordinate_map[0][:,0][lims[1]][::int(np.sign(delta[1]))]
        decmean = np.mean(declim)
        vlim = cube.spectral_axis[lims[2]][::int(np.sign(delta[2]))]
        vmean = np.mean(vlim).to('km/s')
        #set 'deg', 'deg', 'km/s' for write_x3d()
        coords = np.array([ralim.to('deg'), declim.to('deg'), vlim.to('km/s')])
        
        cube = cube.unmasked_data[lims[2,0]:lims[2,1]+1,lims[1,0]:lims[1,1]+1,lims[0,0]:lims[0,1]+1]
        
        if unit != None:
            cube = cube.to(unit)
        else:
            # for pixel values
            if cube.unit == u.Unit('Jy / beam'):
                i = 0
            elif cube.unit == u.Unit('mJy / beam'):
                i = 1
            elif cube.unit == u.Unit('uJy / beam'):
                i = 2
            while np.max(cube).to_value() < 1:
                if i == 0:
                    cube = cube.to(u.mJy/u.beam)
                elif i == 1:
                    cube = cube.to(u.uJy/u.beam)
                elif i == 2:
                    cube = cube.to(u.nJy/u.beam)
                i = i+1
        
        print("Data in %s"%cube.unit)
        cube = transpose(cube, delta)
        cube = cube.to_value()
        
        if isolevels is None:
            if np.min(cube) < 0:    
                isolevels = [np.max(cube)/10., np.max(cube)/5., np.max(cube)/3., np.max(cube)/1.5]
            elif np.min(cube) < np.max(cube)/5.:
                isolevels = [np.min(cube), np.max(cube)/5., np.max(cube)/3., np.max(cube)/1.5]
            print("Automatic isolevels = "+str(isolevels))
        
        if gals == 'query':
            #all galaxies in v=0 until finding a way to query velocity
            from astroquery.simbad import Simbad
            crit = f"ra < {coords[0,0]} & ra > {coords[0,1]} & dec < {coords[1,0]} & dec > {coords[1,1]} & maintype = G"
            print(crit)
            result_table = Simbad.query_object(crit)
            for gal in result_table:
                galcoords = SkyCoord(ra=gal['RA'], dec=result_table[2]['DEC'], unit="hour,degree")
                galra = (galcoords.ra-ramean)*np.cos(declim[0].to('rad'))
                galdec = (galcoords.dec-decmean)
                gals[gal['MAIN_ID']]['coord']= np.array([galra.to('arcsec').to_value(), galdec.to('arcsec').to_value(), 0])
            
        elif gals is not None:
            for (k,gal) in enumerate(gals.keys()):
                # get galaxy positions from Astropy SkyCoord query. Other options?
                galcoords = SkyCoord.from_name(gal)
                galra = (galcoords.ra-ramean)*np.cos(declim[0].to('rad'))
                galdec = (galcoords.dec-decmean)
                gals[gal]['coord'] = np.array([galra.to('arcsec').to_value(), galdec.to('arcsec').to_value(), gals[gal]['v']])
        
        # if isolevels is None:
        #     if np.min(cube) < 0:
        #         isolevels = np.linspace(0,np.max(cube),25)[[2,6,9,13]]
        #     else:
        #         isolevels = np.linspace(np.min(cube), np.max(cube), 20)[[1,5,8,12]]
                
        self.color = create_colormap('CMRmap', isolevels)
        
        if image2d is not None:
            survey, pixels = image2d
            verts = (coords[0,0], coords[0,1], coords[1,0], coords[1,1])
            print("Downloading image...")
            self.imcol, self.img_shape, _ = get_imcol(position=self.obj, survey=survey, verts=verts, unit='deg',
                                    pixels=pixels, coordinates='J2000',grid=True, gridlabels=True)
            print("Done.")
        
        self.delta = np.abs(delta)
        self.coords = coords
        self.cube = cube
        self.isolevels = isolevels
        self.gals = gals
        if image2d is not None:
            self.x3dim2d = True
        else:
            self.x3dim2d = False
        self.cubehead = cubehead
        self.ramean = ramean.to_value()
        self.decmean = decmean.to_value()
        self.vmean = vmean.to_value()

        
    def choose_buttons(self, layers=True, galaxies=True, gallab=True, grids=True, axes='both', 
                       viewpoints=True, image2d=True, move2d=True, scalev=True, cmaps=None, picking=True):
        """
        Axes can be 'both', 'real', 'diff' or None
        """
        self.layers = layers
        self.galaxies = galaxies
        self.gallab = gallab
        self.grids = grids
        self.axes = axes
        self.viewpoints = viewpoints
        self.image2d = image2d
        self.move2d = move2d
        self.scalev = scalev
        self.cmaps = cmaps
        self.picking = picking
        if cmaps is None:
            self.cmaps = ['magma', 'CMRmap', 'inferno', 'plasma', 'viridis', 'Greys',
           'Blues', 'OrRd', 'PuRd', 'Reds', 'Spectral', 'Wistia',
          'YlGn', 'YlOrRd', 'afmhot', 'autumn', 'cool', 'coolwarm',
          'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_heat',
          'gist_ncar', 'gist_stern', 'gnuplot', 'gnuplot2', 'hot',
          'nipy_spectral', 'prism', 'winter', 'Paired']
        else:
            self.cmaps = cmaps
    
    def make(self, path=None, meta=None, tabtitle=None, pagetitle=None, desc=None):
        """
        

        Parameters
        ----------
        path : string
            Path to where the files will be saved, including the name of the files
            but not the extension. e.g. '~/username/data/somecube'.

        Returns
        -------
        None.

        """
        if path is None:
            path = self.obj
        file = write_x3d(path+'.x3d', delta=self.delta,
                    coords=self.coords, meta=meta, picking=self.picking)
        file.make_layers(self.cube, self.isolevels, self.color)
        file.make_outline()
        if self.gals is not None:
            file.make_galaxies(gals=self.gals, labels=self.gallab)
        if self.x3dim2d:
            file.make_image2d(self.imcol, self.img_shape)
        file.make_ticklines()
        file.make_labels(gals=self.gals, axlab=self.axes) 
        # html.func_scalev(axes) should be same as axlab, not func_axes() though.
        file.close()
        
        
        if tabtitle == None: tabtitle = self.obj
        if pagetitle == None: pagetitle = self.obj+' interactive datacube with X3D'
        if desc == None:
            desc = f"Object: {self.obj}.<t> Telescope: {self.cubehead['TELESCOP']}. RestFreq = {self.cubehead['RESTFREQ']/1e6:.4f} MHz.<br>\n\t Center: (RA,Dec,V)=({np.round(self.ramean,5)}, {np.round(self.decmean,5)}, {np.round(self.vmean,5)} km/s)"
        html = write_html(path+'.html',
                     tabtitle=tabtitle, pagetitle=pagetitle,
                     description=desc)
        if self.layers:
            html.func_layers(len(self.isolevels))
        if self.galaxies:
            html.func_galaxies(self.gals)
        if self.gallab:
            html.func_gallab()
        if self.grids:
            html.func_grids()
        if self.axes is not None:
            html.func_axes(self.axes)
            
        html.start_x3d()
        if self.viewpoints:
            html.viewpoints(maxco=(file.diff_coords[0,2], file.diff_coords[1,2]),
                            vrad=file.diff_coords[2])
        html.close_x3d(path.split('/')[-1]+'.x3d')
        if self.layers or self.galaxies or self.gallab or self.grids or self.axes is not None or self.picking or self.viewpoints or self.image2d or self.cmaps is not None or self.image2d or self.scalev:
            html.buttons(self.isolevels, colormaps=self.cmaps, hide2d=self.image2d, scalev=self.scalev, move2d=self.move2d)
        #func_move2dimage, func_colormaps, func_picking and func_scalev must always go after buttons
        if self.image2d:
            html.func_image2d(vmax=file.diff_coords[2,2], scalev=True)
        if self.picking:
            html.func_pick()
        if self.cmaps is not None:
            html.func_colormaps(self.isolevels)
        if self.scalev:
            html.func_scalev(len(self.isolevels), self.gals, axes=self.axes, coords=file.diff_coords, vmax=file.diff_coords[2,2])
        if self.move2d:
            html.func_move2dimage(vmax=file.diff_coords[2,2])
        html.close_html()
        

#Some miscellaneoues functions

def marching_cubes(cube, level, delta, mins):
    ramin, decmin, vmin = mins
    verts, faces, _, _ = measure.marching_cubes(cube, level = level,
                    #spacing gives the spacing in each dimension.
                    #we multiply by the sign to have the coordinates
                    #in increasing order, same as cube
                            spacing = delta,
                            allow_degenerate=False)
    return np.array([verts[:,0]+ramin, verts[:,1]+decmin, 
                     verts[:,2]+vmin]).T, faces

def calc_scale(shape):
    #scale = 0.71096782*np.sqrt(np.max(shape))-3.84296963 #sqrt
    #scale = 0.02985932*np.max(shape)-0.16425599 #linear
    scale = 3.72083418*np.log(shape)-19.84129672 #logarithmic
    if scale < 1:
        scale = 1
    return scale

def create_colormap(colormap, isolevels):
    colors = cm.get_cmap(colormap)(range(256))[:,:-1]
    if np.sum(colors[0]) < np.sum(colors[-1]):
        colors = colors[::-1]
    cmap = []
    for i in range(len(isolevels)):
        m = 255/(np.max(isolevels)-np.min(isolevels))
        pos = int(m*isolevels[i]-m*np.min(isolevels))
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

def labpos(coords):
    ramin1, ramean1, ramax1 = coords[0]
    decmin1, decmean1, decmax1 = coords[1]
    vmin1, vmean1, vmax1 = coords[2]
    ax = np.array([[0,decmin1*1.1,vmin1],
                   [ramax1*1.1, 0,vmin1],
                   [ramin1,decmin1*1.1,vmean1],
                   [ramin1,0,vmin1*1.1],
                   [ramin1*1.1, decmax1, vmean1],
                   [0, decmax1, vmin1*1.1]])
    axtick = np.array([[ramax1, decmin1*1.1, vmin1], 
                  [ramin1, decmin1*1.1, vmin1], 
                  [ramax1*1.1, decmax1, vmin1], 
                  [ramax1*1.1, decmin1, vmin1], 
                  [ramin1, decmin1*1.1, vmin1], 
                  [ramin1, decmin1*1.1, vmax1], 
                  [ramin1, decmax1, vmin1*1.1],
                  [ramin1, decmin1, vmin1*1.1], 
                  [ramin1*1.1, decmax1, vmin1], 
                  [ramin1*1.1, decmax1, vmax1], 
                  [ramax1, decmax1, vmin1*1.1], 
                  [ramin1, decmax1, vmin1*1.1]])
    return ax, axtick

def preview2d(cube, vmin1=None, vmax1=None, vmin2=None, vmax2=None, norm='asinh'):
    """
    

    Parameters
    ----------
    cube : 3d array
        The data cube. Must be unitless.
    vmin(1,2) : float, optional
        Minimum value for the colormap. If None the minimum of the image is taken. The default is None.
    vmax(1,2) : float, optional
        Maximum value for the colormap. If None the maximum of the image is taken. The default is None.
    norm : string
        A scale name, one of 'asinh', 'function', 'functionlog', 'linear', 'log', 'logit' or 'symlog'. Default is 'asinh'.
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
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
    ax[0,0].hist(cs1.flatten(), density=True)
    ax[0,1].imshow(cs1, vmin=vmin1, vmax=vmax1, norm=norm) #imshow plots axes fist -> y , second -> x
    ax[0,1].set_ylabel('DEC')
    ax[0,1].set_xlabel('RA')

    ax[0,1].set_yticks(np.arange(0,ny+1,50), major=True)
    ax[0,1].set_xticks(np.arange(0,nx+1,50), major=True)
    ax[0,1].grid(which='major')

    ax[1,0].hist(cs2.flatten(), density=True)
    ax[1,1].imshow(cs2.transpose(), vmin=vmin2, vmax=vmax2, norm=norm) #imshow plots axes fist -> y , second -> x
    ax[1,1].set_ylabel('DEC')
    ax[1,1].set_xlabel('V')

    ax[1,1].set_yticks(np.arange(0,ny+1,50), major=True)
    ax[1,1].set_xticks(np.arange(0,nz+1,50), major=True)
    ax[1,1].grid(which='major')
    
    
def get_imcol(position, survey, verts, unit='deg', cmap='Greys', **kwargs):
    """
    verts = file.real_coords[0,0], file.real_coords[0,2], file.real_coords[1,0], file.real_coords[1,2]

    Parameters
    ----------
    position : TYPE
        DESCRIPTION.
    survey : TYPE
        DESCRIPTION.
    verts : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    imcol : TYPE
        DESCRIPTION.

    """
    from astroquery.skyview import SkyView
    from astropy.coordinates import SkyCoord
    import matplotlib.colors as colors
    
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
        print("Pixel indices for [ra1, dec1, ra2, dec2] = "+list(ll_ra, ll_dec, lr_ra, ul_dec)+". Set 'pixels' parameter higher than the difference between 1 and 2.")
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
    dra, ddec, dv = delta
    return np.transpose(array, (2,1,0))[::int(np.sign(dra)), ::int(np.sign(ddec)),::int(np.sign(dv))]
    
# Some attributes for the classes and functions

roundTo = "\t<script>\n\t\t //Round a float value to x.xx format\n\t\t function roundTo(value, decimals)\n\t\t{\n\t\t\t return (Math.round(value * 10**decimals)) / 10**decimals;\n\t\t }\n\t</script>\n"
    
    
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
side,nam = np.array([['front',"R.A. - Dec."],['side',"V - Dec."],['side2',"V - R.A."],['perspective',"Perspective View"]]).T