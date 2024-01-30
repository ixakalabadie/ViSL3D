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


class MakeAll:
    """
    Class to create a X3D model of iso-surfaces with 3D spectral line data.
    """
    
    def __init__(self, path, isolevels=None, units=None, lims=None, gals=None, image2d=None):
        """
        if want to leave some units as default set to -1, e.g. (-1, 'deg', -1, 'Angs')
        Create the X3D and HTML files all at once.
        
        Parameters
        ----------
        path : string
            Path to FITS file containing the data cube to be visualized.
        isolevels : array-like, optional
            The values of the datacube from which to create isosurfaces, no maximum length (not tested big numbers). Must be in the 
            same units given by the "units" parameter, but the array itself dimensionless. If None, four layers will be created automatically. The default is None.
        gals : string, dict or array-like, optional
            Specify what galaxies to plot. There are three options:
                - 'query': plots all galaxies inside the cube with a query in NED (it can get very crowded).
                - An array of names of galaxies to plot. The coordinates will be obtained through NED.
                - A dictionary with the form {'Galaxy Name 1': {'coord': np.array([ra,dec,v]), 'col':'r g b'}}.
                  The coordinates must be quantities (<Quantity>) of angles and velocity.
            The default is None.
        image2d : tuple, optional
            A two-element tuple. The first element is a string with the name of a survey and the second is the dimenson
            of the image in pixels. For example: ('DSS2 Blue', '2000x2000'). The default is None.
        lims : 2D array, optional
            The minimum and maximum of each dimension, to make a cutout, as the number of pixels. The default is None.
        unit : astropy unit, optional
            The unit for the values in the data cube, does not have to be the same the FITS file has.
            Also works in string form. The default is None.

        Returns
        -------
        None.
        """

        from spectral_cube import SpectralCube
        from astropy.coordinates import SkyCoord
        
        cube = SpectralCube.read(path)
        cubehead = cube.header
        self.obj = cubehead['OBJECT']
        nz, ny, nx = cube.shape
        cubeunits = [cubehead['BUNIT'], cubehead['CUNIT1'], cubehead['CUNIT2'], cubehead['CUNIT3']]
        
        if units is not None:
            for i in range(len(units)):
                if units[i] == -1:
                    units[i] = cubeunits[i]
            delta = np.array([cubehead['CDELT1']*u.Unit(cubeunits[1]).to(units[1]),
                          cubehead['CDELT2']*u.Unit(cubeunits[2]).to(units[2]), 
                          cubehead['CDELT3']*u.Unit(cubeunits[3]).to(units[3])])
        else:
            units = cubeunits
            delta = np.array([cubehead['CDELT1'], cubehead['CDELT2'], cubehead['CDELT3']])
        
        if lims is None:
            lims = np.array([[0,nx-1], [0,ny-1], [0, nz-1]])
            
        ralim = cube.spatial_coordinate_map[1][0,:][lims[0]][::int(np.sign(delta[0]))]
        ramean = np.mean(ralim)
        declim = cube.spatial_coordinate_map[0][:,0][lims[1]][::int(np.sign(delta[1]))]
        decmean = np.mean(declim)
        vlim = cube.spectral_axis[lims[2]][::int(np.sign(delta[2]))]
        vmean = np.mean(vlim)
        
        coords = np.array([ralim, declim, vlim])
        print(coords)
        
        cube = cube.unmasked_data[lims[2,0]:lims[2,1]+1,lims[1,0]:lims[1,1]+1,lims[0,0]:lims[0,1]+1]
        
        i = 0
        while np.max(cube).to_value() < 1:
            i = i+1
            cube = cube*10
        if i <10:
            units[0] = "e0%s %s"%(i, units[0])
        elif i > 0:
            units[0] = "e%s %s"%(i, units[0])
            
        cube = transpose(cube, delta)
        cube = cube.to_value()
        
        if isolevels is None:
            isolevels = calc_isolevels(cube)
            print("Automatic isolevels = "+str(isolevels))
        
        if gals == 'query':
            #query a region from NED -> gives velocity
            from astroquery.ipac.ned import Ned
            sc = SkyCoord(ralim[0],declim[0])
            sepa = SkyCoord(ramean,decmean).separation(sc)
            result = Ned.query_region(self.obj, radius=sepa)['Object Name', 'Type', 'RA', 'DEC', 'Velocity']
            result = objquery(result, coords, otype='G')
            galdict = {}
            for gal in result:
                galcoords = SkyCoord(ra=gal['RA']*u.deg, dec=gal['DEC']*u.deg)
                galra = (galcoords.ra-ramean)*np.cos(declim[0].to('rad'))
                galdec = (galcoords.dec-decmean)
                galdict[gal['Object Name']] = {'coord':np.array([galra.to('arcsec').to_value(), galdec.to('arcsec').to_value(), gal['Velocity']-vmean]), 'col': '0 0 1'}

        elif type(gals) == 'dict':
            galdict = {}
            for gal in gals:
                galcoords = SkyCoord(ra=gal['coord'][0], dec=gal['coord'][1])
                galra = (galcoords.ra-ramean)*np.cos(gal['coord'][1].to('rad'))
                galdec = (galcoords.dec-decmean)
                galv = gal['coord'].to(u.km/u.s).to_value-vmean
                galdict[gal] = {'coord':np.array([galra.to('arcsec').to_value(), galdec.to('arcsec').to_value(), galv]), 'col': gal['col']}

        elif gals is not None:
            # make so that you can introduce dictionary with just name and color?
            from astroquery.ipac.ned import Ned
            galdict = {}
            for gal in gals:
                result = Ned.query_object(gal)
                galcoords = SkyCoord(ra=result['RA']*u.deg, dec=result['DEC']*u.deg)
                galra = (galcoords.ra-ramean)*np.cos(declim[0].to('rad'))
                galdec = (galcoords.dec-decmean)
                galdict[gal] = {'coord':np.array([galra.to('arcsec').to_value(), galdec.to('arcsec').to_value(), gal['Velocity']-vmean]), 'col': '0 0 1'}

        self.color = create_colormap('CMRmap', isolevels)
        
        if image2d == -1:
            #CREATE INDEXEDSURFACE FOR A PLANE, JUST NEED FOUR POINTS
            self.imcol = None
            self.img_shape = None
        elif image2d is not None:
            survey, pixels = image2d
            verts = (coords[0,0], coords[0,1], coords[1,0], coords[1,1])
            print("Downloading image...")
            self.imcol, self.img_shape, _ = get_imcol(position=self.obj, survey=survey, verts=verts, unit='deg',
                                    pixels=pixels, coordinates='J2000',grid=True, gridlabels=True)
            #Allow using kwargs in get_imcol
            print("Done.")
        
        self.delta = np.abs(delta)
        self.coords = coords
        self.cube = cube
        self.isolevels = isolevels
        self.hdr = cubehead
        if gals is not None:
            self.gals = galdict
        else:
            self.gals = None
        if image2d is not None:
            self.x3dim2d = True
        else:
            self.x3dim2d = False
        self.cubehead = cubehead
        self.cubeunits = cubeunits
        self.units = units
        
    def choose_buttons(self, layers=True, galaxies=True, gallab=True, grids=True, axes='both', 
                       viewpoints=True, image2d=True, move2d=True, scalev=True, cmaps=None, picking=True):
        """
        Choose the buttons to be included in the HTML.

        Parameters
        ----------
        layers : bool
            Hide/show different isosurfaces. Default is True.
        galaxies : bool
            Hide/show galaxy positions. Default is True.
        gallab : bool
            Hide/show galaxy labels. Default is True.
        grids : bool
            Hide/Show grids. Default is True.
        axes : string
            Hide/show axes labels. 'real' for only the actual values, 'diff for only the difference to 
            the center of the cube, 'both' for both, or None for none. Default is 'both'.
        viewpoints : bool
            Show different viewpoints (ra-dec, dec-z, ra-z). Default is True. FALSE GIVES ERROR. 
        image2d : bool
            Hide/show the 2D image in the background. Default is True.
        move2d : bool
            Move 2D image along the Z axis and show the value it is at. Default is True.
        scalev : bool
            Change the scale of the Z axis. Default is True.
        cmaps : list
            List of colormaps to give as an option, must be from matplotlib.colormaps. If None a default set of colormaps is given.
            Default is None. MUST CHANGE TO GIVE OPTION NOT TO CHANGE COLORMAP.
        picking : bool
            Show coordinates by clicking in the isosurfaces. Default is True.
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
            self.cmaps = default_cmaps
        else:
            self.cmaps = cmaps
            
    def make(self, path=None, meta=None, tabtitle=None, pagetitle=None, desc=None):
        """
        Create the X3D and HTML files.

        Parameters
        ----------
        path : string
            Path to where the files will be saved, including the name of the files but not the extension,
            e.g. '~/username/data/somecube'. If None, it will create the files in the working directory.
        meta : string
            Something. Default is None.
        tabtitle : string
            Title of the tab. If None, it is made from the header.
        pagetitle : string
            Title of the web page. If None, it is made from the header.
        desc : string
            Some text to include at the beggining of the HTML. If None, a default describtion will be created from
            the cube header. Default is None.

        Returns
        -------
        Creates an X3D file and an HTML file in path.

        """
        if path is None:
            path = self.obj
        file = write_x3d(path+'.x3d', delta=self.delta, header=self.hdr, units=self.units,
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
            try:
                desc = f"Object: {self.obj}.<t> Telescope: {self.cubehead['TELESCOP']}. RestFreq = {self.cubehead['RESTFREQ']/1e6:.4f} MHz.<br>\n\t Center: (RA,Dec,V)=({np.round(self.ramean,5)}, {np.round(self.decmean,5)}, {np.round(self.vmean,5)} km/s)"
            except:
                pass
        html = write_html(path+'.html', units=self.units,
                     tabtitle=tabtitle, pagetitle=pagetitle,
                     description=desc)
        if self.layers:
            html.func_layers(self.isolevels)
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
            html.viewpoints(maxcoord=file.diff_coords[:,2])
        html.close_x3d(path.split('/')[-1]+'.x3d')
        if self.layers or self.galaxies or self.gallab or self.grids or self.axes is not None or self.picking or self.viewpoints or self.image2d or self.cmaps is not None or self.scalev:
            html.buttons(self.isolevels, self.color, colormaps=self.cmaps, hide2d=self.image2d, scalev=self.scalev, move2d=self.move2d)
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
        

class write_x3d:
    """
    Class to create a X3D model of iso-surfaces with 3D spectral line data.
    Creates an X3D file with the model.

    Parameters
    ----------
    filename : string
        Name of the file to be created. Should have the extension '.x3d'.
    delta : len 3 array
        Array with the step in each direction of the cube. (TIPS!)
    coords : 3x2 array
        Array with the minimum and maximum of the RA, DEC and VRAD
        in each row, in that order, of the cube.
    header : astropy header (OBJECT NAME)
        Header of the FITS image that is being used (MAYBE HAVING THIS REMOVES THE NEED OF "coords")
    units : list
        A list of strings representing the units to be shown in the model.
        First the unit of the data and then RA, DEC and the spectral axis, in that order.
        (ALSO REMOVE THIS?)
    meta : dictionary
        Dictionary with names of the metadata and their descriptions. Default is None.
    picking : boolean
        Whether the option of picking coordinates will be activated. (NOT IMPLEMENTED). Default is False.
    style : string
        The style of the models. Either 'transparent' or 'opaque'. Default is 'transparent'.
        (ADD THIS AS INTERACTIVE OPTION IN HTML?)

    """
    
    def __init__(self, filename, delta, coords, header, units, meta=None, picking=False, style='transparent'):
        self.delta = delta
        self.hdr = header
        self.units = units
        self.style = style # style can be 'transparent' or 'opaque'
        # set labels and outline black or white depending on style
        if self.style == 'transparent':
            self.col = '0 0 0'
        if self.style == 'opaque':
            self.col = '1 1 1'
        self.real_coords, self.diff_coords = get_coords(coords[0], coords[1], coords[2])
        self.diff_coords[0] = self.diff_coords[0] * u.Unit(header["CUNIT1"]).to(units[1])
        self.diff_coords[1] = self.diff_coords[1] * u.Unit(header["CUNIT2"]).to(units[2])
        self.real_coords[2] = self.real_coords[2] * u.Unit(header["CUNIT3"]).to(units[3])
        self.diff_coords[2] = self.diff_coords[2] * u.Unit(header["CUNIT3"]).to(units[3])
        print(self.diff_coords)
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
        self.file_x3d.write('\n </head>\n\t<Scene doPickPass="%s">\n\t\t<Background DEF="back" skyColor="0.6 0.6 0.6"/>'%picking)
        self.file_x3d.write('\n\t\t<NavigationInfo type=\'"EXAMINE" "ANY"\' speed="4" headlight="true"/>')
        self.file_x3d.write('\n\t\t<DirectionalLight ambientIntensity="1" intensity="0" color="1 1 1"/>')
        self.file_x3d.write('\n\t\t<Transform DEF="ROOT" translation="0 0 0">')
        
    def make_layers(self, l_cubes, l_isolevels, l_colors, shifts=None, step_size=1, add_normals=False):
        #create function to set step size automatically
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
        self.iso_split = []
        for nc in range(numcubes):
            cube_full = l_cubes[nc]
            isolevels = l_isolevels[nc]
            self.iso_split.append(np.zeros((len(isolevels)), dtype=int))
            for i in range(len(isolevels)):

                split = int(np.sum(cube_full>isolevels[i])/700000)+1 # calculate how many times to split the cube, 1 means the cube stays the same
                self.iso_split[nc][i] = split
                nx,ny,nz = cube_full.shape

                for sp in range(split):

                    cube = cube_full[:,:,int(nz/split*sp):int(nz/split*(sp+1))]
                    mins = (self.diff_coords[0][0], self.diff_coords[1][0], self.diff_coords[2][0]*(1-2*sp/split))
                    try:
                        if shifts is not None:
                            verts, faces, normals = marching_cubes(cube, level=isolevels[i], delta=self.delta, mins=mins, shift=shifts[nc], step_size=step_size)
                        else:
                            verts, faces, normals = marching_cubes(cube, level=isolevels[i], delta=self.delta, mins=mins, step_size=step_size)
                    except Exception as ex:
                        print(ex)
                        continue
                    self.file_x3d.write('\n\t\t\t<Transform DEF="%slt%s_sp%s" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">'%(nc,i,sp))
                    self.file_x3d.write('\n\t\t\t\t<Shape DEF="%slayer%s_sp%s_shape">'%(nc,i,sp))
                    self.file_x3d.write('\n\t\t\t\t\t<Appearance sortKey="%s">'%(len(isolevels)-1-i))
                    if self.style == 'transparent':
                        #set color and transparency of layer
                        if i == len(isolevels)-1:
                            op = 0.4
                        else:
                            op = 0.8
                        self.file_x3d.write('\n'+tabs(6)+'<Material DEF="%slayer%s_sp%s" ambientIntensity="0" emissiveColor="0 0 0" diffuseColor="%s" specularColor="0 0 0" shininess="0.0078" transparency="%s"/>'%(nc,i,sp,l_colors[nc][i],op))
                    elif self.style == 'opaque':
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
        self.file_x3d.write('\n\t\t\t\t\t\t<Material DEF="outline" emissiveColor="%s" diffuseColor="0 0 0"/>'%self.col)
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
        ramin1, _, ramax1 = self.diff_coords[0]
        decmin1, _, decmax1 = self.diff_coords[1]
        vmin1, _, vmax1 = self.diff_coords[2]
        self.mar = np.min([ramax1-ramin1, decmax1-decmin1, vmax1-vmin1])
        
        sphereradius = self.mar/45 #min(self.delta)*8
        crosslen = self.mar/20 #min(self.delta)*20
        #create galaxy crosses and spheres
        for i, gal in enumerate(gals.keys()):
            #galaxy crosses
            self.file_x3d.write(tabs(3)+'<Transform DEF="%s_cross_tra" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n'%gal)
            self.file_x3d.write(tabs(4)+'<Shape ispickable="false">\n')
            self.file_x3d.write(tabs(5)+'<Appearance>\n')
            self.file_x3d.write(tabs(6)+'<Material DEF="%s" emissiveColor="%s" diffuseColor="0 0 0"/>\n'%(gal+'_cross',self.col))
            self.file_x3d.write(tabs(5)+'</Appearance>\n')
            #cross indices
            self.file_x3d.write(tabs(5)+'<IndexedLineSet colorPerVertex="true" coordIndex="\n'+tabs(6)+'0 1 -1\n'+tabs(6)+'2 3 -1\n'+tabs(6)+'4 5 -1\n'+tabs(6)+'">\n')
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
        self.file_x3d.write(tabs(3)+'<Transform DEF="tlt" translation="0 0 0" rotation="0 0 1 -0" scale="1 1 1">\n')
        self.file_x3d.write(tabs(4)+'<Shape ispickable="false">\n')
        self.file_x3d.write(tabs(5)+'<Appearance>\n')
        #set color
        self.file_x3d.write(tabs(6)+'<Material DEF="ticklines"  emissiveColor="%s" diffuseColor="0 0 0"/>\n'%self.col)
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
            A string indicating what axis labels to include. Can be 'real' for the equatorial
            coordinates; 'diff' for the difference from the center of the cube or
            'both' for both options. Leave None for no axis labels.
            The default is None.

        """
        self.file_x3d.write('\n\t\t<ProximitySensor DEF="PROX_LABEL" size="1.0e+06 1.0e+06 1.0e+06"/>')
        self.file_x3d.write('\n\t\t<Collision enabled="false">')
        self.file_x3d.write('\n\t\t\t<Transform DEF="TRANS_LABEL">')
        
        ramin1, _ , ramax1 = self.diff_coords[0]
        decmin1, _ , decmax1 = self.diff_coords[1]
        vmin1, _ , vmax1 = self.diff_coords[2]
        ramin2, _, ramax2 = Angle(self.real_coords[0] * u.Unit('deg')).to_string(u.hour, precision=0)
        decmin2, _, decmax2 = Angle(self.real_coords[1] * u.Unit('deg')).to_string(u.degree, precision=0)
        vmin2, _, vmax2 = self.real_coords[2]
        
        # scale of labels
        shape = np.min([ramax1-ramin1, decmax1-decmin1, vmax1-vmin1])
        labelscale = calc_scale(shape)
        
        ax,axtick = labpos(self.diff_coords)
        
        #Names for the axes tick labels
        axticknames1 = np.array([f'{ramax1:.2f}',f'{ramin1:.2f}',f'{decmax1:.2f}',
                       f'{decmin1:.2f}',f'{vmin1:.0f}',f'{vmax1:.0f}',
                       f'{decmax1:.2f}',f'{decmin1:.2f}',f'{vmin1:.0f}',
                       f'{vmax1:.0f}',f'{ramax1:.2f}',f'{ramin1:.2f}'])
        
        axticknames2 = np.array([ramax2, ramin2, decmax2,
                       decmin2, f'{vmin2:.0f}', f'{vmax2:.0f}',
                       decmax2, decmin2, f'{vmin2:.2f}',
                       f'{vmax2:.0f}', ramax2, ramin2])
        
        #galaxy labels
        if gals:
            for (i,gal) in enumerate(gals.keys()):
                self.file_x3d.write('\n\t\t\t\t<Transform DEF="glt%s" translation="%s %s %s" rotation="0 1 0 3.14" scale="%s %s %s">'%(i,gals[gal]['coord'][0],gals[gal]['coord'][1], gals[gal]['coord'][2], labelscale, labelscale, labelscale))
                self.file_x3d.write('\n\t\t\t\t\t<Billboard axisOfRotation="0,0,0"  bboxCenter="0,0,0">')
                self.file_x3d.write('\n\t\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t<Material DEF="%s" diffuseColor="0 0 0" emissiveColor="%s"/>'%('label_'+gal,self.col))
                self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t<Text string="%s">'%gal)
                self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle DEF="%s_fs" family=\'"SANS"\' topToBottom="false" justify=\'"BEGIN" "BEGIN"\' size="8"/>'%gal)
                self.file_x3d.write('\n\t\t\t\t\t\t</Text> \n\t\t\t\t\t\t</Shape>\n\t\t\t\t\t</Billboard>\n\t\t\t\t</Transform>')
        
        axlabnames = get_axlabnames(head=self.hdr, units=self.units)
        
        #CHANGE this, all in same for
        if axlab == 'diff' or axlab == 'both':
            #ax labels
            for i in range(6):
                self.file_x3d.write('\n\t\t\t\t<Transform DEF="alt_diff%s" translation="%s %s %s" rotation="%s" scale="%s %s %s">'%(i,ax[i,0],ax[i,1],ax[i,2],axlabrot[i], labelscale, labelscale, labelscale))
                self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axlab_diff%s" diffuseColor="0 0 0" emissiveColor="%s"/>'%(i,self.col))
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
                self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axtick_diff%s" diffuseColor="0 0 0" emissiveColor="%s"/>'%(i,self.col))
                self.file_x3d.write('\n\t\t\t\t\t\t</Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t<Text string="%s">'%axticknames1[i])
                self.file_x3d.write('\n\t\t\t\t\t\t\t<FontStyle family=\'"SANS"\' topToBottom="false" justify=\'%s\' size="8"/>'%axticklabjus[i])
                self.file_x3d.write('\n\t\t\t\t\t\t</Text>\n\t\t\t\t\t</Shape>\n\t\t\t\t</Transform>')
        
        if axlab == 'real' or axlab == 'both':
            if axlab == 'real': trans = '0'
            if axlab == 'both': trans = '1'
            #ax labels
            for i in range(6):
                self.file_x3d.write('\n\t\t\t\t<Transform DEF="alt_real%s" translation="%s %s %s" rotation="%s" scale="%s %s %s">'%(i,ax[i,0],ax[i,1],ax[i,2],axlabrot[i], labelscale, labelscale, labelscale))
                self.file_x3d.write('\n\t\t\t\t\t<Shape ispickable="false">\n\t\t\t\t\t\t<Appearance>')
                self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axlab_real%s" diffuseColor="0 0 0" emissiveColor="%s" transparency="%s"/>'%(i,self.col,trans))
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
                self.file_x3d.write('\n\t\t\t\t\t\t\t<Material DEF="axtick_real%s" diffuseColor="0 0 0" emissiveColor="%s" transparency="%s"/>'%(i,self.col, trans))
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

class write_html:
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
    def __init__(self, filename, units, l_isolevels, split, tabtitle='new_html_x3d', pagetitle=None, description=None, style='tranparent', format='full'):
        #some attributes to use later
        self.grids = False
        self.gals = False
        self.gallabs = False
        self.axes = False
        self.hclick = False
        self.viewp = False
        self.units = units
        self.anim = False
        self.intmarks = False
        self.style = style
        self.format = format
        self.iso_split = split

        self.file_html = open(filename, 'w')
        self.file_html.write('<html>\n\t <head>\n')
        self.file_html.write('\t\t <title> %s </title>\n'%tabtitle)
        self.file_html.write("\t\t <script type='text/javascript' src='https://www.x3dom.org/download/x3dom.js'></script>\n")
        self.file_html.write("\n\t\t <script type='text/javascript'  src='https://www.maths.nottingham.ac.uk/plp/pmadw/LaTeXMathML.js'></script>\n")
        self.file_html.write("\t\t <link rel='stylesheet' type='text/css' href='https://www.x3dom.org/download/x3dom.css'></link>\n")
        self.file_html.write("\t\t <script type='text/javascript' src='https://code.jquery.com/jquery-3.6.3.min.js'></script>\n")
        self.file_html.write(tabs(2)+'<script src="x3dom/js-colormaps.js"></script> <!-- FOR COLORMAPS IN JS-->\n')
        if format == 'minimal':
            self.file_html.write("\n\t\t<style>\n"+tabs(3)+"x3d\n"+tabs(4)+"{\n"+tabs(5)+"border:2px solid darkorange;\n"+tabs(5)+"width:100%;\n"+tabs(5)+"height: 100%;\n"+tabs(3)+"}\n"+tabs(3)+"</style>\n\t</head>\n\t<body>\n")
        else:
            self.file_html.write("\n\t\t<style>\n"+tabs(3)+"x3d\n"+tabs(4)+"{\n"+tabs(5)+"border:2px solid darkorange;\n"+tabs(5)+"width:95%;\n"+tabs(5)+"height: 80%;\n"+tabs(3)+"}\n"+tabs(3)+"</style>\n\t</head>\n\t<body>\n")
        if pagetitle is not None:
            self.file_html.write('\t<h1 align="middle"> %s </h1>\n'%pagetitle)
            self.file_html.write('\t<hr/>\n')
        if description is not None:
            self.file_html.write("\t<p>\n\t %s</p> \n"%description)

        #ANOTHER WAY TO CHANGE TRANSPARENCY instead of loading()
        # self.file_html.write(tabs(3)+"const nl = [%s,%s,%s];"%())
        # self.file_html.write(tabs(3)+"for (let nc = 0; nc < %s; nc++) {\n"%len(l_isolevels))
        # self.file_html.write(tabs(4)+"for (let nl = 0; nl < ; nl++) {\n"%len(l_isolevels[nc]))
        # self.file_html.write(tabs(5)+"if (nl === %s) \n"+tabs(6)+"const op = 0.4;\n"+tabs(5)+"else\n"+tabs(6)+"const op = 0.8;\n")
        # self.file_html.write(tabs(5)+"document.getElementById('cube__'+nc+'layer'+nl).setAttribute('transparency', op);\n")

        if self.style == 'opaque':
            self.file_html.write(tabs(1)+'<script>\n')
            self.file_html.write(tabs(2)+'function loading() {\n')
            numcubes = len(l_isolevels)
            for nc in range(numcubes):
                isolevels = l_isolevels[nc]
                for nl in range(len(isolevels)):
                    if nl == len(isolevels)-1:
                        op = 0.4
                    else:
                        op = 0.8
                    for sp in range(self.iso_split[nc][nl]):
                        self.file_html.write(tabs(3)+"document.getElementById('cube__%slayer%s_sp%s').setAttribute('transparency', '%s');\n"%(nc,nl,sp,op))
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
        self.nlayers = [len(l) for l in l_isolevels]
        self.file_html.write(tabs(2)+"<script>\n")
        self.file_html.write(tabs(3)+"function hideall() {\n")
        for nc in range(numcubes):
            for i in range(self.nlayers[nc]):
                self.file_html.write(tabs(4)+"setHI%slayer%s();\n"%(nc,i))
        self.file_html.write(tabs(3)+"}\n")
        self.file_html.write(tabs(2)+"</script>\n")
        for nc in range(numcubes):
            for i in range(self.nlayers[nc]):
                if i != self.nlayers[nc]-1:
                    self.file_html.write("\t <script>\n\t \t function setHI%slayer%s()\n\t \t {\n\t \t if(document.getElementById('cube__%slayer%s_sp0').getAttribute('transparency') != '0.8') {\n"%(nc,i,nc,i))
                    self.file_html.write("\t\t document.getElementById('%sbut%s').style.border = '5px dashed black';\n"%(nc,i))
                    for sp in range(self.iso_split[nc][i]):
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s').setAttribute('transparency', '0.8');\n"%(nc,i,sp))
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s_shape').setAttribute('ispickable', 'true');\n"%(nc,i,sp))
                    self.file_html.write("\t\t } else { \n")
                    self.file_html.write("\t\t document.getElementById('%sbut%s').style.border = 'inset black';\n"%(nc,i))
                    for sp in range(self.iso_split[nc][i]):
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s').setAttribute('transparency', '1');\n"%(nc,i,sp))
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s_shape').setAttribute('ispickable', 'false');\n"%(nc,i,sp))
                    self.file_html.write("\t\t } \n\t\t }\n\t </script>\n")
                else:
                    self.file_html.write("\t <script>\n\t\t function setHI%slayer%s()\n\t\t {\n\t \t if(document.getElementById('cube__%slayer%s_sp0').getAttribute('transparency') != '0.4') {\n"%(nc,i,nc,i))
                    self.file_html.write("\t\t document.getElementById('%sbut%s').style.border = '5px dashed black';\n"%(nc,i))
                    for sp in range(self.iso_split[nc][i]):
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s').setAttribute('transparency', '0.4');\n"%(nc,i,sp))
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s_shape').setAttribute('ispickable', 'true');\n"%(nc,i,sp))
                    self.file_html.write("\t\t } else { \n")
                    self.file_html.write("\t\t document.getElementById('%sbut%s').style.border = 'inset black';\n"%(nc,i))
                    for sp in range(self.iso_split[nc][i]):
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s').setAttribute('transparency', '1');\n"%(nc,i,sp))
                        self.file_html.write("\t\t document.getElementById('cube__%slayer%s_sp%s_shape').setAttribute('ispickable', 'false');\n"%(nc,i,sp))
                    self.file_html.write("\t\t } \n\t\t }\n\t </script>\n")

    def func_galaxies(self, gals):
        """
        Make function to hide/show galaxies.
        The X3D file must have galaxies for this to work.

        Parameters
        ----------
        gals : dictionary
            Same dictionary as the one used in make_galaxies().

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
        self.file_html.write(tabs(3)+"if(document.getElementById('cube__ticklines').getAttribute('transparency') == '0') {\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__ticklines').setAttribute('transparency', '1');\n")
        self.file_html.write(tabs(3)+"} else if (document.getElementById('cube__outline').getAttribute('transparency') == '0') {\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__outline').setAttribute('transparency', '1');\n")
        self.file_html.write(tabs(3)+"} else {\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__ticklines').setAttribute('transparency', '0');\n")
        self.file_html.write(tabs(4)+"document.getElementById('cube__outline').setAttribute('transparency', '0');\n")
        self.file_html.write(tabs(3)+"}\n")
        self.file_html.write(tabs(2)+"}\n\t\t </script>\n")
            
    def func_axes(self, axes):
        """
        axes : string
            A string indicating what axis labels to include. Can be 'real' for the equatorial
            coordinates; 'diff' for the difference from the center of the cube or
            'both' for both options. Leave None for no axis labels.
            Must be the same as in make_labels(). The default is None.
        """
        self.axes = True

        if axes == 'both':
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
                
                
        self.file_html.write("\t\t }\n\t\t }\n\t\t </script>\n")
        
    def func_pick(self):
        """
        Allows picking the coordinates by clicking in the figure.
        WORKS WITH VIEWPOINT NOT WITH ORTHOVIEWPOINT.
        NOT FINISHED, DON'T USE.

        """
        self.hclick = True
        self.file_html.write(roundTo) #premade string with function to round to two decimals
        self.file_html.write(tabs(1)+"<script>\n")
        self.file_html.write(tabs(3)+"const picksca = document.querySelector('#scalev');\n")
        self.file_html.write(tabs(2)+"function handleClick(event) {\n")
        self.file_html.write(tabs(3)+"const sca = picksca.value;\n")
        self.file_html.write(tabs(3)+"var coordinates = event.hitPnt;\n")
        self.file_html.write(tabs(3)+"$('#coordX').html(roundTo(coordinates[0], 2)+' %s');\n"%self.units[1])
        self.file_html.write(tabs(3)+"$('#coordY').html(roundTo(coordinates[1], 2)+' %s');\n"%self.units[2])
        self.file_html.write(tabs(3)+"$('#coordZ').html(roundTo(coordinates[2], 2)/sca+' %s');\n"%self.units[3])
        self.file_html.write(tabs(2)+"}\n\t </script>\n")

    def func_animation(self):
        """
        Function to start/stop the animation of the X3D models.
        """
        self.anim = True
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

    def close_x3d(self, x3dname):
        """
        Insert the X3D file and close the X3D part of the HTML.
        Must go after viewpoints() and start_x3d().

        Parameters
        ----------
        x3dname : string
            Name of the X3D file to be inserted.

        """
        if self.hclick:
            self.file_html.write(tabs(3)+'<inline url="%s" nameSpaceName="cube" mapDEFToID="true" onclick="handleClick(event)" onload="loading()"/>\n'%x3dname)
        else:
            self.file_html.write(tabs(3)+'<inline url="%s" nameSpaceName="cube" mapDEFToID="true" onclick="" onload="loading()"/>\n'%x3dname)
        self.file_html.write(tabs(2)+"</scene>\n\t</x3d></center>\n")
        
    def viewpoints(self, maxcoord):
        """
        Define viewpoints for the X3D figure. Must go after start_x3d() and
        before close_x3d(). Mandatory if the X3D does not have a Viewpoint,
        which is the case with the one created with this module.

        Parameters
        ----------
        maxcoord : len 3 array
            Maximum values of RA, DEC and third axis, in that order, for the difference to the center.

        """
        self.viewp = True # to use in buttons()
        ramax, decmax, vmax = maxcoord
        self.file_html.write("\t\t <scene>\n")
        #correct camera postition and FoV, not to clip (hide) the figure
        ma = np.max(maxcoord)
        self.file_html.write(tabs(3)+"<OrthoViewpoint id=\"front\" bind='false' centerOfRotation='0,0,0' description='RA-Dec view' fieldOfView='[%s,%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='0,1,0,3.141593' position='0,0,-%s' zFar='10000' zNear='0.0001' ></OrthoViewpoint>\n"%(-ramax*1.4,-decmax*1.4,ramax*1.4,decmax*1.4,ma*1.4))
        self.file_html.write("\t\t\t <OrthoViewpoint id=\"side\" bind='false' centerOfRotation='0,0,0' description='Z - Dec view' fieldOfView='[%s,%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='0,-1,0,1.570796' position='-%s,0,0' zFar='10000' zNear='0.0001' ></OrthoViewpoint>\n"%(-vmax*1.4,-decmax*1.4,vmax*1.4,decmax*1.4,ma*1.4))
        self.file_html.write("\t\t\t <OrthoViewpoint id=\"side2\" bind='false' centerOfRotation='0,0,0' description='Z - RA view' fieldOfView='[%s,%s,%s,%s]' isActive='false' metadata='X3DMetadataObject' orientation='1,1,1,4.1888' position='0,%s,0' zFar='10000' zNear='0.0001' ></OrthoViewpoint>\n"%(-vmax*1.4,-ramax*1.4,vmax*1.4,ramax*1.4,ma*1.4))

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
        self.intmarks = True
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

    def buttons(self, l_isolevels=None, l_colors=None, colormaps=None, hide2d=False, scalev=False, move2d=False, lineLabs=False, centRot=False, background=False):
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
        lineLabs : False or list of strings
            Strings with the labels to use for different cubes. If False, the labels will be 'Cube 0', 'Cube 1', etc.
        centRot : bool
            Create button to change the center of rotation.
        background : bool
            Create button to change the background color.
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
        
        if centRot:
            self.file_html.write(tabs(2)+'&nbsp <label for="rotationCenter"><b>Center of Rotation</b> </label>\n')
            self.file_html.write(tabs(3)+'<select id="rotationCenter">\n')
            self.file_html.write(tabs(4)+'<option value="Origin">Origin</option>\n')
            for nc in range(len(l_isolevels)):
                if type(lineLabs) == list:
                    self.file_html.write(tabs(4)+'<option value="op%s">%s</option>\n'%(nc,lineLabs[nc]))
                else:
                    self.file_html.write(tabs(4)+'<option value="op%s">Center%s</option>\n'%(nc,nc))
            self.file_html.write(tabs(3)+'</select>\n')
            
        if self.anim:
            self.file_html.write(tabs(3)+'<button id="anim" onclick="animation()">Animation</button>')
        
        if background:
            self.file_html.write(tabs(3)+'&nbsp <label for="back-choice"><b>Background:</b> </label>\n')
            self.file_html.write(tabs(3)+'<input oninput="change_background()" id="back-choice" type="color" value="#999999">\n')

        if self.gallabs:
            self.file_html.write(tabs(3)+'&nbsp <b>Font Size:</b>\n')
            self.file_html.write(tabs(3)+'&nbsp <label for="back-choice">Galaxy: </label>\n')
            self.file_html.write(tabs(3)+'<input oninput="change_galsize()" id="galsize-choice" type="number" min="2" max="100" value="8", step="2">\n')
           
        if l_isolevels is not None:
            numcubes = len(self.nlayers)
            for nc in range(numcubes):
                self.file_html.write(tabs(2)+'<br><br>\n')
                if type(lineLabs) == list:
                    self.file_html.write(tabs(2)+'&nbsp <b>%s (%s):</b>\n'%(lineLabs[nc],self.units[0]))
                else:
                    self.file_html.write(tabs(2)+'&nbsp <b>Cube %s (%s):</b>\n'%(nc,self.units[0]))
                for i in range(self.nlayers[nc]):
                    ca = np.array(l_colors[nc][i].split(' ')).astype(float)*255
                    c = 'rgb('+str(ca.astype(int))[1:-1]+')'
                    if (ca[0]*0.299 + ca[1]*0.587 + ca[2]*0.114) > 130:
                        self.file_html.write(tabs(3)+'<button id="%sbut%s" onclick="setHI%slayer%s();" style="font-size:20px ; border:5px dashed black ; background:%s ; color:black"><b>%s</b></button>\n'%(nc,i,nc,i,c,np.round(l_isolevels[nc][i],1)))
                    else:
                        self.file_html.write(tabs(3)+'<button id="%sbut%s" onclick="setHI%slayer%s();" style="font-size:20px ; border:5px dashed black ; background:%s ; color:white"><b>%s</b></button>\n'%(nc,i,nc,i,c,np.round(l_isolevels[nc][i],1)))
            self.file_html.write(tabs(3)+'&nbsp <button id="all" onclick="hideall()"><b>Invert</b></button>')
            self.file_html.write(tabs(2)+'<br><br>\n')
        elif l_isolevels is None and self.nlayers:
            self.file_html.write(tabs(2)+'<br><br>\n')
            self.file_html.write(tabs(2)+' &nbsp <b>Layers (%s):</b>\n'%self.units[0])
            for i in range(self.nlayers):
                ca = np.array(l_colors[i].split(' ')).astype(float)*255
                c = 'rgb('+str(ca.astype(int))[1:-1]+')'
                self.file_html.write(tabs(3)+'<button id="but%s" onclick="setHI1layer%s();" style="font-size:20px ; background:%s ; color:black"><b>Layer %s</b></button>\n'%(i,i,c,i))
            self.file_html.write(tabs(2)+'<br><br>\n')
                    
        # to separate buttons in two parts
        #if self.grids or self.gals or self.gallabs or self.axes or self.hclick or colormaps is not None:
            #self.file_html.write('\n\t <div style="position:absolute;left:800px;top:140px;width:600px">\n')
            
        if colormaps is not None:
            numcubes = len(self.nlayers)
            for nc in range(numcubes):
                self.colormaps = colormaps
                #self.file_html.write('\t\t <br><br>\n')
                self.file_html.write(tabs(2)+'&nbsp <label for="cmaps-choice%s"><b>Cmap %s</b> </label>\n'%(nc,nc))
                self.file_html.write(tabs(2)+'<select id="cmaps-choice%s">\n'%nc)
                for c in self.colormaps:
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

            
        if scalev:
            #self.file_html.write(tabs(2)+'<br><br>\n')
            self.file_html.write(tabs(2)+'&nbsp <label for="scalev"><b>Z scale:</b> </label>\n')
            self.file_html.write(tabs(2)+'<input oninput="changescalev()" id="scalev" type="range" list="marker" min="0" max="5" step="0.001" value="1"/>\n')
            self.file_html.write(tabs(2)+'<datalist id="marker">\n')
            self.file_html.write(tabs(3)+'<option value="1"></option>\n')
            self.file_html.write(tabs(2)+'</datalist>\n')
            
        if move2d:
            #self.file_html.write('\t\t <br><br>\n')
            self.file_html.write(tabs(2)+'&nbsp <label for="move2dimg"><b>2D image:</b> </label>\n')
            self.file_html.write(tabs(2)+'<input oninput="move2d()" id="move2dimg" type="range" min="-1" max="1" step="0.0001" value="1"/>\n')
            self.file_html.write(tabs(2)+'<b>$Z=$</b> <output id="showvalue"></output> %s\n'%self.units[3])
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
        
    def func_move2dimage(self, diff_vmax, real_vmax=None):
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
        if self.scalev:
            self.file_html.write(tabs(2)+"<script>\n")
            self.file_html.write(tabs(2)+"const inpscam2 = document.querySelector('#scalev');\n")
            self.file_html.write(tabs(2)+"const inpmovem2 = document.querySelector('#move2dimg');\n")
            self.file_html.write(tabs(2)+"const showval = document.querySelector('#showvalue');\n")
            self.file_html.write(tabs(2)+"function move2d()\n\t\t{\n")
            self.file_html.write(tabs(3)+"const sca = inpscam2.value;\n")
            self.file_html.write(tabs(3)+"const move = inpmovem2.value;\n")
            if real_vmax is not None:
                self.file_html.write(tabs(3)+"showval.textContent = roundTo(%s+(move-1)*%s, 3)+' ( '+roundTo(move*%s, 3)+' )';\n"%(real_vmax,diff_vmax,diff_vmax))
            else:
                self.file_html.write(tabs(3)+"showval.textContent = roundTo(move*%s, 3);\n"%(diff_vmax))
            self.file_html.write(tabs(3)+"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(sca*move-1)*Number(document.getElementById('cube__imgCoords').getAttribute('point').slice(24,31))); }\n")
            self.file_html.write(tabs(2)+"</script>\n")
        else:
            self.file_html.write(tabs(2)+"<script>\n")
            self.file_html.write(tabs(2)+"const inpmovem2 = document.querySelector('#move2dimg');\n")
            self.file_html.write(tabs(2)+"const showval = document.querySelector('#showvalue');\n")
            self.file_html.write(tabs(2)+"function move2d()\n\t\t {\n")
            self.file_html.write(tabs(3)+"const move = inpmovem2.value;\n")
            if real_vmax is not None:
                self.file_html.write(tabs(3)+"showval.textContent = roundTo(%s+(move-1)*%s, 3)+' ( '+roundTo(move*%s, 3)+' )';\n"%(real_vmax,diff_vmax,diff_vmax))
            else:
                self.file_html.write(tabs(3)+"showval.textContent = roundTo(move*%s, 3);\n"%(diff_vmax))
            self.file_html.write(tabs(3)+"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(move-1)*Number(document.getElementById('cube__imgCoords').getAttribute('point').slice(24,31)));\n")
            self.file_html.write(tabs(2)+"}\n")
            self.file_html.write(tabs(2)+"</script>\n")

    def func_galsize(self, gals):
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
        self.file_html.write(tabs(4)+"let gals = %s;\n"%str(list(gals.keys())))
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
            self.file_html.write(tabs(5)+"var split%s = %s;\n"%(nc,repr(self.iso_split[nc]).replace('array(','').replace(')','')))
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
            
            
    def func_scalev(self, coords, gals=None, axes='both', move2d=True):
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
        if move2d:
            self.file_html.write(tabs(2)+"const inpmovesv = document.querySelector('#move2dimg');\n")
        #self.file_html.write(tabs(2)+"inpscasv.addEventListener('change', changescalev);\n")
        self.file_html.write(tabs(2)+"function changescalev()\n\t\t {\n")
        self.file_html.write(tabs(3)+"const sca = inpscasv.value;\n")
        if move2d:
            self.file_html.write(tabs(3)+"const move = inpmovesv.value;\n")
            self.file_html.write(tabs(4)+"document.getElementById('cube__image2d').setAttribute('translation', '0 0 '+(sca*move-1)*Number(document.getElementById('cube__imgCoords').getAttribute('point').slice(24,31)));\n")
        #scale layers
        numcubes = len(self.nlayers)
        for nc in range(numcubes):
            for nlays in range(self.nlayers[nc]):
                for sp in range(self.iso_split[nc][nlays]):
                    self.file_html.write(tabs(3)+"document.getElementById('cube__%slt%s_sp%s').setAttribute('scale', '1 1 '+sca);\n"%(nc,nlays,sp))
        #scale outline
        self.file_html.write(tabs(3)+"document.getElementById('cube__ot').setAttribute('scale', '1 1 '+sca);\n")
        #move galaxies
        if gals != None:
            for (n,gal) in enumerate(gals):
                v = gals[gal]['coord'][2]
                a = gals[gal]['coord'][0]
                b = gals[gal]['coord'][1]
                self.file_html.write(tabs(3)+"document.getElementById('cube__%s_cross_tra').setAttribute('translation', '0 0 '+(sca-1)*%s);\n"%(gal,v))
                self.file_html.write(tabs(3)+"document.getElementById('cube__%s_sphere_tra').setAttribute('translation', '%s %s '+sca*%s);\n"%(gal,a,b,v))
                self.file_html.write(tabs(3)+"document.getElementById('cube__glt%s').setAttribute('translation', '%s %s '+sca*%s);\n"%(n,a,b,v))
        #scale grids
        self.file_html.write("\t\t\t document.getElementById('cube__tlt').setAttribute('scale', '1 1 '+sca);\n")

        #scale interactive tubes
        if self.intmarks:
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
        
        self.file_html.write(tabs(2)+"}\n")
        self.file_html.write(tabs(2)+"</script>\n")
    
    def func_image2d(self):
        """
        Function to show/hide the 2D image.
        Must be after buttons().
        """
        if self.hclick == False:
            self.file_html.write(roundTo) #premade string with function to round to two decimals
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
        if self.format != 'minimal':
            self.file_html.write(tablehtml)
        self.file_html.write('\n\t</body>\n</html>')
        self.file_html.close()
    

#Some miscellaneoues functions

def marching_cubes(cube, level, delta, mins, shift=(0,0,0), step_size=1):
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
    ramin, decmin, vmin = mins
    verts, faces, normals, _ = measure.marching_cubes(cube, level = level,
                    #spacing gives the spacing in each dimension.
                    #we multiply by the sign to have the coordinates
                    #in increasing order, same as cube
                            spacing = delta,
                            allow_degenerate=False,
                            step_size=step_size)
    return np.array([verts[:,0]+ramin+shift[0], verts[:,1]+decmin+shift[1], 
                     verts[:,2]+vmin+shift[2]]).T, faces, np.array([normals[:,0], normals[:,1], normals[:,2]]).T

def calc_scale(shape):
    """
    Function to calculate the scale for labels
    Parameters
    ----------
    shape : float
        Minimum between the difference in RA, DEC and spectral axis of the cube.
        shape = np.min([ramax1-ramin1, decmax1-decmin1, vmax1-vmin1])

    Returns
    -------
    scale : float
        Value for the X3D scale parameter of the labels.
    """
    #scale = 0.71096782*np.sqrt(shape)-3.84296963 #sqrt
    #scale = 0.02985932*shape-0.16425599 #linear
    #scale = 3.72083418*np.log(shape)-19.84129672 #logarithmic
    scale = shape*0.01 #linear, seems best (ORIGINALLY 0.01)
    #if scale < 1:
    #    scale = 1
    return scale

def change_magnitude(data, magnitude='rms'):
    """
    Function to change the magnitude of the data.

    Parameters
    ----------
    data : 3D array
        Datacube.
    magnitude : string, optional
        Magnitude to change to. Can be 'rms' or 'percent'. The default is 'rms'.

    Returns
    -------
    Datacube with the new magnitude.
    """
    if magnitude == 'rms':
        # calculate std from first 10 spectral slices and divede data by it.
        rms = np.std(data[:,:,:10])
        return (data/rms, rms)
    if magnitude == 'percent':
        return data/np.max(data)*100
    

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
    npix = np.sum(cube > np.min(isolevels))
    if npix > 5e6:
        step = 1
    else:
        step = npix*2.5e-6


def labpos(coords):
    ramin1, _ , ramax1 = coords[0]
    decmin1, _ , decmax1 = coords[1]
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
    import matplotlib.pyplot as plt

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
    Downloads an image from astroquery and returns the colors of the pixels using
    a certain colormap, in hexadecimal format, as required by 'write_x3d().make_image2d'.
    See astroquery.skyview.SkyView.get_images() for more information.

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
    from astroquery.skyview import SkyView
    from astropy.coordinates import SkyCoord
    import matplotlib.colors as colors
    from astropy import wcs
    
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

def get_axlabnames(head, units):
    return np.array([head['CTYPE1'].split('-')[0]+' ('+units[1]+')',
                     head['CTYPE2'].split('-')[0]+' ('+units[2]+')',
                     head['CTYPE3'].split('-')[0]+' ('+units[3]+')',
                     head['CTYPE2'].split('-')[0]+' ('+units[2]+')',
                     head['CTYPE3'].split('-')[0]+' ('+units[3]+')',
                     head['CTYPE1'].split('-')[0]+' ('+units[1]+')'])
    
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
