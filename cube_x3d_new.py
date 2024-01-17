#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 08/10/2023

@author: Ixaka Labadie
"""

import numpy as np
from skimage import measure
from matplotlib import cm
import astropy.units as u

class Visualise3D:
    """
    Class to visualise 3D data
    """

    def __init__(self, filename, cube, header, coords, axes='both', gals=None, animation=False, picking=False, style='transparent', format='full',):
        """
        Parameters
        ----------
        uni
        """
        self.filename = filename
        self.cube = cube
        self.header = header
        self.axes = axes
        self.gals = gals
        self.animation = animation
        self.picking = picking
        self.style = style
        self.format = format
        self.units = [header['BUNIT'], header['CUNIT1'], header['CUNIT2'], header['CUNIT3']]

        self.real_coords, self.diff_coords = get_coords(coords[0], coords[1], coords[2])  

        self.delta = np.array([header['CDELT1'], header['CDELT2'], header['CDELT3']])

    def set_units(self, bunit=None):
        """
        Maybe make option to tranform units, Jy/beam to gas density etc.??
        """
        # Set data units
        self.units[0] = bunit
        if bunit == 'rms':
            from scipy.stats import norm
            _, rms = norm.fit(self.cube[:20].flatten())
            self.cube = self.cube/rms #rms is in units of the cube
        elif bunit == 'percent':
            self.cube = self.cube/np.max(self.cube)*100
        elif bunit is not None:
            self.cube = self.cube*u.Unit(self.header['BUNIT']).to(bunit)
        else:
            mord = 0
            while np.max(self.cube) < 1:
                mord += 3
                self.cube = self.cube*1000
            self.units[0] = "e0%s %s"%(mord,self.header['BUNIT'])
        
        # Set coordinate units
        for i in range(1,4):
            if u.Unit(self.units[i]).is_equivalent(u.rad):
                self.diff_coords[i] = self.diff_coords[i]*u.Unit(self.units[i]).to('arcsec')
                self.units[i] = 'arcsec'
                if self.diff_coords[i] > 3600:
                    self.diff_coords[i] = self.diff_coords[i]*u.Units('arcsec').to('arcmin')
                    self.units[i] = 'arcmin'
                if self.diff_coords[i] > 3600:
                    self.diff_coords[i] = self.diff_coords[i]*u.Units('arcmin').to('deg')
                    self.units[i] = 'deg'
            elif u.Unit(self.units[i]).is_equivalent('m/s'):
                self.diff_coords[i] = self.diff_coords[i]*u.Unit(self.units[i]).to('km/s')
                self.units[i] = 'km/s'
            elif u.Unit(self.units[i]).is_equivalent('Hz'):
                self.diff_coords[i] = self.diff_coords[i]*u.Unit(self.units[i]).to('MHz')
                self.units[i] = 'MHz'
            elif u.Unit(self.units[i]).is_equivalent('m'):
                self.diff_coords[i] = self.diff_coords[i]*u.Unit(self.units[i]).to('nm')
                self.units[i] = 'nm'
        
        self.delta = np.array([self.delta[0]*u.Unit(self.header['CUNIT1']).to(self.units[1]),
                               self.delta[1]*u.Unit(self.header['CUNIT2']).to(self.units[2]),
                               self.delta[2]*u.Unit(self.header['CUNIT3']).to(self.units[3])])

    def start_write(self):
        
        # Start X3D
        if picking:
            picking = 'true'
        else:
            picking = 'false'
        self.file_x3d = open(self.filename+'.x3d', 'w')
        self.file_x3d.write('<?xml version="1.0" encoding="UTF-8"?>\n <!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN" \n "http://www.web3d.org/specifications/x3d-3.3.dtd">')
        self.file_x3d.write('\n <X3D profile="Immersive" version="3.3" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.3.xsd">')
        self.file_x3d.write('\n <head>\n\t<meta name="file" content="%s"/>'%(self.filename+'.x3d'))
        # Additional metadata
        if self.meta != None:
            for met in self.meta.keys():
                self.file_x3d.write('\n\t<meta name="%s" content="%s"/>'%(met,self.meta[met]))
        self.file_x3d.write('\n </head>\n\t<Scene doPickPass="%s">\n\t\t<Background DEF="back" skyColor="0.6 0.6 0.6"/>'%picking)
        self.file_x3d.write('\n\t\t<NavigationInfo type=\'"EXAMINE" "ANY"\' speed="4" headlight="true"/>')
        self.file_x3d.write('\n\t\t<DirectionalLight ambientIntensity="1" intensity="0" color="1 1 1"/>')
        self.file_x3d.write('\n\t\t<Transform DEF="ROOT" translation="0 0 0">')

        # Start HTML
        self.file_html = open(self.filename+'.html', 'w')
        self.file_html.write('<html>\n\t <head>\n')
        self.file_html.write('\t\t <title> %s </title>\n'%self.tabtitle)
        self.file_html.write("\t\t <script type='text/javascript' src='x3dom/x3dom.js'></script>\n")
        self.file_html.write("\n\t\t <script type='text/javascript'  src='https://www.maths.nottingham.ac.uk/plp/pmadw/LaTeXMathML.js'></script>\n")
        self.file_html.write("\t\t <link rel='stylesheet' type='text/css' href='x3dom/x3dom.css'></link>\n")
        self.file_html.write("\t\t <script type='text/javascript' src='https://code.jquery.com/jquery-3.6.3.min.js'></script>\n")
        self.file_html.write(tabs(2)+'<script src="x3dom/js-colormaps.js"></script> <!-- FOR COLORMAPS IN JS-->\n')
        if format == 'minimal':
            self.file_html.write("\n\t\t<style>\n"+tabs(3)+"x3d\n"+tabs(4)+"{\n"+tabs(5)+"border:2px solid darkorange;\n"+tabs(5)+"width:100%;\n"+tabs(5)+"height: 100%;\n"+tabs(3)+"}\n"+tabs(3)+"</style>\n\t</head>\n\t<body>\n")
        else:
            self.file_html.write("\n\t\t<style>\n"+tabs(3)+"x3d\n"+tabs(4)+"{\n"+tabs(5)+"border:2px solid darkorange;\n"+tabs(5)+"width:95%;\n"+tabs(5)+"height: 80%;\n"+tabs(3)+"}\n"+tabs(3)+"</style>\n\t</head>\n\t<body>\n")
        if self.pagetitle is not None:
            self.file_html.write('\t<h1 align="middle"> %s </h1>\n'%self.pagetitle)
            self.file_html.write('\t<hr/>\n')
        if self.description is not None:
            self.file_html.write("\t<p>\n\t %s</p> \n"%self.description)


    def layers(self, l_cubes, l_isolevels, l_colors, l_shifts=None, step_size=1):
        """
        """
        




# Other functions

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