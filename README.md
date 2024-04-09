# cubeX3D
Ixaka Labadie Garcia<br/>
09/04/2024
---
The `cubeX3D` module can create X3D and HTML files to visualize data cubes from astrophysics in 3D, in an interactive way. We use [X3D](https://www.web3d.org/x3d/what-x3d) and [x3dom](https://www.web3d.org/x3d/what-x3d) to represent figures in 3D and integrate them into an HTML. The models have been made taking the x3d-pathway (Vogt et al. 2016) as a starting point. The current code was made for radio data, although can be used with other types of datacubes.

## How to use
First of all, the files must be opened through a server to be displayed, e.g. [Apache](https://httpd.apache.org/). By default, the Apache root directory is `/var/www/html` in Linux and `C:\Apache24\htdocs` in Windows. In the *examples* folder some examples can be found, a simple way to view one is to move the two files (one .html and one .x3d) into the Apache root directory and open them in the browser typing `localhost\example_name.html`. In addition, the *x3dom* folder must be in the same directory as the files.

To create the files, we recommend following the notebook *template_short.ipynb* in the *notebooks* folder. Note that the file *cubeX3D* must be in the python path.

## Features
- Plot any number of contour surfaces.
- Plot galaxies with labels.
- Add a 2D image in any wavelength in background.
- Change the scale of the velocity axis.
- Shift 2D image along velocity axis and display the value.
- Rotate, zoom and pan complete figure.
- Hide/Show different components of the figure.
- Change ax labels.
- Change viewpoints.
- Change the colormap
- Add markers

## References
Vogt, Owen, Verdes-Montenegro & Borthakur, Advanced Data Visualization in Astrophysics: the X3D Pathway, ApJ 818, 115 (2016) ([arxiv](http://arxiv.org/abs/1510.02796); [ADS](http://adsabs.harvard.edu/abs/2015arXiv151002796V))

## Dependencies
### Python
- [NumPy](https://numpy.org/) 
- [Scikit-image](https://scikit-image.org/)
- [Matplotlib](https://matplotlib.org/)
- [Astropy](https://www.astropy.org/)
- [Spectral Cube](https://github.com/radio-astro-tools/spectral-cube.git)
- [Astroquery](https://github.com/astropy/astroquery)
- [Astroquery](https://scipy.org/)
### JavaScript
- [X3DOM](https://www.x3dom.org/)
- [js-colormaps](https://github.com/timothygebhard/js-colormaps)
