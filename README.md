# cubeX3D
Ixaka Labadie García<br/>
12/04/2024
---
`cubeX3D` is a Python package that creates X3D and HTML files to visualize datacubes from astrophysics in 3D in an interactive way. We use [X3D](https://www.web3d.org/x3d/what-x3d) and [x3dom](https://www.web3d.org/x3d/what-x3d) to represent figures in 3D and to integrate them into an HTML. The models have been made taking the x3d-pathway (Vogt et al. 2016) as a starting point. The current code was made for radio data, although it can be used with other types of 3D data.

## Prerequisites

1. Install a local HTTP Server. Visualisations produced by **cubeX3D** must be opened through a local server (for now) in order to be displayed. We reccomend using [Apache](https://httpd.apache.org/), which is compatible with most OS. 

2. Copy the [x3dom](https://github.com/ixakalabadie/cube_x3d/tree/master/x3dom) folder into the Apache DocumentRoot directory. By default, it is `/var/www/html` in Linux, `C:\Apache24\htdocs` in Windows and `/usr/local/var/www` (or similar) in Mac. The DocumentRoot can be found and modified in the `httpd.config` file.

3. Create a Python environment with the following packages:
    - [NumPy](https://numpy.org/)
    - [Scikit-image](https://scikit-image.org/)
    - [Matplotlib](https://matplotlib.org/)
    - [Astropy](https://www.astropy.org/)
    - [Astroquery](https://github.com/astropy/astroquery)
    - [Scipy](https://scipy.org/)

4. Add [cubeX3D](https://github.com/ixakalabadie/cube_x3d/tree/master/cubeX3D) to PYTHONPATH (this can be done simply by having it in the same folder as the Python notebook or script).

## How to use

1. Examples of how to use the package are provided in the Jupyter notebook [template_short.ipynb](https://github.com/ixakalabadie/cube_x3d/blob/master/notebooks/template_short.ipynb).

2. Move the created X3D and HTML files into the Apache DocumentRoot directory and open the visualisation in a browser (most common browsers are supported) by typing the URL `localhost\example_file.html`.

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
