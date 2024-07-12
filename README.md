# ViSL3D
Ixaka Labadie García<br/>
12/04/2024
---
`ViSL3D` is a Python package that creates X3D and HTML files to visualize datacubes from astrophysics in 3D in an interactive way. We use [X3D](https://www.web3d.org/x3d/what-x3d) and [x3dom](https://www.web3d.org/x3d/what-x3d) to represent figures in 3D and to integrate them into an HTML. The models have been made taking the x3d-pathway (Vogt et al. 2016) as a starting point. The current code was made for radio data, although it can be used with other types of 3D data.

## Prerequisites

1. Install a local HTTP Server. Visualisations produced by **ViSL3D** must be opened through a local server in order to be displayed. There are a few options:
    - [Apache](https://httpd.apache.org/) is a popular HTTP server that can be installed in most operating systems.
    - [Python](https://www.python.org/) has a built-in HTTP server that can be used by running `python -m http.server` in the directory where the HTML and X3D files are located.
    - [VS Code](https://code.visualstudio.com/) has a built-in HTTP server that can be used by installing the [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) extension.

2. Copy the [x3dom](https://github.com/ixakalabadie/ViSL3D/tree/master/x3dom) folder into the Apache DocumentRoot directory. By default, it is `/var/www/html` in Linux, `C:\Apache24\htdocs` in Windows and `/usr/local/var/www` (or similar) in Mac. The DocumentRoot can be found and modified in the `httpd.config` file.

3. Create a Python environment with the following packages:
    - [NumPy](https://numpy.org/)
    - [Scikit-image](https://scikit-image.org/)
    - [Matplotlib](https://matplotlib.org/)
    - [Astropy](https://www.astropy.org/)
    - [Astroquery](https://github.com/astropy/astroquery)
    - [Scipy](https://scipy.org/)

4. Add [ViSL3D](https://github.com/ixakalabadie/ViSL3D/tree/master/visl3d) to PYTHONPATH (this can be done simply by having it in the same folder as the Python notebook or script).

## How to use

1. Examples of how to use the package are provided in the Jupyter notebook [jupyter_example.ipynb](https://github.com/ixakalabadie/cube_x3d/blob/master/example/jupyter_example.ipynb).

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
