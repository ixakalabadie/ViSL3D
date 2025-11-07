# ViSL3D
Ixaka Labadie García<br/>
07/11/2025 - v0.5
---
`ViSL3D` is a Python package that creates X3D and HTML files to visualize datacubes from astrophysics in 3D in an interactive way. We use [X3D](https://www.web3d.org/x3d/what-x3d) and [x3dom](https://www.web3d.org/x3d/what-x3d) to represent figures in 3D and to integrate them into an HTML. The models have been made taking the x3d-pathway (Vogt et al. 2016) as a starting point. The current code was made for radio data, although it can be used with other types of 3D data.

[This is an example](https://ixakalabadie.github.io/visualisations/HCG16_custom.html) of a 3D visualisation produced with **ViSL3D**. It shows the HI emission of the Hickson Compact Group 16 along an optical image from DSS2 and the positions of the galaxies from NED.

## Installation

```pip install visl3d```

## How to use

Examples of how to use the package are provided in the Jupyter Notebooks inside [this folder](https://github.com/ixakalabadie/ViSL3D/tree/master/example).<br>
The produced HTML file can be visualised in any standard browser or in notebooks themselves.

## Prerequisites to visualise external X3D file

If you use the function `createVis`, which creates integrates the X3D model in the HTML, there are NO prerequisites. You can directly open the produced HTML file in a browser or run in a Jupyter Notebook.<br>
If you create the X3D and the HTML files separately, using `createX3D` and `createHTML`, follow these instructions:

1. Install a local HTTP Server. Visualisations produced by **ViSL3D** must be opened through a local server in order to be displayed. There are a few options:
    - [Apache](https://httpd.apache.org/) is a popular HTTP server that can be installed in most operating systems.
    - [Python](https://www.python.org/) has a built-in HTTP server that can be used by running `python -m http.server` in the directory where the HTML and X3D files are located.
    - [VS Code](https://code.visualstudio.com/) has a built-in HTTP server that can be used by installing the [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) extension. 

2. Move the created X3D and HTML files into the Apache (or other) DocumentRoot directory. By default, in Apache, it is `/var/www/html` in Linux, `C:\Apache24\htdocs` in Windows, and `/usr/local/var/www` (or similar) in Mac. The DocumentRoot can be found and modified in the `httpd.config` file. Finally, open the visualisation in a browser (most common browsers are supported) by typing the URL `localhost\example_file.html`.

## Features
- Plot any number of contour surfaces,
- Plot galaxies with labels,
- Add a 2D image in any wavelength in background,
- Change the scale of the velocity axis,
- Shift 2D image along velocity axis and display the value,
- Rotate, zoom and pan complete figure,
- Hide/Show different components of the figure,
- Change ax labels,
- Change viewpoints,
- Change the colormap,
- Add markers,
- Represent multiple spectral lines in the same visualisation,
- Overlay spectral lines.

## Disclaimer
This package is in an early stage of development. It is not intended for production use and may contain bugs or incomplete features. These are some issues that you may encounter:

- Long loading time. If the 3D model is large the visualisation might take a long time to render or not render at all. In this case, try to make a cutout of the cube (can be done directly in `ViSL3D`), decrease the resolution of the 3D model (also in `ViSL3D`) or decrease the number of iso-surfaces or increase the S/N of the lowest surface. 

- The animation stops after completing at least one loop. Do not click the button many times if you don't see it stopping.

- The opacity feature does not work properly along the 2D image.

- The calculation of the RMS, which can be chosen as the unit to represent the cube, can be inaccurate. It uses negative voxels to create half a Gaussian and calculates the RMS from that. Therefore, if there is absorption or negative artifacts this calculation will be wrong. Another option is to calculate the RMS yourself and give it in the isosurfaces parameter, e.g., `[3*rms, 5*rms...]`.

Please report any other issue in the [GitHub issues](https://github.com/ixakalabadie/ViSL3D/issues).

## Citation
If you use this package in your work, please cite [Labadie-García et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025A%26C....5200949L/abstract).

## References
Vogt, Owen, Verdes-Montenegro & Borthakur, Advanced Data Visualization in Astrophysics: the X3D Pathway, ApJ 818, 115 (2016) ([arxiv](http://arxiv.org/abs/1510.02796); [ADS](http://adsabs.harvard.edu/abs/2015arXiv151002796V))
