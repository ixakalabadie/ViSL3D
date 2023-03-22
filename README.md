# cube_x3d
Ixaka Labadie Garcia<br/>
22/03/2023
---
The `cube_x3d` module can create X3D and HTML files to visualize data cubes from astrophysics in 3D, in an interactive way. We use [X3D](https://www.web3d.org/x3d/what-x3d) and [x3dom](https://www.web3d.org/x3d/what-x3d) to represent figures in 3D and integrate them into an HTML. The models have been made taking the x3d-pathway (Vogt et al. 2016) as a starting point. The current code was made for data in radio, right now it only works with cubes including a velocity axis.

## How to use
First of all, the files must be opened through a server to be displayed, e.g. Apache. In the *examples* folder some examples can be found, a simple way to view one is to move the two files (one .html and one .x3d) into the Apache root directory.

To create the files, the notebooks *create_3d_visualization.ipynb* and *make_all.ipynb* can be used, the first one being step by step while the second is more straightforward, with the option of makeing a default visualization (just introducing a FITS file).

The documentation inside the code is not complete and some more comments are needed.

## Features
### Current
- Plot any number of contour surfaces.
- Plot galaxies with labels.
- Add a 2D image in any wavelength in background (from Astroquery).
- Change the scale of the velocity axis.
- Shift 2D image along velocity axis and display the value.
- Rotate, zoom and pan complete figure.
- Hide/Show different components of the figure.
- Change ax labels.
- Change viewpoints.

### Workin on
- Pick the coordinates by clicking the figure.
- Add tubes and rectangular/spherical markers.

### Future
- Show 2D images in a side (moment maps, slices...)
- Plot other type of data (optical, time dependent, large scale structure simulations...)

## References
Vogt, Owen, Verdes-Montenegro & Borthakur, Advanced Data Visualization in Astrophysics: the X3D Pathway, ApJ 818, 115 (2016) ([arxiv](http://arxiv.org/abs/1510.02796); [ADS](http://adsabs.harvard.edu/abs/2015arXiv151002796V))

## Dependencies
- [NumPy](https://numpy.org/) 
- [Scikit-image](https://scikit-image.org/)
- [Matplotlib](https://matplotlib.org/)
- [Astropy](https://www.astropy.org/)
- [Spectral Cube](https://github.com/radio-astro-tools/spectral-cube.git)
- [Astroquery](https://github.com/astropy/astroquery)
