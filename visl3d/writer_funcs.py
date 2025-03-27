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