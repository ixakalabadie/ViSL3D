function newlayout() {
    document.getElementById('spherediv').style.display = 'none'
    document.getElementById('boxdiv').style.display = 'none'
    document.getElementById('tubdiv').style.display = 'none'
    document.getElementById('condiv').style.display = 'none'
    if (marktype.value != 'none') {
        document.getElementById(marktype.value+'div').style.display = 'inline-block'
    }
}

function newSphere(nspheres, selsph) {
    nspheres += 1;
    if (selsph.value != 'none') {
        document.getElementById(selsph.value).style.display = 'none'
        // const len = document.getElementById("new-sphere").length
        // change nspheres-1 to next available number
        document.getElementById("new-sphere").add(new Option(label='Sphere '+nspheres, value='sph'+nspheres, true, true));
    } else {
        document.getElementById("new-sphere")[0] = new Option(label='Sphere '+nspheres, value='sph'+nspheres, true, true); //this value is selected in the input button
    }

    spherediv = document.createElement('div');
    spherediv.setAttribute('id', 'sph'+nspheres);
    document.getElementById('spherediv').appendChild(spherediv);
    spherediv.appendChild(document.createElement("br"));
    for (const coor of ['RA','DEC','Z']) {
        sph = document.createElement('input');
        spherediv.appendChild(sph);
        sph.setAttribute('type', 'number');
        sph.setAttribute('id', 'sph'+coor+nspheres);
        sph.setAttribute('step', '1');
        sph.setAttribute('placeholder', coor);
    }
    rad = document.createElement('input');
    spherediv.appendChild(rad);
    rad.setAttribute('type', 'number');
    rad.setAttribute('id', 'sphrad'+nspheres);
    rad.setAttribute('min', '0');
    rad.setAttribute('max', '1000');
    rad.setAttribute('step', '5');
    rad.setAttribute('placeholder', 'Radius');

    return nspheres;
}

function changeSphere() { // nspheres, selsph
    for (let i=1; i<=nspheres; i++) {
        // alert('sph'+i+' '+selsph.value)
        if ('sph'+i != selsph.value) {
            if (document.getElementById('sph'+i) != null) {
                document.getElementById('sph'+i).style.display = 'none';
            }
        }
    }
    if (selsph.value != 'none') {
        document.getElementById(selsph.value).style.display = 'inline-block';
    }
    
}

function createSphere(sca, selsph, col) {
    const selsphnum = selsph.value.slice(3);
    const x = Number(document.querySelector('#sphRA'+selsphnum).value);
    const y = Number(document.querySelector('#sphDEC'+selsphnum).value);
    const z = Number(document.querySelector('#sphZ'+selsphnum).value);
    const rad = document.querySelector('#sphrad'+selsphnum);
    if (document.getElementById('sphtra'+selsphnum) == null) {
        const newtra = document.createElement('transform');
        newtra.setAttribute('id', 'sphtra'+selsphnum);
        const newshape = document.createElement('shape');
        newshape.setAttribute('id', 'sphsha'+selsphnum);
        const newape = document.createElement('appearance');
        const newmat = document.createElement('material');
        newmat.setAttribute('diffuseColor', col.value);
        newmat.setAttribute('id', 'sphmat'+selsphnum);
        const newgeo = document.createElement('sphere');
        newgeo.setAttribute('id', 'sphgeo'+selsphnum);
        newgeo.setAttribute('radius', rad.value);
        newtra.setAttribute('translation', x+' '+y+' '+sca*z);
        newgeo.setAttribute('solid', 'false');
        newshape.appendChild(newape).appendChild(newmat);
        newtra.appendChild(newshape).appendChild(newgeo);
        document.getElementById('cube__ROOT').appendChild(newtra);
        sph_coords.push([x, y, z, rad.value]);
    } else {
        document.getElementById('sphtra'+selsphnum).setAttribute('translation', x+' '+y+' '+sca*z);
        document.getElementById('sphgeo'+selsphnum).setAttribute('radius', rad.value);
        document.getElementById('sphmat'+selsphnum).setAttribute('diffuseColor', col.value);
        sph_coords[selsphnum-1] = [x, y, z, rad.value];
    }
    return sph_coords;
}

function removeSphere(selsph) {
    const selsphnum = selsph.value.slice(3);
    document.getElementById('sphtra'+selsphnum).remove();
    document.getElementById(selsph.value).remove();
    if (document.getElementById("new-sphere").length == 1) {
        document.getElementById("new-sphere")[0] = new Option("None", "none", true, true);
    } else {
        for (let i=0; i<document.getElementById("new-sphere").length; i++) {
            if (document.getElementById("new-sphere")[i].value == selsph.value) {
                document.getElementById("new-sphere")[i].remove();
                document.getElementById("new-sphere")[0].setAttribute('selected', 'selected');
                const next = document.getElementById("new-sphere")[0].value;
                document.getElementById(next).style.display = 'inline-block';
            }
        }
    }
}

// Boxes
function newBox(nboxes, selbox) {
    nboxes += 1;
    if (selbox.value != 'none') {
        document.getElementById(selbox.value).style.display = 'none'
        document.getElementById("new-box").add(new Option(label='Box '+nboxes, value='box'+nboxes, true, true));
    } else {
        document.getElementById("new-box")[0] = new Option(label='Box '+nboxes, value='box'+nboxes, true, true);
    }

    boxdiv = document.createElement('div');
    boxdiv.setAttribute('id', 'box'+nboxes);
    document.getElementById('boxdiv').appendChild(boxdiv);
    boxdiv.appendChild(document.createElement("br"));
    for (const coor of ['RA','DEC','Z']) {
        box = document.createElement('input');
        boxdiv.appendChild(box);
        box.setAttribute('type', 'number');
        box.setAttribute('id', 'box'+coor+nboxes);
        box.setAttribute('step', '1');
        box.setAttribute('placeholder', coor);
    }
    rad = document.createElement('input');
    boxdiv.appendChild(rad);
    rad.setAttribute('type', 'text');
    rad.setAttribute('id', 'boxrad'+nboxes);
    rad.setAttribute('placeholder', "shape, e.g. '20 20 20'");

    return nboxes;
}

function changeBox() {
    for (let i=1; i<=nboxes; i++) {
        if ('box'+i != selbox.value) {
            if (document.getElementById('box'+i) != null) {
                document.getElementById('box'+i).style.display = 'none';
            }
        }
    }
    document.getElementById(selbox.value).style.display = 'inline-block';
}

function createBox(sca, selbox, col, box_coords) {
    const selboxnum = selbox.value.slice(3);

    const x = Number(document.querySelector('#boxRA'+selboxnum).value);
    const y = Number(document.querySelector('#boxDEC'+selboxnum).value);
    const z = Number(document.querySelector('#boxZ'+selboxnum).value);
    const rad = document.querySelector('#boxrad'+selboxnum);
    if (document.getElementById('boxtra'+selboxnum) == null) {
        const newtra = document.createElement('transform');
        newtra.setAttribute('id', 'boxtra'+selboxnum);
        const newshape = document.createElement('shape');
        newshape.setAttribute('id', 'boxsha'+selboxnum);
        const newape = document.createElement('appearance');
        const newmat = document.createElement('material');
        newmat.setAttribute('diffuseColor', col.value);
        newmat.setAttribute('id', 'boxmat'+selboxnum);
        const newgeo = document.createElement('box');
        newgeo.setAttribute('id', 'boxgeo'+selboxnum);
        newgeo.setAttribute('size', rad.value);
        newtra.setAttribute('translation', x+' '+y+' '+sca*z);
        newgeo.setAttribute('solid', 'false');
        newshape.appendChild(newape).appendChild(newmat);
        newtra.appendChild(newshape).appendChild(newgeo);
        document.getElementById('cube__ROOT').appendChild(newtra);
        box_coords.push([x, y, z, rad.value]);
    } else {
        document.getElementById('boxtra'+selboxnum).setAttribute('translation', x+' '+y+' '+sca*z);
        document.getElementById('boxgeo'+selboxnum).setAttribute('size', rad.value);
        document.getElementById('boxmat'+selboxnum).setAttribute('diffuseColor', col.value);
        box_coords[selboxnum-1] = [x, y, z, rad.value];
    }
    return box_coords;
}

function removeBox(selbox) {
    const selboxnum = selbox.value.slice(3);
    document.getElementById('boxtra'+selboxnum).remove();
    document.getElementById(selbox.value).remove();
    if (document.getElementById("new-box").length == 1) {
        document.getElementById("new-box")[0] = new Option("None", "none", true, true);
    } else {
        for (let i=0; i<document.getElementById("new-box").length; i++) {
            if (document.getElementById("new-box")[i].value == selbox.value) {
                document.getElementById("new-box")[i].remove();
                document.getElementById("new-box")[0].setAttribute('selected', 'selected');
                const next = document.getElementById("new-box")[0].value;
                document.getElementById(next).style.display = 'inline-block';
            }
        }
    }
}

// Cones
function newCon(ncones, selcon) {
    ncones += 1;
    if (selcon.value != 'none') {
        document.getElementById(selcon.value).style.display = 'none'
        document.getElementById("new-con").add(new Option(label='Cone '+ncones, value='con'+ncones, true, true));
    } else {
        document.getElementById("new-con")[0] = new Option(label='Cone '+ncones, value='con'+ncones, true, true);
    }

    condiv = document.createElement('div');
    condiv.setAttribute('id', 'con'+ncones);
    document.getElementById('condiv').appendChild(condiv);
    condiv.appendChild(document.createElement("br"));
    for (const coor of ['RA','DEC','Z']) {
        con = document.createElement('input');
        condiv.appendChild(con);
        con.setAttribute('type', 'number');
        con.setAttribute('id', 'con'+coor+ncones);
        con.setAttribute('step', '1');
        con.setAttribute('placeholder', coor);
    }
    rad = document.createElement('input');
    condiv.appendChild(rad);
    rad.setAttribute('type', 'number');
    rad.setAttribute('id', 'conrad'+ncones);
    rad.setAttribute('min', '0');
    rad.setAttribute('max', '1000');
    rad.setAttribute('step', '5');
    rad.setAttribute('placeholder', "Radius");
    height = document.createElement('input');
    condiv.appendChild(height);
    height.setAttribute('type', 'number');
    height.setAttribute('id', 'conheight'+ncones);
    height.setAttribute('min', '0');
    height.setAttribute('max', '2000');
    height.setAttribute('step', '5');
    height.setAttribute('placeholder', "Height");
    ori = document.createElement('input');
    condiv.appendChild(ori);
    ori.setAttribute('type', 'text');
    ori.setAttribute('id', 'conori'+ncones);
    ori.setAttribute('placeholder', "Orientation axis: e.g. '-1 0 1'");

    return ncones;
}

function changeCon() {
    for (let i=1; i<=ncones; i++) {
        if ('con'+i != selcon.value) {
            if (document.getElementById('con'+i) != null) {
                document.getElementById('con'+i).style.display = 'none';
            }
        }
    }
    document.getElementById(selcon.value).style.display = 'inline-block';
}

function createCon(sca, selcon, col, con_coords) {
    const selconnum = selcon.value.slice(3);

    const x = Number(document.querySelector('#conRA'+selconnum).value);
    const y = Number(document.querySelector('#conDEC'+selconnum).value);
    const z = Number(document.querySelector('#conZ'+selconnum).value);
    const rad = document.querySelector('#conrad'+selconnum);
    const height = document.querySelector('#conheight'+selconnum);
    const ori = document.querySelector('#conori'+selconnum);

    if (document.getElementById('contra'+selconnum) == null) {
        const newtra = document.createElement('transform');
        newtra.setAttribute('id', 'contra'+selconnum);
        const newshape = document.createElement('shape');
        newshape.setAttribute('id', 'consha'+selconnum);
        const newape = document.createElement('appearance');
        const newmat = document.createElement('material');
        newmat.setAttribute('diffuseColor', col.value);
        newmat.setAttribute('id', 'conmat'+selconnum);
        const newgeo = document.createElement('cone');
        newgeo.setAttribute('id', 'congeo'+selconnum);
        newgeo.setAttribute('height', height.value); 
        newgeo.setAttribute('bottomRadius', rad.value); 
        newtra.setAttribute('translation', x+' '+y+' '+sca*z);
        newtra.setAttribute('rotation', ori.value+' 1.570796');
        newgeo.setAttribute('solid', 'false');
        newshape.appendChild(newape).appendChild(newmat);
        newtra.appendChild(newshape).appendChild(newgeo);
        document.getElementById('cube__ROOT').appendChild(newtra);
        con_coords.push([x, y, z, height.value, rad.value, ori.value]);
    } else {
        document.getElementById('contra'+selconnum).setAttribute('translation', x+' '+y+' '+sca*z);
        document.getElementById('contra'+selconnum).setAttribute('orientation', ori.value+' 0');
        document.getElementById('congeo'+selconnum).setAttribute('bottomRadius', rad.value);
        document.getElementById('congeo'+selconnum).setAttribute('height', height.value);
        document.getElementById('conmat'+selconnum).setAttribute('diffuseColor', col.value);
        con_coords[selconnum-1] = [x, y, z, height.value, rad.value, ori.value];
    }
    return con_coords;
}

function removeCon(selcon) {
    const selconnum = selcon.value.slice(3);
    document.getElementById('contra'+selconnum).remove();
    document.getElementById(selcon.value).remove();
    if (document.getElementById("new-con").length == 1) {
        document.getElementById("new-con")[0] = new Option("None", "none", true, true);
    } else {
        for (let i=0; i<document.getElementById("new-con").length; i++) {
            if (document.getElementById("new-con")[i].value == selcon.value) {
                document.getElementById("new-con")[i].remove();
                document.getElementById("new-con")[0].setAttribute('selected', 'selected');
                const next = document.getElementById("new-con")[0].value;
                document.getElementById(next).style.display = 'inline-block';
            }
        }
    }
}

// Tubes
function newTub(ntubes, seltub, tubelen) {
    ntubes += 1;
    tubelen.push(2);
    if (seltub.value != 'none') {
        document.getElementById(seltub.value).style.display = 'none'
        document.getElementById("new-tub").add(new Option(label='Tube '+ntubes, value='tub'+ntubes, true, true));
    } else {
        document.getElementById("new-tub")[0] = new Option(label='Tube '+ntubes, value='tub'+ntubes, true, true);
    }

    const tubdiv = document.createElement('div');
    tubdiv.setAttribute('id', 'tub'+ntubes);
    document.getElementById('tubdiv').appendChild(tubdiv);
    tubdiv.appendChild(document.createElement("br"));

    rad = document.createElement('input');
    tubdiv.appendChild(rad);
    rad.setAttribute('type', 'number');
    rad.setAttribute('id', 'tubrad'+ntubes);
    rad.setAttribute('min', '0');
    rad.setAttribute('max', '500');
    rad.setAttribute('step', '5');
    rad.setAttribute('placeholder', "Radius");
    for (i=1 ; i<=2 ; i++) {
        tubdiv.appendChild(document.createElement("br"));
        for (const coor of ['RA','DEC','Z']) {
            tub = document.createElement('input');
            tubdiv.appendChild(tub);
            tub.setAttribute('type', 'number');
            tub.setAttribute('id', 'tub'+coor+ntubes+'_'+i);
            tub.setAttribute('step', '1');
            tub.setAttribute('placeholder', coor+i);
        }
    }
    return ntubes, tubelen;
}

function addpoint(seltub, tubelen) {
    const seltubnum = seltub.value.slice(3);
    tubelen[seltubnum-1] += 1;
    document.getElementById('tub'+seltubnum).appendChild(document.createElement("br"));
    for (const coor of ['RA','DEC','Z']) {
        tub = document.createElement('input');
        document.getElementById('tub'+seltubnum).appendChild(tub);
        tub.setAttribute('type', 'number');
        tub.setAttribute('id', 'tub'+coor+seltubnum+'_'+tubelen[seltubnum-1]);
        tub.setAttribute('step', '1');
        tub.setAttribute('placeholder', coor+tubelen[seltubnum-1]);
    }
    return tubelen;
}

function changeTub() {
    for (let i=1; i<=ntubes; i++) {
        if ('tub'+i != seltub.value) {
            if (document.getElementById('tub'+i) != null) {
                document.getElementById('tub'+i).style.display = 'none';
            }
        }
    }
    document.getElementById(seltub.value).style.display = 'inline-block';
}

function createTub(sca, seltub, col, tub_coords, tubelen) {
    const seltubnum = seltub.value.slice(3);
    const cyl_coord = [];
    if (document.getElementById('tubtra'+seltubnum+'_1') == null) {
        for (i=1; i<tubelen[seltubnum-1]; i++) {
            const x0 = Number(document.querySelector('#tubRA'+seltubnum+'_'+i).value);
            const y0 = Number(document.querySelector('#tubDEC'+seltubnum+'_'+i).value);
            const z0 = Number(document.querySelector('#tubZ'+seltubnum+'_'+i).value);
            const x1 = Number(document.querySelector('#tubRA'+seltubnum+'_'+Number(i+1)).value);
            const y1 = Number(document.querySelector('#tubDEC'+seltubnum+'_'+Number(i+1)).value);
            const z1 = Number(document.querySelector('#tubZ'+seltubnum+'_'+Number(i+1)).value);
            const rad = document.querySelector('#tubrad'+seltubnum);
            const trans = [(x0+x1)/2, (y0+y1)/2, (z0+z1)/2]
            const diff = [x1-x0, y1-y0, z1-z0]
            const height = Math.sqrt(diff[0]**2+diff[1]**2+(sca*diff[2])**2)*1.03;
            const angle = Math.acos(diff[1]/height);
            cyl_coord.push([trans,diff]);
            const newtra = document.createElement('transform');
            newtra.setAttribute('id', 'tubtra'+seltubnum+'_'+i);
            const newshape = document.createElement('shape');
            newshape.setAttribute('id', 'tubsha'+seltubnum+'_'+i);
            const newape = document.createElement('appearance');
            const newmat = document.createElement('material');
            newmat.setAttribute('diffuseColor', col.value);
            newmat.setAttribute('id', 'tubmat'+seltubnum+'_'+i);
            const newgeo = document.createElement('cylinder');
            newgeo.setAttribute('id', 'tub'+seltubnum+'_'+i);
            newgeo.setAttribute('radius', rad.value);
            newgeo.setAttribute('solid', 'false');
            newgeo.setAttribute('height', height.toString());
            newtra.setAttribute('translation', trans[0]+' '+trans[1]+' '+sca*trans[2]);
            newtra.setAttribute('rotation', sca*diff[2]+' 0 '+(-diff[0])+' '+angle);
            newshape.appendChild(newape).appendChild(newmat);
            newtra.appendChild(newshape).appendChild(newgeo);
            document.getElementById('cube__ROOT').appendChild(newtra);
        }
        tub_coords.push(cyl_coord);
    } else {
        for (i=1; i<tubelen[seltubnum-1]; i++) {
            const x0 = Number(document.querySelector('#tubRA'+seltubnum+'_'+i).value);
            const y0 = Number(document.querySelector('#tubDEC'+seltubnum+'_'+i).value);
            const z0 = Number(document.querySelector('#tubZ'+seltubnum+'_'+i).value);
            const x1 = Number(document.querySelector('#tubRA'+seltubnum+'_'+Number(i+1)).value);
            const y1 = Number(document.querySelector('#tubDEC'+seltubnum+'_'+Number(i+1)).value);
            const z1 = Number(document.querySelector('#tubZ'+seltubnum+'_'+Number(i+1)).value);
            const rad = document.querySelector('#tubrad'+seltubnum);
            const trans = [(x0+x1)/2, (y0+y1)/2, (z0+z1)/2]
            const diff = [x1-x0, y1-y0, z1-z0]
            const height = Math.sqrt(diff[0]**2+diff[1]**2+(sca*diff[2])**2)*1.03;
            const angle = Math.acos(diff[1]/height);
            cyl_coord.push([trans,diff]);
            document.getElementById('tubtra'+seltubnum+'_'+i).setAttribute('rotation', sca*diff[2]+' 0 '+(-diff[0])+' '+angle);
            document.getElementById('tubtra'+seltubnum+'_'+i).setAttribute('translation', trans[0]+' '+trans[1]+' '+sca*trans[2]);
            document.getElementById('tub'+seltubnum+'_'+i).setAttribute('height', height.toString());
            document.getElementById('tub'+seltubnum+'_'+i).setAttribute('radius', rad.value);
        }
        tub_coords[seltubnum-1] = cyl_coord;
    }
    return tub_coords;
}

function removeTub(seltub, tubelen) {
    const seltubnum = seltub.value.slice(3);
    for (i=1; i<tubelen[seltubnum-1]; i++) {
        document.getElementById('tubtra'+seltubnum+'_'+i).remove();
    }
    document.getElementById(seltub.value).remove();

    if (document.getElementById("new-tub").length == 1) {
        document.getElementById("new-tub")[0] = new Option("None", "none", true, true);
    } else {
        for (let i=0; i<document.getElementById("new-tub").length; i++) {
            if (document.getElementById("new-tub")[i].value == seltub.value) {
                document.getElementById("new-tub")[i].remove();
                document.getElementById("new-tub")[0].setAttribute('selected', 'selected');
                const next = document.getElementById("new-tub")[0].value;
                document.getElementById(next).style.display = 'inline-block';
            }
        }
    }
}