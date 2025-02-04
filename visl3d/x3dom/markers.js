function newlayout() {
    document.getElementById('spherediv').style.display = 'none'
    document.getElementById('boxdiv').style.display = 'none'
    document.getElementById('tubdiv').style.display = 'none'
    document.getElementById('condiv').style.display = 'none'
    if (marktype.value != 'none') {
        document.getElementById(marktype.value+'div').style.display = 'inline-block'
    }
}

function makelabel(objtype, x, y, z, sca, lab, num, html=false) {
    const labtra = document.createElement('transform');
    const labbill = document.createElement('billboard');
    const labshape = document.createElement('shape');
    const labape = document.createElement('appearance');
    const labmat = document.createElement('material');
    const labtext = document.createElement('text');
    const labfont = document.createElement('fontstyle');
    labtra.setAttribute('id', objtype+'labtra'+num);
    labtra.setAttribute('translation', x+' '+y+' '+sca*z);
    labtra.setAttribute('rotation', '0 1 0 3.14');
    labtra.setAttribute('scale', '20 20 20');
    labbill.setAttribute('axisOfRotation', '0 0 0');
    labmat.setAttribute('diffuseColor', '0 0 0');
    labmat.setAttribute('id', objtype+'labmat'+num);
    labtext.setAttribute('string', lab);
    labfont.setAttribute('family', 'SANS');
    labfont.setAttribute('topToBottom', 'false');
    labfont.setAttribute('justify', 'BEGIN BEGIN');
    labfont.setAttribute('size', '8');
    labshape.appendChild(labape).appendChild(labmat);
    labshape.appendChild(labtext).appendChild(labfont);
    labtra.appendChild(labbill).appendChild(labshape);
    if (html) {
        document.getElementById('ROOT').appendChild(labtra);
    } else {
        document.getElementById('cube__ROOT').appendChild(labtra);
    }
}

function newSphere(nspheres, selsph) {
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

    lab = document.createElement('input');
    spherediv.appendChild(lab);
    lab.setAttribute('type', 'text');
    lab.setAttribute('id', 'sphlab'+nspheres);
    lab.setAttribute('placeholder', 'Label');

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
    rad.setAttribute('placeholder', 'Radius (50)');

    return nspheres;
}

function changeSphere() { // nspheres, selsph
    for (let i=1; i<=nspheres; i++) {
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

function createSphere(sca, selsph, col, sph_coords, means, delt, trans, html=false) {
    const selsphnum = selsph.value.slice(3);
    var x = Number(document.querySelector('#sphRA'+selsphnum).value);
    var y = Number(document.querySelector('#sphDEC'+selsphnum).value);
    var z = Number(document.querySelector('#sphZ'+selsphnum).value);
    const rad = document.querySelector('#sphrad'+selsphnum);
    x = (x - means[0])/delt[0]*trans[0];
    y = (y - means[1])/delt[1]*trans[1];
    z = (z - means[2])/delt[2]*trans[2];

    const lab = document.querySelector('#sphlab'+selsphnum).value;
    if (document.getElementById('sphtra'+selsphnum) == null) {
        if (lab != '') {
            makelabel('sph', x, y, z, sca, lab, selsphnum, html);
        }
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
        if (html) {
            document.getElementById('ROOT').appendChild(newtra);
        } else {
            document.getElementById('cube__ROOT').appendChild(newtra);
        }
        sph_coords.push([x, y, z, rad.value]);
    } else {
        document.getElementById('sphtra'+selsphnum).setAttribute('translation', x+' '+y+' '+sca*z);
        document.getElementById('sphlabtra'+selsphnum).setAttribute('translation', x+' '+y+' '+sca*z);
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
    document.getElementById('sphlabtra'+selsphnum).remove();
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

function hideSphere(selsph) {
    const selsphnum = selsph.value.slice(3);
    if (document.getElementById('sphmat'+selsphnum).getAttribute('transparency') == '0') {
        document.getElementById('sphmat'+selsphnum).setAttribute('transparency', '1');
        document.getElementById('sphlabmat'+selsphnum).setAttribute('transparency', '1');
    } else {
        document.getElementById('sphmat'+selsphnum).setAttribute('transparency', '0');
        document.getElementById('sphlabmat'+selsphnum).setAttribute('transparency', '0');
    }
}

// Boxes
function newBox(nboxes, selbox) {
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
    lab = document.createElement('input');
    boxdiv.appendChild(lab);
    lab.setAttribute('type', 'text');
    lab.setAttribute('id', 'boxlab'+nboxes);
    lab.setAttribute('placeholder', 'Label');
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

function createBox(sca, selbox, col, box_coords, means, delt, trans, html=false) {
    const selboxnum = selbox.value.slice(3);
    var x = Number(document.querySelector('#boxRA'+selboxnum).value);
    var y = Number(document.querySelector('#boxDEC'+selboxnum).value);
    var z = Number(document.querySelector('#boxZ'+selboxnum).value);
    const rad = document.querySelector('#boxrad'+selboxnum);
    x = (x - means[0])/delt[0]*trans[0];
    y = (y - means[1])/delt[1]*trans[1];
    z = (z - means[2])/delt[2]*trans[2];
    const lab = document.querySelector('#boxlab'+selboxnum).value;
    if (document.getElementById('boxtra'+selboxnum) == null) {
        if (lab != '') {
            makelabel('box', x, y, z, sca, lab, selboxnum, html);
        }
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
        if (html) {
            document.getElementById('ROOT').appendChild(newtra);
        } else {
            document.getElementById('cube__ROOT').appendChild(newtra);
        }
        box_coords.push([x, y, z, rad.value]);
    } else {
        document.getElementById('boxtra'+selboxnum).setAttribute('translation', x+' '+y+' '+sca*z);
        document.getElementById('boxlabtra'+selboxnum).setAttribute('translation', x+' '+y+' '+sca*z);
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
    document.getElementById('boxlabtra'+selboxnum).remove();
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

function hideBox(selbox) {
    const selboxnum = selbox.value.slice(3);
    if (document.getElementById('boxmat'+selboxnum).getAttribute('transparency') == '0') {
        document.getElementById('boxmat'+selboxnum).setAttribute('transparency', '1');
        document.getElementById('boxlabmat'+selboxnum).setAttribute('transparency', '1');
    } else {
        document.getElementById('boxmat'+selboxnum).setAttribute('transparency', '0');
        document.getElementById('boxlabmat'+selboxnum).setAttribute('transparency', '0');
    }
}

// Cones
function newCon(ncones, selcon) {
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
    lab = document.createElement('input');
    condiv.appendChild(lab);
    lab.setAttribute('type', 'text');
    lab.setAttribute('id', 'conlab'+ncones);
    lab.setAttribute('placeholder', 'Label');
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
    rad.setAttribute('placeholder', "Radius (50)");
    height = document.createElement('input');
    condiv.appendChild(height);
    height.setAttribute('type', 'number');
    height.setAttribute('id', 'conheight'+ncones);
    height.setAttribute('min', '0');
    height.setAttribute('max', '2000');
    height.setAttribute('step', '5');
    height.setAttribute('placeholder', "Height (100)");
    ori = document.createElement('input');
    condiv.appendChild(ori);
    ori.setAttribute('type', 'text');
    ori.setAttribute('id', 'conori'+ncones);
    ori.setAttribute('placeholder', "Axis: e.g. '-1 0 1'");

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

function createCon(sca, selcon, col, con_coords, means, delt, trans, html=false) {
    const selconnum = selcon.value.slice(3);
    var x = Number(document.querySelector('#conRA'+selconnum).value);
    var y = Number(document.querySelector('#conDEC'+selconnum).value);
    var z = Number(document.querySelector('#conZ'+selconnum).value);
    const rad = document.querySelector('#conrad'+selconnum);
    const height = document.querySelector('#conheight'+selconnum);
    const ori = document.querySelector('#conori'+selconnum);
    x = (x - means[0])/delt[0]*trans[0];
    y = (y - means[1])/delt[1]*trans[1];
    z = (z - means[2])/delt[2]*trans[2];
    const lab = document.querySelector('#conlab'+selconnum).value;
    if (document.getElementById('contra'+selconnum) == null) {
        if (lab != '') {
            makelabel('con', x, y, z, sca, lab, selconnum, html);
        }
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
        if (html) {
            document.getElementById('ROOT').appendChild(newtra);
        } else {
            document.getElementById('cube__ROOT').appendChild(newtra);
        }
        con_coords.push([x, y, z, height.value, rad.value, ori.value]);
    } else {
        document.getElementById('contra'+selconnum).setAttribute('translation', x+' '+y+' '+sca*z);
        document.getElementById('conlabtra'+selconnum).setAttribute('translation', x+' '+y+' '+sca*z);
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
    document.getElementById('conlabtra'+selconnum).remove();
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

function hideCon(selcon) {
    const selconnum = selcon.value.slice(3);
    if (document.getElementById('conmat'+selconnum).getAttribute('transparency') == '0') {
        document.getElementById('conmat'+selconnum).setAttribute('transparency', '1');
        document.getElementById('conlabmat'+selconnum).setAttribute('transparency', '1');
    } else {
        document.getElementById('conmat'+selconnum).setAttribute('transparency', '0');
        document.getElementById('conlabmat'+selconnum).setAttribute('transparency', '0');
    }
}

// Tubes
function newTub(ntubes, seltub, tubelen) {
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
    lab = document.createElement('input');
    tubdiv.appendChild(lab);
    lab.setAttribute('type', 'text');
    lab.setAttribute('id', 'tublab'+ntubes);
    lab.setAttribute('placeholder', 'Label');
    rad = document.createElement('input');
    tubdiv.appendChild(rad);
    rad.setAttribute('type', 'number');
    rad.setAttribute('id', 'tubrad'+ntubes);
    rad.setAttribute('min', '0');
    rad.setAttribute('max', '500');
    rad.setAttribute('step', '5');
    rad.setAttribute('placeholder', "Radius (30)");
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

function createTub(sca, seltub, col, tub_coords, tubelen, means, delt, trans, html) {
    const seltubnum = seltub.value.slice(3);
    const cyl_coord = [];
    const transtube = trans
    if (document.getElementById('tubtra'+seltubnum+'_1') == null) {
        for (i=1; i<tubelen[seltubnum-1]; i++) {
            var x0 = Number(document.querySelector('#tubRA'+seltubnum+'_'+i).value);
            var y0 = Number(document.querySelector('#tubDEC'+seltubnum+'_'+i).value);
            var z0 = Number(document.querySelector('#tubZ'+seltubnum+'_'+i).value);
            var x1 = Number(document.querySelector('#tubRA'+seltubnum+'_'+Number(i+1)).value);
            var y1 = Number(document.querySelector('#tubDEC'+seltubnum+'_'+Number(i+1)).value);
            var z1 = Number(document.querySelector('#tubZ'+seltubnum+'_'+Number(i+1)).value);
            x0 = (x0 - means[0])/delt[0]*transtube[0];
            y0 = (y0 - means[1])/delt[1]*transtube[1];
            z0 = (z0 - means[2])/delt[2]*transtube[2];
            x1 = (x1 - means[0])/delt[0]*transtube[0];
            y1 = (y1 - means[1])/delt[1]*transtube[1];
            z1 = (z1 - means[2])/delt[2]*transtube[2];
            const rad = document.querySelector('#tubrad'+seltubnum);
            var trans = [(x0+x1)/2, (y0+y1)/2, (z0+z1)/2]
            const diff = [x1-x0, y1-y0, z1-z0]
            const height = Math.sqrt(diff[0]**2+diff[1]**2+(sca*diff[2])**2)*1.015;
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
            if (html) {
                document.getElementById('ROOT').appendChild(newtra);
            } else {
                document.getElementById('cube__ROOT').appendChild(newtra);
            }
        }
        const lab = document.querySelector('#tublab'+seltubnum).value;
        if (lab != '') {
            makelabel('tub', trans[0], trans[1], trans[2], sca, lab, seltubnum, html);
        }
        tub_coords.push(cyl_coord);
    } else {
        for (i=1; i<tubelen[seltubnum-1]; i++) {
            var x0 = Number(document.querySelector('#tubRA'+seltubnum+'_'+i).value);
            var y0 = Number(document.querySelector('#tubDEC'+seltubnum+'_'+i).value);
            var z0 = Number(document.querySelector('#tubZ'+seltubnum+'_'+i).value);
            var x1 = Number(document.querySelector('#tubRA'+seltubnum+'_'+Number(i+1)).value);
            var y1 = Number(document.querySelector('#tubDEC'+seltubnum+'_'+Number(i+1)).value);
            var z1 = Number(document.querySelector('#tubZ'+seltubnum+'_'+Number(i+1)).value);
            x0 = (x0 - means[0])/delt[0]*transtube[0];
            y0 = (y0 - means[1])/delt[1]*transtube[1];
            z0 = (z0 - means[2])/delt[2]*transtube[2];
            x1 = (x1 - means[0])/delt[0]*transtube[0];
            y1 = (y1 - means[1])/delt[1]*transtube[1];
            z1 = (z1 - means[2])/delt[2]*transtube[2];
            const rad = document.querySelector('#tubrad'+seltubnum);
            var trans = [(x0+x1)/2, (y0+y1)/2, (z0+z1)/2]
            const diff = [x1-x0, y1-y0, z1-z0]
            const height = Math.sqrt(diff[0]**2+diff[1]**2+(sca*diff[2])**2)*1.015;
            const angle = Math.acos(diff[1]/height);
            cyl_coord.push([trans,diff]);
            document.getElementById('tubtra'+seltubnum+'_'+i).setAttribute('rotation', sca*diff[2]+' 0 '+(-diff[0])+' '+angle);
            document.getElementById('tubtra'+seltubnum+'_'+i).setAttribute('translation', trans[0]+' '+trans[1]+' '+sca*trans[2]);
            document.getElementById('tub'+seltubnum+'_'+i).setAttribute('height', height.toString());
            document.getElementById('tub'+seltubnum+'_'+i).setAttribute('radius', rad.value);
        }
        document.getElementById('tublabtra'+seltubnum).setAttribute('translation', trans[0]+' '+trans[1]+' '+sca*trans[2]);
        tub_coords[seltubnum-1] = cyl_coord;
    }
    return tub_coords;
}

function removeTub(seltub, tubelen) {
    const seltubnum = seltub.value.slice(3);
    document.getElementById('tublabtra'+seltubnum).remove();
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

function hideTub(seltub, tubelen) {
    const seltubnum = seltub.value.slice(3);
    for (i=1; i<tubelen[seltubnum-1]; i++) {
        if (document.getElementById('tubmat'+seltubnum+'_'+i).getAttribute('transparency') == '0') {
            document.getElementById('tubmat'+seltubnum+'_'+i).setAttribute('transparency', '1');
            document.getElementById('tublabmat'+seltubnum).setAttribute('transparency', '1');
        } else {
            document.getElementById('tubmat'+seltubnum+'_'+i).setAttribute('transparency', '0');
            document.getElementById('tublabmat'+seltubnum).setAttribute('transparency', '0');
        }
    }
}