from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import scipy.ndimage as nd
import glob
import os

def split_to_quadrants(dd, overscan=53):
    """Split a frame into quadrants according to Ian Price's convention"""
    newshape = (dd.shape[0], dd.shape[1]-2*overscan)
    quadshape = (4,newshape[0]//2, newshape[1]//2)
    quads = np.zeros(quadshape)
    
    y0 = 0
    for i in range(quadshape[1]):
        quads[0,i,:] = dd[y0+i,overscan:overscan+newshape[1]//2]
        quads[1,i,:] = dd[y0+i,dd.shape[1]//2:dd.shape[1]-overscan]
    y0 = dd.shape[0]//2
    for i in range(quadshape[1]):
        quads[3,i,:] = dd[y0+i,overscan:overscan+newshape[1]//2]
        quads[2,i,:] = dd[y0+i,dd.shape[1]//2:dd.shape[1]-overscan]
    return quads

def quadrant_medians(infile, median_filter=True):
    """Find the quadrant medians, excluding overscan"""
    quads = split_to_quadrants(pyfits.getdata(infile))
    
    qmeds = np.empty( (4), dtype='int')
    rnoise = np.empty( (4) )
    for i in range(4):
        qmeds[i] = np.median(quads[i])
        absdev = np.abs(quads[i] - qmeds[i])
        mad = np.median(absdev)
        if median_filter:
           qsub = quads[i] - nd.median_filter(quads[i], 5)
        else:
           qsub = quads[i]
        rnoise[i] = np.std(qsub[absdev/mad < 1.4826*5])
    return qmeds, rnoise

if __name__=="__main__":
    dir = '/Users/mireland/data/veloce/'
    if False:
        fnames = glob.glob(dir + 'jamie_darks/*.fits')
        fnames = glob.glob(dir + '12sep_baddark/*.fits')
        fnames = glob.glob(dir + 'turbo/*.fits')
        #fnames = glob.glob(dir + 'annino1/*.fits')
        #fnames = glob.glob(dir + 'annino2/*.fits')
        fnames = glob.glob(dir + 'bad_offset_tests/*.fits')
        fnames = glob.glob(dir + 'gain_bright/*.fits')
        fnames = glob.glob(dir + 'gain_biases/*.fits')
        fnames = glob.glob(dir + 'offset_checks9/*.fits')
        #fnames = glob.glob(dir + 'gain_bright2/*.fits')
        #fnames = glob.glob(dir + 'gain_biases2/*.fits')
        #fnames = glob.glob(dir + '180917/*.fits')
        fnames.sort(key=os.path.getmtime)
        for f in fnames:
            print(f)
            qmeds, rnoise = quadrant_medians(f)
            print(qmeds)
            print(rnoise)
    if False:
        qmeds, rnoise = quadrant_medians(dir + 'speed3/g1_1to2k.fits')
        print("Quadrant ordering test:")
        print(qmeds)
    if True:
        print("Quadrant offset checks:")
        for speed in ['0', '1', '2', '3']:
            for g in ['1','2','5']:
                qmeds, rnoise = quadrant_medians(dir + '180917/offset_checks10/s' + speed + 'g' + g + '_init.fits')
                #print(speed + ', ' + g + ', {:d}, {:d}, {:d}, {:d}'.format(qmeds[0], qmeds[1], qmeds[2], qmeds[3]))
                print(speed + ', ' + g + ', {:5.2f}, {:5.2f}, {:5.2f}, {:5.2f}'.format(rnoise[0], rnoise[1], rnoise[2], rnoise[3]))
    if False:
        print("First pass offset checks")
        for speed in ['0', '1', '2', '3']:
            for g in ['1','2','5']:
                qmeds, rnoise = quadrant_medians(dir + 'speed' + speed + '/g' + g + '_' + offset + '.fits')
                print(speed + ', ' + g + ',' + offset + ', {:d}, {:d}, {:d}, {:d}'.format(qmeds[0], qmeds[1], qmeds[2], qmeds[3]))
                