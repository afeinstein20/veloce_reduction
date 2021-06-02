from __future__ import division, print_function
import astropy.io.fits as pyfits
import numpy as np
import glob
import matplotlib.pyplot as plt
import subtract_overscan as so
from scipy.spatial import KDTree
import os
import pdb
plt.ion()

PEAK_SEARCH_RAD=7

def default_badpix():
    """Return a default bad pixel mask.
    """
    badpix = np.zeros( (4112, 4096), dtype=np.int8)
    badpix[2056:2254,809:812]=1
    badpix[2249:2254,808]=1
    badpix[2240:2253,812]=1
    badpix[2251:2254,813]=1
    badpix[2251:2253,3284:3287]=1 #!!! Maybe not actually bad, but bad-ish once...
    #Edges, including where there is some sort of ghost or other order.
    badpix[4050:]=1
    badpix[:3]=1
    badpix[:,:3]=1
    badpix[:,-3:]=1
    return badpix
    
def get_median_frame(files):
    """Lets get a nice median frame to remove cosmics etc"""
    cube = []
    for f in files:
        cube.append(so.read_and_overscan_correct(f))
    return np.median(np.array(cube), axis=0)

def get_cutouts(frame, peaks, sz=5):
    cutouts = np.empty( (len(peaks),sz,sz) )
    for ix, xy in enumerate(peaks):
        cutouts[ix] = frame[xy[0]-sz//2:xy[0]+sz//2+1, xy[1]-sz//2:xy[1]+sz//2+1]
    return cutouts

def simple_centroids(cutouts):
    sz = cutouts.shape[1]
    xy = np.meshgrid(np.arange(-(sz//2),sz//2+1), np.arange(-(sz//2),sz//2+1))
    n_pk = cutouts.shape[0]
    fluxes = np.sum(np.sum(cutouts, axis=2), axis=1)
    xc = np.empty_like(fluxes)
    yc = np.empty_like(fluxes)
    for i in range(n_pk):
        xc[i] = np.sum(xy[0]*cutouts[i])/fluxes[i]
        yc[i] = np.sum(xy[1]*cutouts[i])/fluxes[i]
    return xc, yc, fluxes

def centroid_diff():
    return    

if __name__=="__main__":
    ddir = '/Users/mireland/data/veloce/180919/ccd_3/'
    files = glob.glob(ddir + '*3035[678].fits')
    print("Finding a median LFC frame from 19 Sep...")
    medframe1 = get_median_frame(files)

    badpix = default_badpix()
    if os.path.isfile('peaks.txt'):
        peaks = np.loadtxt('peaks.txt', dtype=np.int16)
    else:
        #plt.imshow( (1-badpix)*medframe, aspect='auto', vmin=0, vmax=50)
        brightpix = np.where((1-badpix)*medframe1 > 100)
        brightpix = np.array(brightpix).T
        peak_tree = KDTree(brightpix)
        npts = brightpix.shape[0]
        peaks = []
        print("Iterating over bright pixels...")
        npeaks=0
        for pt in brightpix:
            neighbors = peak_tree.query_ball_point(pt, PEAK_SEARCH_RAD)
            fluxes = [medframe1[tuple(brightpix[n])] for n in neighbors]
            candidate = list(brightpix[neighbors[np.argmax(fluxes)]])
            if candidate not in peaks:
                peaks.append(candidate)
                npeaks += 1
                if npeaks % 100==0:
                    print("found {:d} peaks...".format(npeaks))
            
        peaks = np.array(peaks)
        np.savetxt('peaks.txt', peaks, fmt='%d')
        
        #Now display it
        plt.imshow(medframe1, vmax=1000, aspect='auto')
        plt.plot(peaks[:,1], peaks[:,0], '.')
        plt.pause(.01)
    
    #Cut out sub-arrays
    cutouts1 = get_cutouts(medframe1, peaks)
    xc1, yc1, fluxes1 = simple_centroids(cutouts1)
    
    #Compare to next night
    ddir = '/Users/mireland/data/veloce/180920/ccd_3/'
    files = glob.glob(ddir + '*3001[234].fits')
    print("Finding a median LFC frame from 20 Sep...")
    medframe2 = get_median_frame(files)
    cutouts2 = get_cutouts(medframe2, peaks)
    xc2, yc2, fluxes2 = simple_centroids(cutouts2)
    
    flux_min = 4000
    dx = 500
    dy = 500
    x1s = np.arange(100,4100,dx)
    y1s = np.arange(100,4100,dy)
    
    binned_xdiffs = np.empty( len(x1s) )
    binned_xdiff_err = np.empty( len(x1s) )
    binned_ydiffs = np.empty( len(x1s) )
    binned_ydiff_err = np.empty( len(x1s) )
    coord = np.empty( len(x1s) )
    for i, x1 in enumerate(x1s):
        ix = (peaks[:,1] > x1) & (peaks[:,1] < x1 + dx) & (fluxes1 > flux_min)
        binned_xdiffs[i] = np.median( (xc2-xc1)[ix])
        binned_xdiff_err[i] = np.std((xc2-xc1)[ix])/np.sqrt(sum(ix))
        binned_ydiffs[i] = np.median( (yc2-yc1)[ix])
        binned_ydiff_err[i] = np.std((yc2-yc1)[ix])/np.sqrt(sum(ix))
        coord[i] = np.median(peaks[ix,1])
    plt.figure(1)
    plt.clf()
    plt.errorbar(coord, binned_xdiffs, yerr=binned_xdiff_err, label='x diff', fmt='o')
    plt.errorbar(coord, binned_ydiffs, yerr=binned_ydiff_err, label='y diff', fmt='o')
    plt.legend()
    plt.xlabel('Pixel x coordinate')
    plt.ylabel('Mean Centroid offset (pix)')
    plt.title('20 Sep - 19 Sep')
    plt.tight_layout()
    
    binned_xdiffs = np.empty( len(y1s) )
    binned_xdiff_err = np.empty( len(y1s) )
    binned_ydiffs = np.empty( len(y1s) )
    binned_ydiff_err = np.empty( len(y1s) )
    coord = np.empty( len(y1s) )
    for i, y1 in enumerate(y1s):
        ix = (peaks[:,0] > y1) & (peaks[:,0] < y1 + dx) & (fluxes1 > flux_min)
        binned_xdiffs[i] = np.median( (xc2-xc1)[ix])
        binned_xdiff_err[i] = np.std((xc2-xc1)[ix])/np.sqrt(sum(ix))
        binned_ydiffs[i] = np.median( (yc2-yc1)[ix])
        binned_ydiff_err[i] = np.std((yc2-yc1)[ix])/np.sqrt(sum(ix))
        coord[i] = np.median(peaks[ix,0])
    plt.figure(2)
    plt.clf()
    plt.errorbar(coord, binned_xdiffs, yerr=binned_xdiff_err, label='x diff', fmt='o')
    plt.errorbar(coord, binned_ydiffs, yerr=binned_ydiff_err, label='y diff', fmt='o')
    plt.legend()
    plt.xlabel('Pixel y coordinate')
    plt.ylabel('Mean Centroid offset (pix)')
    plt.title('20 Sep - 19 Sep')
    plt.tight_layout()
    