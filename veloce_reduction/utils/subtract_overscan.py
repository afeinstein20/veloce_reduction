import astropy.io.fits as pyfits
import numpy as np
import glob
import pdb
import matplotlib.pyplot as plt

#Change this to run on the veloce commissioning run test data in your directory.
rootdir = '/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/'

def read_and_overscan_correct(infile, overscan=53, discard_ramp=17, 
    savefile=False, return_overscan=False, ian_price_convention=True):
    """Read in fits file and overscan correct. Assume that
    the overscan region is indepedent of binning."""
    dd = pyfits.getdata(infile)
    newshape = (dd.shape[0], dd.shape[1]-2*overscan)
    quadshape = (4,newshape[0]//2, newshape[1]//2)
    quads = np.zeros(quadshape)
    corrected = np.zeros(newshape)
    overscans = np.zeros((4,dd.shape[0]//2, overscan))
    
    #Split the y axis in 2
    for y0, y1, qix in zip([0, dd.shape[0]//2], [dd.shape[0]//2, dd.shape[0]], [0,2]):
        #Deal with left quadrant
        overscans[qix] = dd[y0:y1,:overscan]
        overscan_ramp = np.median(overscans[qix] + \
            np.random.random(size=overscans[qix].shape) - 0.5, axis=0)
        overscan_ramp -= overscan_ramp[-1] 
        for i in range(len(overscans[qix])):
            overscans[qix,i] = overscans[qix, i] - overscan_ramp
            quads[qix,i,:]= dd[y0+i,overscan:overscan+newshape[1]//2] - \
                np.median(overscans[qix,i] + np.random.random(size=overscans[qix,i].shape) - 0.5)
                
        #Now deal with right quadrant.
        overscans[qix+1] = dd[y0:y1,-overscan:]
        overscan_ramp = np.median(overscans[qix+1] + \
            np.random.random(size=overscans[qix+1].shape) - 0.5, axis=0)
        overscan_ramp -= overscan_ramp[0]  
        for i in range(len(overscans[qix+1])):
            overscans[qix+1,i] = overscans[qix+1,i] - overscan_ramp
            quads[qix+1,i,:] = dd[y0+i,dd.shape[1]//2:dd.shape[1]-overscan] - \
                np.median(overscans[qix+1,i] + np.random.random(size=overscans[qix+1,i].shape) - 0.5)
    qsum = quads[0] + quads[1,:,::-1] + quads[2,::-1,:] + quads[3,::-1,::-1]
    mask = quads > 4e3
    #q2 = quads[2] - quads[0,::-1,:]*2e-3 - quads[3,:,::-1]*0.8e-3
    #q0 = quads[0]-quads[1,:,::-1]*1.5e-3 -quads[2,::-1,:]*7e-4 - quads[3,::-1,::-1]*7e-4
    
    quads[0] = quads[0] - mask[1,:,::-1]*20 - mask[2,::-1,:]*3 - mask[3,::-1,::-1]*3
    quads[1] = quads[1] - mask[0,:,::-1]*18  #Not finished...
    quads[2] = quads[2] - mask[0,::-1,:]*12 - mask[3,:,::-1]*3 
    
    
    #quads[0] imprints everywhere.
    #quads[1] doesn't
    #Saturation is tricky - very bright arc lines go down after going up.
    
    corrected[:newshape[0]//2,:newshape[1]//2] = quads[0]
    corrected[:newshape[0]//2,newshape[1]//2:newshape[1]] = quads[1]
    corrected[newshape[0]//2:newshape[0],:newshape[1]//2] = quads[2]
    corrected[newshape[0]//2:newshape[0],newshape[1]//2:newshape[1]] = quads[3]

    if savefile:
        outfile = infile+'overscan_corrected.fits'

        pyfits.writeto(outfile, corrected, clobber=True)

        print("outfile", outfile)

        return outfile
    else:
        if return_overscan:
            if ian_price_convention:
                overscans[1],overscans[3] = overscans[3], overscans[1].copy()
            return corrected, overscans
        else:
            return corrected    
        
def find_arcs(dir, normal_exptime_only=True):
    all_fn = np.sort(glob.glob(dir + '/?????30???.fits'))
    arc_fn = []
    elapsed = []
    utmjd = []
    utstart = []
    for fn in all_fn:
        hh = pyfits.getheader(fn)
        if hh['OBJECT'].find('ThAr') >= 0:
            arc_fn.append(fn)
            elapsed.append(hh['ELAPSED'])
            utmjd.append(hh['UTMJD'])
            utstart.append(hh['UTSTART'])
    arc_fn = np.array(arc_fn)
    utmjd = np.array(utmjd)
    utstart = np.array(utstart)
    if normal_exptime_only:
        elapsed = np.array(elapsed)
        exptime = np.median(elapsed)
        print("ARC Exposure time: {:5.1f}".format(exptime))
        good = np.where(elapsed == exptime)[0]
        print("Using {:d} of {:d} arcs".format(len(good), len(elapsed)))
        arc_fn = arc_fn[good]
        utmjd = utmjd[good]
        utstart = utstart[good]
    return arc_fn, utmjd, utstart

def arc_shifts(fns, arcsinh_threshold=300):
    arcs = []
    for fn in fns:
        arcs.append(read_and_overscan_correct(fn))
    arcs = np.array(arcs)
    #!!! 
    xy=np.meshgrid(np.arange(arcs.shape[1]//2)-arcs.shape[1]//4, \
        np.arange(arcs.shape[2]//2)-arcs.shape[2]//4, indexing='ij')
    pyramid = np.maximum(np.abs(xy[0]), np.abs(xy[1]))
    window = np.minimum(1024 - pyramid,100)/100
    medarc = np.median(arcs, axis=0)
    medarc_fts = []
    for y0, y1 in zip([0, arcs.shape[1]//2], [arcs.shape[1]//2, arcs.shape[1]]):
        for x0, x1 in zip([0, arcs.shape[2]//2], [arcs.shape[2]//2, arcs.shape[2]]):
            quad = np.arcsinh(medarc[y0:y1,x0:x1]/arcsinh_threshold)
            medarc_fts.append(np.conj(np.fft.rfft2(quad*window)))
    all_shifts = []
    for aa in arcs:
        #Bad pixel correct
        ww = np.where(np.abs(aa-medarc)/(np.abs(medarc)+10) > 1)
        print("{:d} bad pixels.".format(len(ww[0])))
        aa[ww] = medarc[ww]
        qix = 0
        q_shifts = []
        for y0, y1 in zip([0, arcs.shape[1]//2], [arcs.shape[1]//2, arcs.shape[1]]):
            for x0, x1 in zip([0, arcs.shape[2]//2], [arcs.shape[2]//2, arcs.shape[2]]):
                quad = np.arcsinh(aa[y0:y1,x0:x1]/arcsinh_threshold)
                quad_ft = np.fft.rfft2(quad*window)
                ccor = np.fft.fftshift(np.fft.irfft2(quad_ft*medarc_fts[qix]))
                xys = []
                #Central 3 pixels
                zs = [ccor[ccor.shape[0]//2, ccor.shape[0]//2-1], \
                      ccor[ccor.shape[0]//2, ccor.shape[0]//2], \
                      ccor[ccor.shape[0]//2, ccor.shape[0]//2+1]]
                #dz(0.5)=dz1, dz(-0.5)=dz2, dz = ax + b.
                #b = (dz1 + dz2)/2 = (zs[2]-zs[0])/2
                #a = (dz1-dz2) = (zs[0]+zs[2]-2zs[1])
                #intercept = -b/a =  (zs[0] - zs[2])/2/(zs[0]+zs[2]-2zs[1])
                xys.append((zs[0] - zs[2])/2/(zs[0]+zs[2]-2*zs[1]))
                zs = [ccor[ccor.shape[0]//2-1, ccor.shape[0]//2], \
                      ccor[ccor.shape[0]//2, ccor.shape[0]//2], \
                      ccor[ccor.shape[0]//2+1, ccor.shape[0]//2]]
                xys.append((zs[0] - zs[2])/2/(zs[0]+zs[2]-2*zs[1]))
                q_shifts.append(xys)
                qix += 1
        all_shifts.append(q_shifts)
    return arcs, np.array(all_shifts)
    
if __name__=="__main__":
    #Example data to try overscan correction
    corrected = read_and_overscan_correct(rootdir + '180815/bias1.fits')
    corrected[np.where(np.abs(corrected)>20)]=0
    print(np.std(corrected))
    
    #First real night.
    #arc_fn, utmjd, utstart = find_arcs(rootdir + '180814/ccd_3')
    #arcs = arc_shifts(arc_fn[7:])
    
    #Second real night.
    arc_fn, utmjd, utstart = find_arcs(rootdir + '180815/ccd_3') 
    ix = np.concatenate( [np.arange(13,len(arc_fn))]) #np.arange(5) for gain mode 2.
    arcs, sh = arc_shifts(arc_fn[ix])
    print(np.std(np.mean(sh, axis=1)[:,1]))