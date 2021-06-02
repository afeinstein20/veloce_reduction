"""This is a super-quick script designed to calculate gain and readout noise 
for Veloce.

Started coding: 3:15pm
Wrote email: 4:50pm. Some other emails, jobs and Subaru stuff in background.

"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import glob
import subtract_overscan as so
import quadrant_medians as qm
import scipy.ndimage as nd
import pdb

gfile = open('/Users/mireland/delete_this/gains.csv','w')
rfile = open('/Users/mireland/delete_this/rnoise.csv','w')

nbins = 20
bin_min = -10

#NB hardwired
szy=4112
szx=4096
ylos = [0, 0, szy//2, szy//2]
yhis = [szy//2, szy//2, szy, szy]
xlos = [0, szx//2, szx//2, 0]
xhis = [szx//2, szx, szx, szx//2]
quadrants = [0,1,2,3]


for speed in [0,1,2,3]:
#    for gain in [2,5]:
    for gain in [5]:
        #Ignore
        bright_prefix = '/Users/mireland/data/veloce/gain_bright/s{:d}g{:d}'.format(speed, gain)

        bright_files = glob.glob(bright_prefix + "*fits")

        #Read in the bright files and compute a pixel-based variance
        bright_smoothed = []
        bright_resid = []
        for fn in bright_files:
            frame, overscans = so.read_and_overscan_correct(fn, return_overscan=True)
            quads = qm.split_to_quadrants(frame, overscan=0)
            q_smoothed = []
            q_resid = []
            for q,o in zip(quads,overscans):
                #First, create a smoothed version of the quadrant...
                medfilt_fullquad = nd.median_filter(q,7)
                q_smoothed.append(medfilt_fullquad[3:-3,3:-3])
                q_resid = (q - medfilt_fullquad)[3:-3,3:-3]

                
                #!!! the following didn't work adequately
                #Median of a chi-squared distribution is 0.4545 times the mean. So
                #we can use the median square to estimate the mean square.
                #q_var = nd.median_filter((q - q_medfilt)**2, 7)[3:-3,3:-3]/0.4545
                #print(np.median(q_medfilt[3:-3,3:-3]/q_var))
            bright_smoothed.append(q_smoothed)
            bright_resid.append(q_smoothed)
        pdb.set_trace()
        
        #bright_cube.append(so.read_and_overscan_correct(fn))
        #bright_cube = np.array(bright_cube)    

        #Now the tricky bit... scale each frame to account for varying flux from the source.
        mean_bright = np.mean(bright_cube, axis=0)
        median_mean_bright = np.median(mean_bright, axis=1)
        for dd in bright_cube:
            for i in range(len(dd)):
                dd[i] *= median_mean_bright[i]/np.median(dd[i])
            #scale_factor = median_mean_bright/np.median(dd)
            #DEBUG print("Scaling by {:5.2f}".format(scale_factor))
            #dd *= scale_factor
        #pdb.set_trace()
        var_bright = np.var(bright_cube, axis=0, ddof=1)

        gains_e_ADU = []
        rnoise_e = []
        for ylo, yhi, xlo, xhi, quadrant in zip(ylos, yhis, xlos, xhis, quadrants):
            print("Working on quadrant: {:d}".format(quadrant))
            var_bright_q = var_bright[ylo:yhi,xlo:xhi]
            mean_bright_q = mean_bright[ylo:yhi,xlo:xhi]
            bin_max = np.percentile(mean_bright_q, 99.5)
    
            #Average these in bins.
            binned_means = np.zeros(nbins)
            binned_vars = np.zeros(nbins)
            for i in range(nbins):
                ww = np.where((mean_bright_q > bin_min + i*(bin_max-bin_min)/nbins) &
                              (mean_bright_q< bin_min + (i+1)*(bin_max-bin_min)/nbins))
                if len(ww[0])>10:
                    binned_means[i] = np.mean(mean_bright_q[ww])
                    binned_vars[i] = np.mean(var_bright_q[ww])
                #DEBUG print("Done bin: {:d}".format(i))
    
            gain_pixels = (var_bright_q - dark_sdev[quadrant]**2)/np.maximum(mean_bright_q,bin_min)
            gain_ADU_e = np.mean(gain_pixels[(gain_pixels < 10*np.median(gain_pixels)) & \
                (gain_pixels > 0*np.median(gain_pixels))])
            gains_ADU_e = (binned_vars - dark_sdev[quadrant]**2)/binned_means
            plt.clf()
            plt.plot(binned_means, binned_vars,'o')
            plt.plot(binned_means, dark_sdev[quadrant]**2**2 + gain_ADU_e*binned_means)
            plt.title('Gain ix: {:d} Speed ix: {:d}'.format(gain, speed))
            plt.axis([0,np.max(binned_means),0,2.5*np.median(binned_vars)])
            plt.xlabel("Flux")
            plt.ylabel("Variance")
            plt.pause(.01)
            #pdb.set_trace()
    

            print("Gain in e/ADU {:5.3f}".format(1/gain_ADU_e))
            gains_e_ADU.append(1/gain_ADU_e)
            print("Readout noise in e: {:5.3f}".format(dark_sdev[quadrant]/gain_ADU_e))
            rnoise_e.append(dark_sdev[quadrant]/gain_ADU_e)
        gfile.write("{:d}, {:d}, {:5.2f}, {:5.2f}, {:5.2f}, {:5.2f}\n".format(speed, gain, 
            gains_e_ADU[0], gains_e_ADU[1], gains_e_ADU[2], gains_e_ADU[3]))
        rfile.write("{:d}, {:d}, {:5.2f}, {:5.2f}, {:5.2f}, {:5.2f}\n".format(speed, gain, 
            rnoise_e[0], rnoise_e[1], rnoise_e[2], rnoise_e[3]))
gfile.close()
rfile.close()