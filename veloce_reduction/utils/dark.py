from __future__ import division, print_function
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import subtract_overscan as so
import glob
import scipy.ndimage as nd

if False:
    files = glob.glob('/Users/mireland/data/veloce/180815/ccd_3/15aug3022?.fits')
    darks_sso = []
    for f in files:
        print(f)
        darks_sso.append(so.read_and_overscan_correct(f))
    darks_sso = np.array(darks_sso)
  
if False:   
    files = glob.glob('/Users/mireland/data/veloce/Video_Offsets/*fit')
    biases = []
    for f in files:
        print(f)
        biases.append(so.read_and_overscan_correct(f))
    biases = np.array(biases)

#files = glob.glob('/Users/mireland/data/veloce/Dark_25_8_2018/Bias_Pre_Dark*.fit')
#files = glob.glob('/Users/mireland/data/veloce/Dark_27_7_2018/Bias_Cold_10?.fit')
#files = glob.glob('/Users/mireland/data/veloce/Darks_4_9_2018/bias_*.fit')
#files = glob.glob('/Users/mireland/data/veloce/bias_tests_180908/Rosso000.fits')
#files = glob.glob('/Users/mireland/data/veloce/bias_tests_180908/Rosso000.fits')
files = glob.glob('/Users/mireland/data/veloce/180919/ccd_3/19sep3043[0123456].fits')
files.extend(glob.glob('/Users/mireland/data/veloce/180919/ccd_3/19sep3044[23456789].fits'))
files.extend(glob.glob('/Users/mireland/data/veloce/180919/ccd_3/19sep30450.fits'))

files = glob.glob('/Users/mireland/data/veloce/180918/ccd_3/18sep3022[456789].fits')
files.extend(glob.glob('/Users/mireland/data/veloce/180918/ccd_3/18sep3023[012].fits'))

files = glob.glob('/Users/mireland/data/veloce/180917/ccd_3/17sep3018[1234567].fits')

files = glob.glob('/Users/mireland/data/veloce/180920/ccd_3/20sep3000[123456789].fits')

files = glob.glob('/Users/mireland/data/veloce/180920/ccd_3/20sep3015[89].fits')
files.extend(glob.glob('/Users/mireland/data/veloce/180920/ccd_3/20sep3016[0123456].fits'))

dark_biases = []
for f in files:
    print(f)
    dark_biases.append(so.read_and_overscan_correct(f))
dark_biases = np.array(dark_biases)

#files = glob.glob('/Users/mireland/data/veloce/Dark_25_8_2018/Dark_2_Hour*.fit')
#files = glob.glob('/Users/mireland/data/veloce/Dark_27_7_2018/Exp_1h_Cold_10[45].fit')
#files = glob.glob('/Users/mireland/data/veloce/Darks_4_9_2018/bs_dark_2hr_*.fit')
#files = glob.glob('/Users/mireland/data/veloce/Darks_4_9_2018/Dark_2_hour*.fit')
#files = glob.glob('/Users/mireland/data/veloce/bias_tests_180908/Rosso002.fits')
files = glob.glob('/Users/mireland/data/veloce/180919/ccd_3/19sep3043[789].fits')
files.extend(glob.glob('/Users/mireland/data/veloce/180919/ccd_3/19sep3044[01].fits'))

files = glob.glob('/Users/mireland/data/veloce/180918/ccd_3/18sep30219.fits')
files.extend(glob.glob('/Users/mireland/data/veloce/180918/ccd_3/18sep3022[0123].fits'))

files = glob.glob('/Users/mireland/data/veloce/180917/ccd_3/17sep3017[6789].fits')
files.extend(glob.glob('/Users/mireland/data/veloce/180917/ccd_3/18sep30180.fits'))

files = glob.glob('/Users/mireland/data/veloce/180920/ccd_3/20sep30010.fits')

files = glob.glob('/Users/mireland/data/veloce/180920/ccd_3/20sep3015[567].fits')

darks = []
for f in files:
    print(f)
    darks.append(so.read_and_overscan_correct(f))
darks = np.array(darks)

mn_bias = np.mean(dark_biases, axis=0)
for f, d in zip(files,darks):
    print("Dark current (ADU/frame) for file {:s}: {:6.2f}".format(f, np.median(d - mn_bias)))
   
meddark = np.median(darks, axis=0) - np.median(dark_biases, axis=0)
plt.clf()
plt.imshow(nd.median_filter(meddark,7), vmin=0, vmax=3.6*2)
plt.title('Filtered median of 5 x 30min darks')
plt.title('Single 2 hour dark')
plt.title('Median of 3 x 1 hour dark')
plt.colorbar()
plt.tight_layout()