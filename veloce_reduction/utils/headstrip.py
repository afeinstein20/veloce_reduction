"""Create a useful csv file from the headers, and (when possible) from the pixel data"""
from __future__ import print_function
import astropy.io.fits as pyfits
import glob
import os

fnames = glob.glob('/Users/mireland/data/veloce/180920/ccd_3/[0123]*fits')
fnames.sort()
for f in fnames:
    header = pyfits.getheader(f)
    print(str(header['RUN']) + ', ' + header['OBJECT'] + ', ' + header['OBSTYPE'] + ', '+ str(header['NAXIS1']) + ', ' +  str(header['NAXIS1']))
    