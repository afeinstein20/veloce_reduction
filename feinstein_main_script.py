import os
import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from veloce_reduction.calibration import *
from veloce_reduction.get_info_from_headers import *
from veloce_reduction.process_scripts import *
from veloce_reduction.order_tracing import *
from veloce_reduction.spatial_profiles import *
from veloce_reduction.wavelength_solution import *
from veloce_reduction.chipmasks import make_chipmask
from veloce_reduction.process_scripts import *

date = str(sys.argv[1])

path = '/Users/arcticfox/Documents/youngStars/veloce/spectra/fitsfiles/'

fitsfiles = np.sort([os.path.join(path, i) for i in os.listdir(path)
                     if i.endswith('.fits')])
i = 100
hdu = fits.open(fitsfiles[i])
data = hdu[0].data
hdu[0].header['OBJECT']
nx, ny = data.shape[0], data.shape[1]

hdu.close()

gain = [0.88,0.93,0.99,0.93]

print('got file names')
acq_list, bias_list, dark_list, flat_list, arc_list, simth_only_list, laser_only_list, laser_and_simth_list, stellar_list, unknown_list = get_obstype_lists(path)

file_lists = {'acq':acq_list, 'bias':bias_list, 
              'dark':dark_list, 'flat':flat_list, 
              'arc':arc_list, 'simth_only':simth_only_list, 
              'laser_only':laser_only_list, 
              'laser_and_simth':laser_and_simth_list, 
              'stellar':stellar_list, 
              'unknown':unknown_list}

### PULLS FILES ON THE SAME DATE ###

files_on_date = {}
for key in list(file_lists.keys()):
    files_on_date[key] = np.sort([i for i in file_lists[key] if
                                  date in i])

### MED BIAS FRAME
print('making median bias frame...')
medbias, coeffs, offsets, rons = get_bias_and_readnoise_from_bias_frames(files_on_date['bias'],
                                                                         degpol=5,
                                                                         clip=5,
                                                                         gain=[0.88,0.93,0.99,0.93],
                                                                         save_medimg=True,
                                                                         path=path,
                                                                         date=date)
offmask,ronmask = make_offmask_and_ronmask(offsets, rons, nx, 
                                           ny, gain=gain, savefiles=True, 
                                           path=path, timit=True)
np.save(os.path.join(path, '{0}_ronmask.npy'.format(date)), ronmask)

MB = make_master_bias_from_coeffs(coeffs, nx, ny, savefile=True, 
                                  path=path, timit=True, date=date)

### MASTER DARK 
print('making master dark frame...')
MD = make_master_dark(files_on_date['dark'], medbias, gain=gain, 
                      savefile=True, path=path,
                      date=date)

### MASTER WHITE FRAME
print('making master white frame...')
MW,err_MW = process_whites(files_on_date['flat'], 
                           MB=crop_overscan_region(MB), 
                           ronmask=crop_overscan_region(ronmask), 
                           MD=MD, date=date,
                           gain=gain, scalable=False, 
                           fancy=False, clip=5., savefile=True, 
                           saveall=False, diffimg=False, path=path, timit=False,
                           remove_bg=False)

### TRACING ORDERS & MAKING ORDER MASK
MW = np.nanmedian(MW, axis=0)
err_MW = np.nanmedian(err_MW, axis=0)

P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, 
                          gauss_filter_sigma=3., simu=False)

P_id = make_P_id(P)
mask = make_mask_dict(tempmask)

print('making stripes...')
MW_stripes, MW_stripe_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=30)
err_MW_stripes = extract_stripes(err_MW, P_id, return_indices=False, slit_height=30)

np.save(os.path.join(path, '{0}_stripe_masks.npy'.format(date)), MW_stripe_indices)
