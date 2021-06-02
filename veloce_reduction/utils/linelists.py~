'''
Created on 15 Jun. 2018

@author: christoph
'''

import numpy as np

from veloce_reduction.veloce_reduction.helper_functions import find_nearest, CMB_pure_gaussian





def make_LFC_linelist(delta_f=25., wlmin=550., wlmax=950., shift=0., savefile=True, outpath = '/Users/christoph/OneDrive - UNSW/linelists/'):
    """
    PURPOSE:
    make fake laser-comb line-list for JS's Echelle++
    
    INPUT:
    'delta_f'  : line spacing in GHz
    'wlmin'    : minimum wavelength in nm
    'wlmax'    : maximum wavelength in nm
    'shift'    : apply this RV shift (negative for blueshift)
    'savefile' : boolean - do you want to save the linelist to a file?
    'outpath'  : directory for outpuf file 
    
    OUTPUT:
    'wl'     : wl of lines in microns
    'relint' : relative intensities (should be all equal for the LFC)
    
    MODHIST:
    15/06/2018 - CMB create
    """
    
    veloce_grating_constant = 61.4   # m*lambda_c in microns for Veloce (only approximately)
    
    if shift % np.floor(shift) != 0:
        print('WARNING: non-integer RV shift provided!!! It will be rounded to the nearest integer [in m/s]!')
        shift = np.round(shift,0)
        
    c = 2.99792458E8    #speed of light in m/s
    #convert delta_f to Hertz
    delta_f *= 1e9    
    #min and max frequency in Hz
    fmin = c / (wlmax*1e-9)         
    fmax = c / (wlmin*1e-9)

    #all the frequencies
    f0 = np.arange(fmin,fmax,delta_f)
    
    #calculate the "Doppler" shift (it really is just an offset in pixel space)
    fshift = fmin * (shift/c)
    
    #apply shift to frequencies (WARNING: don't Doppler-shift all frequencies, as this is not actually a Doppler shift, but is supposed to simulate a shift in pixels)
    f = f0 - fshift   #the minus sign is because the shift is applied to frequencies rather than to wavelengths

    #convert to the wavelengths
    wl = np.flip((c / f) / 1e-6, axis=0)     #wavelength in microns (from shortest to longest wavelength)
    relint = [1]*len(wl)

    #make one line near the order centres a different intensity so that it is easier to identify the lines
    for o in np.arange(1,44):
        m = 65 + o
        ordcen = veloce_grating_constant / m
        ordcen_ix = find_nearest(wl,ordcen,return_index=True)
        relint[ordcen_ix] = 2.
    
    if savefile:
        #string manipulations for the output file name
        if shift > 0:
            redblue = '_red'
        elif shift < 0:
            redblue = '_blue'
        else:
            redblue = ''
        np.savetxt(outpath + 'laser_linelist_25GHz'+redblue+'_'+str(int(shift))+'ms.dat', np.c_[wl, relint], delimiter=';', fmt='%12.10f; %i')
        return
    else:
        return wl, relint
    
    
    
    
    
def make_binmask_from_linelist(refwl, relint=None, step=0.01):
    
    ### input could be sth like this:
#     from readcol import readcol
#     #read in master line list
#     linelist = readcol('/Users/christoph/OneDrive - UNSW/linelists/thar_mm.arc', twod=False, skipline=8)
#     refwl = linelist[0]

    if relint is None:
        relint = np.ones(len(refwl))

    refgrid = np.arange(np.floor(np.min(refwl)), np.ceil(np.max(refwl)), step)
    refspec = np.zeros(len(refgrid))
    for i,pos in enumerate(refwl):
        #print('Line '+str(i+1)+'/'+str(len(refwl)))
        ix = find_nearest(refgrid, pos, return_index=True)
        #print('ix = ',ix)
        refspec[np.max([0,ix-1]) : np.min([ix+2,len(refspec)])] = relint[i]
        
    return refgrid,refspec





def make_gaussmask_from_linelist(refwl, relint=None, step=0.01, gauss_sigma=1):
    
    #sigma in units of steps
    #steps should be adjusted depending on np.min(np.diff(refwl)), and spectrograph resolving power
    
    ### input could be sth like this:
#     from readcol import readcol
#     #read in master line list
#     linelist = readcol('/Users/christoph/OneDrive - UNSW/linelists/thar_mm.arc', twod=False, skipline=8)
#     refwl = linelist[0]
#     relint = 10**linelist[2]

    if relint is None:
        relint = np.ones(len(refwl))

    refgrid = np.arange(np.floor(np.min(refwl)), np.ceil(np.max(refwl)), step)
    refspec = np.zeros(len(refgrid))
    for i,pos in enumerate(refwl):
        #print('Line '+str(i+1)+'/'+str(len(refwl)))
        ix = find_nearest(refgrid, pos, return_index=True)
        #print('ix = ',ix)
        if relint is None:
            amp = 1
        else:
            amp = relint[i]
        peak = CMB_pure_gaussian(refgrid[ix-20:ix+21], pos, gauss_sigma * step, amp)
        refspec[ix-20:ix+21] = refspec[ix-20:ix+21] + peak * relint[i]
        
    return refgrid,refspec


# step = 0.01
# refgrid,refspec = make_gaussmask_from_linelist(refwl, relint=relint, step=step)
# lam_found = lam[x.astype(int)]
# grid,spec = make_gaussmask_from_linelist(lam_found, step=step)
# rebinned_refspec = np.interp(grid,refgrid,refspec)
# xc = np.correlate(spec,rebinned_refspec, mode='same')
# #peakix = find maximum...
# shift = (peakix - len(spec)//2) * step   #that's how many Angstroms the detected peaks are shifted wrt the "master" line list



    