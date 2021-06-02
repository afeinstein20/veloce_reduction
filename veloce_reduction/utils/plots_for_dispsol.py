'''
Created on 19 Mar. 2018

@author: christoph
'''

import numpy as np
from readcol import readcol
import matplotlib.pyplot as plt

from veloce_reduction.helper_functions import find_nearest
from veloce_reduction.wavelength_solution import find_suitable_peaks, fit_emission_lines


# thname = '/Users/christoph/UNSW/veloce_spectra/Mar02/clean_thorium-17-12.fits'
# wname = '/Users/christoph/UNSW/veloce_spectra/Mar01/master_white_int.fits'
# flat = pyfits.getdata(wname).T + 1.
# thimg = pyfits.getdata(thname)
# thflux = np.load('/Users/christoph/UNSW/veloce_spectra/reduced/tests/thorium-17-12_40orders.npy').item()
# thflux2 = np.load('/Users/christoph/UNSW/veloce_spectra/reduced/tests/thorium-17-12_collapsed_40orders.npy').item()
#
#
# P,tempmask = find_stripes(flat, deg_polynomial=2,min_peak=0.05,debug_level=3)
# P_id = make_P_id(P)
# mask = make_mask_dict(tempmask)
# # collapsed_mask = np.zeros(39,dtype='bool')
# # collapsed_mask[0] = True
# # collapsed_mask[25:] = True



def make_veloce_dispsol_from_thar(img, P_id, degpol=5, saveplots=False, return_all_pars=False):
    """ img = thorium image (already rotated so that orders are horizontal """
    
    thresholds = np.load('/Users/christoph/UNSW/linelists/AAT_folder/thresholds.npy').item()
    
    #wavelength solution from Zemax as a reference
    zemax_dispsol = np.load('/Users/christoph/UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()   
    
    dispsol = {}
    if return_all_pars:
        fitshapes = {}

    #prepare dictionary for reference wavelengths
    #wl_ref = {}

    for ord in sorted(P_id.keys())[:-1]:    #don't have anough lines in order 40
        ordnum = ord[-2:]
        m = 105 - int(ordnum)
        print('OK, fitting '+ord+'   (m = '+str(m)+')')
        coll = thresholds['collapsed'][ord]
        if coll:
            data = thflux2[ord]
        else:
            data = thflux[ord]
        
        xx = np.arange(len(data))
        
        if return_all_pars:
            fitted_line_pos,fitted_line_sigma,fitted_line_amp = fit_emission_lines(data,return_all_pars=return_all_pars,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
        else:
            fitted_line_pos = fit_emission_lines(data,return_all_pars=return_all_pars,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
        goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])    
        
        line_number, refwlord = readcol('/Users/christoph/UNSW/linelists/AAT_folder/ThAr_linelist_order_'+ordnum+'.dat',fsep=';',twod=False)
        lam = refwlord.copy()  
        #wl_ref[ord] = lam
        
        mask_order = np.load('/Users/christoph/UNSW/linelists/posmasks/mask_order'+ordnum+'.npy')
        x = fitted_line_pos[mask_order]
        #stupid python!?!?!?
        if ordnum == '30':
            x = np.array([x[0][0],x[1][0],x[2],x[3]])
        
        zemax_wl = 10. * zemax_dispsol['order'+str(m)]['model'](xx[::-1])
        
        #perform the fit
        fitdegpol = degpol
        while fitdegpol > len(x)/2:
            fitdegpol -= 1
        if fitdegpol < 2:
            fitdegpol = 2
        thar_fit = np.poly1d(np.polyfit(x, lam, fitdegpol))
        dispsol[ord] = thar_fit
        if return_all_pars:
            fitshapes[ord] = {}
            fitshapes[ord]['x'] = x
            fitshapes[ord]['y'] = P_id[ord](x)
            fitshapes[ord]['FWHM'] = 2.*np.sqrt(2.*np.log(2.)) * fitted_line_sigma[mask_order]
        
        #calculate RMS of residuals in terms of RV
        resid = thar_fit(x) - lam
        rv_resid = 3e8 * resid / lam
        rms = np.std(rv_resid)
        
        if saveplots:
            #first figure: lambda vs x with fit and zemax dispsol
            fig1 = plt.figure()
            plt.plot(x,lam,'bo')
            plt.plot(xx,thar_fit(xx),'g',label='fitted')
            plt.plot(xx,zemax_wl,'r--',label='Zemax')
            plt.title('Order '+str(m))
            plt.xlabel('pixel number')
            plt.ylabel(ur'wavelength [\u00c5]')
            plt.text(3000,thar_fit(500),'n_lines = '+str(len(x)))
            plt.text(3000,thar_fit(350),'deg_pol = '+str(fitdegpol))
            plt.text(3000,thar_fit(100),'RMS = '+str(round(rms, 1))+' m/s')
            plt.legend()
            plt.savefig('/Users/christoph/UNSW/dispsol/lab_tests/fit_to_order_'+ordnum+'.pdf')
            plt.close(fig1)
            
            #second figure: spectrum vs fitted dispsol
            fig2 = plt.figure()
            plt.plot(thar_fit(xx),data)
            #plt.scatter(thar_fit(x), data[x.astype(int)], marker='o', color='r', s=40)
            plt.scatter(thar_fit(goodpeaks), data[goodpeaks], marker='o', color='r', s=30)
            plt.title('Order '+str(m))
            plt.xlabel(ur'wavelength [\u00c5]')
            plt.ylabel('counts')
            plt.savefig('/Users/christoph/UNSW/dispsol/lab_tests/ThAr_order_'+ordnum+'.pdf')
            plt.close(fig2)
            

    if return_all_pars:
        return dispsol,fitshapes
    else:
        return dispsol
    
    
    
    
def plot_free_spectral_range_wl(P_id, dispsol, degpol=5):
    """ img = thorium image (already rotated so that orders are horizontal """
    
    thresholds = np.load('/Users/christoph/UNSW/linelists/AAT_folder/thresholds.npy').item()
    
    fig1 = plt.figure()
    plt.xlabel('pixel number')
    plt.ylabel(ur'wavelength [\u00c5]')
    plt.title('Veloce Rosso CCD')
    
    #prepare wavelength dictionary
    wavelengths = {}
    
    for ord in sorted(dispsol.keys()):    #don't have anough lines in order 40
        ordnum = ord[-2:]
        m = 105 - int(ordnum)
        print('OK, fitting '+ord+'   (m = '+str(m)+')')
        coll = thresholds['collapsed'][ord]
        if coll:
            data = thflux2[ord]
        else:
            data = thflux[ord]
        
        xx = np.arange(len(data))
        
        fitted_line_pos = fit_emission_lines(data,return_all_pars=return_all_pars,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
        goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])    
        
        line_number, refwlord = readcol('/Users/christoph/UNSW/linelists/AAT_folder/ThAr_linelist_order_'+ordnum+'.dat',fsep=';',twod=False)
        lam = refwlord.copy()  
        
        mask_order = np.load('/Users/christoph/UNSW/linelists/posmasks/mask_order'+ordnum+'.npy')
        x = fitted_line_pos[mask_order]
        #stupid python!?!?!?
        if ordnum == '30':
            x = np.array([x[0][0],x[1][0],x[2],x[3]])
                       
        #perform the fit
        fitdegpol = degpol
        while fitdegpol > len(x)/2:
            fitdegpol -= 1
        if fitdegpol < 2:
            fitdegpol = 2
        thar_fit = np.poly1d(np.polyfit(x, lam, fitdegpol))
        
        wavelengths[ord] = thar_fit(xx)
        
        plt.plot(xx,thar_fit(xx))
        plt.scatter(x,lam,marker='.',color='k')
        

    # add labels to redmost and bluemost orders    
    plt.text(4200,9500,'m')            
    plt.text(4200,5930,'104')
    plt.text(4200,9370,'66')
    
    # calculate free spectral range
    #indices for the free spectral range
    lix = np.zeros(len(wavelengths))  
    lix_wl = np.zeros(len(wavelengths))      
    rix = np.zeros(len(wavelengths))  
    rix_wl = np.zeros(len(wavelengths))     
    for i,ord in enumerate(sorted(wavelengths.keys())):
        ordnum = ord[-2:]
        if i != 0:
            lix[i] = find_nearest(wavelengths[ord],np.max(wavelengths['order_'+str(int(ordnum)-1).zfill(2)]),return_index=True)
            lix_wl[i] = wavelengths[ord][find_nearest(wavelengths[ord],np.max(wavelengths['order_'+str(int(ordnum)-1).zfill(2)]),return_index=True)]
        if i != 38:
            rix[i] = find_nearest(wavelengths[ord],np.min(wavelengths['order_'+str(int(ordnum)+1).zfill(2)]),return_index=True)
            rix_wl[i] = wavelengths[ord][find_nearest(wavelengths[ord],np.min(wavelengths['order_'+str(int(ordnum)+1).zfill(2)]),return_index=True)]
    lix_wl = lix_wl[lix != 0]     #runs from order02 to order39   (39 orders total)
    lix = lix[lix != 0]     #runs from order02 to order39   (39 orders total)
    rix_wl = rix_wl[rix != 0]     #runs from order01 to order38   (39 orders total)
    rix = rix[rix != 0]     #runs from order01 to order38   (39 orders total)

    return





def plot_free_spectral_range(P_id):
    """ img = thorium image (already rotated so that orders are horizontal """
    
    thresholds = np.load('/Users/christoph/UNSW/linelists/AAT_folder/thresholds.npy').item()
    fitshapes = np.load('/Users/christoph/UNSW/dispsol/lab_tests/fitshapes.npy').item()
    thflux = np.load('/Users/christoph/UNSW/veloce_spectra/reduced/tests/thorium-17-12_40orders.npy').item()
    thflux2 = np.load('/Users/christoph/UNSW/veloce_spectra/reduced/tests/thorium-17-12_collapsed_40orders.npy').item()
    wl = np.load('/Users/christoph/UNSW/dispsol/lab_tests/wavelengths.npy').item()
#     #wavelength solution from Zemax as a reference
#     zemax_dispsol = np.load('/Users/christoph/UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()   
#     zemax_wl = {}
    
    xx = np.arange(4112)
    
    fig1 = plt.figure()
    plt.xlabel('pixel number')
    plt.ylabel('pixel number')
    plt.title('Veloce Rosso CCD') 
    
    for ord in sorted(P_id.keys()):    #don't have anough lines in order 40
        ordnum = ord[-2:]
        m = 105 - int(ordnum)
        #zemax_wl[ord] = 10. * zemax_dispsol['order'+str(m)]['model'](xx[::-1])
        
        plt.plot(xx,P_id[ord](xx))
        plt.scatter(fitshapes[ord]['x'],fitshapes[ord]['y'],marker='.',color='k')
    
    #now also plot order 40, for which we do not have a wavelength solution
    plt.plot(xx,P_id['order_40'](xx))
    
    # add labels to redmost and bluemost orders    
    plt.text(-150,4300,'m')            
    plt.text(-150,P_id['order_01'](0) - 35,'104')
    plt.text(-150,P_id['order_11'](0) - 35,'94')
    plt.text(-150,P_id['order_21'](0) - 35,'84')
    plt.text(-150,P_id['order_31'](0) - 35,'74')
    plt.text(-150,P_id['order_40'](0) - 35,'65')
    plt.text(4175,4300,r'$\lambda_c$ '+ur'[\u00c5]')            
    plt.text(4175,P_id['order_01'](4111) - 35,np.round(614000./104.,0).astype(int))
    plt.text(4175,P_id['order_11'](4111) - 35,np.round(614000./94.,0).astype(int))
    plt.text(4175,P_id['order_21'](4111) - 35,np.round(614000./84.,0).astype(int))
    plt.text(4175,P_id['order_31'](4111) - 35,np.round(614000./74.,0).astype(int))
    plt.text(4175,P_id['order_40'](4111) - 35,np.round(614000./65.,0).astype(int))
    
    
    # calculate free spectral range
    #indices for the free spectral range
    lix = np.zeros(len(wl))  
    lix_y = np.zeros(len(wl))      
    rix = np.zeros(len(wl))  
    rix_y = np.zeros(len(wl))     
    zlix = np.zeros(len(wl))  
    zlix_y = np.zeros(len(wl))      
    zrix = np.zeros(len(wl))  
    zrix_y = np.zeros(len(wl))     
    for i,ord in enumerate(sorted(wl.keys())):
        ordnum = ord[-2:]
        if i != 0:
            lix[i] = find_nearest(wl[ord],np.max(wl['order_'+str(int(ordnum)-1).zfill(2)]),return_index=True)
            lix_y[i] = P_id[ord](find_nearest(wl[ord],np.max(wl['order_'+str(int(ordnum)-1).zfill(2)]),return_index=True))
            zlix[i] = find_nearest(zemax_wl[ord],np.max(zemax_wl['order_'+str(int(ordnum)-1).zfill(2)]),return_index=True)
            zlix_y[i] = P_id[ord](find_nearest(zemax_wl[ord],np.max(zemax_wl['order_'+str(int(ordnum)-1).zfill(2)]),return_index=True))
        if i != 38:
            rix[i] = find_nearest(wl[ord],np.min(wl['order_'+str(int(ordnum)+1).zfill(2)]),return_index=True)
            rix_y[i] = P_id[ord](find_nearest(wl[ord],np.min(wl['order_'+str(int(ordnum)+1).zfill(2)]),return_index=True))
            zrix[i] = find_nearest(zemax_wl[ord],np.min(zemax_wl['order_'+str(int(ordnum)+1).zfill(2)]),return_index=True)
            zrix_y[i] = P_id[ord](find_nearest(zemax_wl[ord],np.min(zemax_wl['order_'+str(int(ordnum)+1).zfill(2)]),return_index=True))
    lix_y = lix_y[lix != 0]     #runs from order02 to order39   (39 orders total)
    lix = lix[lix != 0]     #runs from order02 to order39   (39 orders total)
    rix_y = rix_y[rix != 0]     #runs from order01 to order38   (39 orders total)
    rix = rix[rix != 0]     #runs from order01 to order38   (39 orders total)
    zlix_y = zlix_y[zlix != 0]     #runs from order02 to order39   (39 orders total)
    zlix = zlix[zlix != 0]     #runs from order02 to order39   (39 orders total)
    zrix_y = zrix_y[zrix != 0]     #runs from order01 to order38   (39 orders total)
    zrix = zrix[zrix != 0]     #runs from order01 to order38   (39 orders total)

    #fit 2nd order polynomial to the FSR points
    l_fsr_fit = np.poly1d(np.polyfit(lix, lix_y, 2))
    r_fsr_fit = np.poly1d(np.polyfit(rix, rix_y, 2))
    zl_fsr_fit = np.poly1d(np.polyfit(zlix, zlix_y, 2))
    zr_fsr_fit = np.poly1d(np.polyfit(zrix, zrix_y, 2))

    #and overplot this fit
    yl = l_fsr_fit(xx)
    plt.plot(xx,yl,'r--')
    yr = r_fsr_fit(xx)
    plt.plot(xx,yr,'r--')
    plt.xlim(-200,4312)
    plt.ylim(-200,4402)
    ytop = np.repeat(4402,4112)
    ylr = np.maximum(yl,yr)
    plt.fill_between(xx, ytop, ylr, where=ytop>yl, facecolor='gold', alpha=0.25)
    
    return
















