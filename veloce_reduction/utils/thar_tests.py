from lmfit.models import LorentzianModel
from bokeh.colors import black



thflux = pyfits.getdata('/Users/christoph/data/commissioning/20180814/14aug30044_extracted.fits',0)
#thflux2 = pyfits.getdata('/Users/christoph/data/commissioning/20180814/14aug30064_extracted.fits',0)
thflux2 = pyfits.getdata('/Users/christoph/data/commissioning/20180916/working/16sep30009_extracted.fits',0)
# thflux = pyfits.getdata('/Volumes/BERGRAID/data/veloce/second_commissioning/180916/working/16sep30009_extracted.fits',0)


now = datetime.datetime.now()
outfn = '/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/thar_lines_as_of_'+str(now)[:10]+'.dat'
outfile = open(outfn, 'w')


linenum, ordnum, m, pix, wl, model_wl, res = readcol('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/5th_order_'+
                                                   'clipped_lines_used_in_fit_as_of_2018-08-16.dat',
                                                   twod=False,skipline=2)
#H-alpha:
#new_ordnum = 29



#METHOD 1 - find new line positions by cross-correlation
shift = []
wshift = []

for new_ordnum in np.arange(2,40):

    print('order ',new_ordnum)

    raw_data = thflux[new_ordnum - 1, :]
    raw_data2 = thflux2[new_ordnum - 1, :]

    data = quick_bg_fix(raw_data)
    data2 = quick_bg_fix(raw_data2)

    scale = 300
    wdata = np.arcsinh(data / scale)
    wdata2 = np.arcsinh(data2 / scale)

    xc = np.correlate(data,data2, mode='same')
    wxc = np.correlate(wdata, wdata2, mode='same')

    xrange = np.arange(np.argmax(xc)-6, np.argmax(xc)+6+1, 1)

    #fit Gaussian to central part of CCF
    mod = GaussianModel()
    mod2 = LorentzianModel()
    #mod2 = Model(fibmodel_with_amp)
    parms = mod.guess(xc[xrange],x=xrange)
    result = mod.fit(xc[xrange],parms,x=xrange,weights=None)
    wparms = mod.guess(wxc[xrange], x=xrange)
    wresult = mod.fit(wxc[xrange], parms, x=xrange, weights=None)

    # #plot it?
    # plot_osf = 10
    # plot_os_grid = np.linspace(xrange[0],xrange[-1],plot_osf * (len(xrange)-1) + 1)
    # guessmodel = mod.eval(result.init_params,x=plot_os_grid)
    # bestmodel = mod.eval(result.params,x=plot_os_grid)
    # wguessmodel = mod.eval(wresult.init_params,x=plot_os_grid)
    # wbestmodel = mod.eval(wresult.params,x=plot_os_grid)
    # plt.legend()

    mu = result.best_values['center']
    wmu = wresult.best_values['center']

    shift.append(mu - (len(xc)//2))
    wshift.append(wmu - (len(wxc) // 2))





#METHOD 2 - find new line positions by checking a small region around each line in the mastertable
lam = np.array([])
x = np.array([])
order = np.array([])
ref_wl = np.array([])
search_region_size = 1
for new_ordnum in np.arange(2,40):

    print('order ', new_ordnum)

    #fix fails
    if new_ordnum == 17:
        thresh = 2000
        bgthresh = 1500
    elif new_ordnum == 25:
        thresh = 4000
        bgthresh = 2000
    else:
        thresh = 1500
        bgthresh = 1000

    mord = 64 + new_ordnum
    ix = np.argwhere(m==mord).flatten()
    ord_pix = pix[ix]
    ord_wl = wl[ix]

    raw_data = thflux[new_ordnum - 1, :]
    raw_data2 = thflux2[new_ordnum - 1, :]

    data = quick_bg_fix(raw_data)
    data2 = quick_bg_fix(raw_data2)

    # fix more fails, ie mask out regions around super bright / saturated lines
    # TODO: make this a predefined set of masks, so that the if statement does not have to be evaluated all the time
    if new_ordnum == 2:
        mask = np.zeros(len(data), dtype='bool')
        mask[3040:3110] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 3:
        mask = np.zeros(len(data),dtype='bool')
        mask[2300:2600] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 8:
        mask = np.zeros(len(data),dtype='bool')
        mask[1670:1860] = True
        mask[3560:3660] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 9:
        mask = np.zeros(len(data),dtype='bool')
        mask[1180:1420] = True
        mask[1560:1850] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 10:
        mask = np.zeros(len(data),dtype='bool')
        mask[2270:2430] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 11:
        mask = np.zeros(len(data),dtype='bool')
        mask[3000:3400] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 12:
        mask = np.zeros(len(data),dtype='bool')
        mask[540:1100] = True
        mask[2930:3220] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 13:
        mask = np.zeros(len(data),dtype='bool')
        mask[470:830] = True
        mask[2130:2330] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 15:
        mask = np.zeros(len(data), dtype='bool')
        mask[2630:2810] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 16:
        mask = np.zeros(len(data), dtype='bool')
        mask[2400:2800] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 17:
        mask = np.zeros(len(data),dtype='bool')
        mask[3060:3160] = True
        mask[3260:3370] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 18:
        mask = np.zeros(len(data),dtype='bool')
        mask[800:960] = True
        mask[1090:1320] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 19:
        mask = np.zeros(len(data), dtype='bool')
        mask[1850:2080] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 23:
        mask = np.zeros(len(data), dtype='bool')
        mask[1325:1400] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 24:
        mask = np.zeros(len(data), dtype='bool')
        mask[1900:2010] = True
        data[mask] = 0
        data2[mask] = 0


    fitted_line_pos2 = fit_emission_lines(data2, return_all_pars=False, varbeta=False, timit=False, verbose=False,
                                          thresh=thresh, bgthresh=bgthresh)

    for j,checkval in enumerate(fitted_line_pos2):
        print(j, checkval)
        q = np.logical_and(ord_pix > checkval - search_region_size, ord_pix < checkval + search_region_size)
        found = np.sum(q)
        if found > 0:
            if found == 1:
                print('OK')
                lam = np.append(lam, ord_wl[q])
                x = np.append(x, ord_pix[q])
                order = np.append(order, new_ordnum)
                ref_wl = np.append(ref_wl, ord_wl[np.argwhere(q==True)])
            else:
                print('ERROR')





def quick_bg_fix(raw_data, npix=4112):
    left_xx = np.arange(npix/2)
    right_xx = np.arange(npix/2, npix)
    left_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[:npix/2],3), size=100)
    right_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[npix/2:],3), size=100)
    data = raw_data.copy()
    data[left_xx] = raw_data[left_xx] - left_bg
    data[right_xx] = raw_data[right_xx] - right_bg
    return data




#re-normalize arrays to [-1,+1]
x_norm = (x / ((len(data)-1)/2.)) - 1.
order_norm = ((order-2) / (37./2.)) - 1.       #TEMP, TODO, FUGANDA, PLEASE FIX ME!!!!!
#order_norm = ((m-1) / ((len(P_id)-1)/2.)) - 1.

#call the fitting routine
p = fit_poly_surface_2D(x_norm, order_norm, ref_wl, weights=None, polytype='chebyshev', poly_deg=5, debug_level=0)

#if return_full:
xx = np.arange(4112)
xxn = (xx / ((len(xx) - 1) / 2.)) - 1.
oo = np.arange(1, len(thflux))
oon = ((oo - 1) / (38. / 2.)) - 1.  # TEMP, TODO, FUGANDA, PLEASE FIX ME!!!!!
# oon = ((oo-1) / ((len(thflux)-1)/2.)) - 1.
X, O = np.meshgrid(xxn, oon)
p_wl = p(X, O)




#########################################################################################



#METHOD 3 - combining above methods to find lines after optics re-alignment
shift = []
wshift = []
lam = np.array([])
x = np.array([])
order = np.array([])
ref_wl = np.array([])
search_region_size = 2.5

for new_ordnum in np.arange(2,40):

    print('order ',new_ordnum)

    mord = 64 + new_ordnum

    ix = np.argwhere(m == mord).flatten()
    ord_pix = pix[ix]
    ord_wl = wl[ix]

    raw_data = thflux[new_ordnum - 1, :]
    raw_data2 = thflux2[new_ordnum - 1, :]

    data = quick_bg_fix(raw_data)
    data2 = quick_bg_fix(raw_data2)

    scale = 300
    wdata = np.arcsinh(data / scale)
    wdata2 = np.arcsinh(data2 / scale)

    xc = np.correlate(data,data2, mode='same')
    wxc = np.correlate(wdata, wdata2, mode='same')

    xrange = np.arange(np.argmax(xc)-6, np.argmax(xc)+6+1, 1)

    #fit Gaussisn to central part of CCF
    mod = GaussianModel()
    parms = mod.guess(xc[xrange],x=xrange)
    result = mod.fit(xc[xrange],parms,x=xrange,weights=None)
    wparms = mod.guess(wxc[xrange], x=xrange)
    wresult = mod.fit(wxc[xrange], parms, x=xrange, weights=None)

    # #plot it?
    # plot_osf = 10
    # plot_os_grid = np.linspace(xrange[0],xrange[-1],plot_osf * (len(xrange)-1) + 1)
    # guessmodel = mod.eval(result.init_params,x=plot_os_grid)
    # bestmodel = mod.eval(result.params,x=plot_os_grid)
    # plt.legend()

    mu = result.best_values['center']
    wmu = wresult.best_values['center']

    #shift.append(mu - (len(xc)//2))
    #wshift.append(wmu - (len(wxc) // 2))
    shift = mu - (len(xc) // 2)
    wshift = wmu - (len(wxc) // 2)

    # fix fails
    if new_ordnum == 17:
        thresh = 2000
        bgthresh = 1500
    elif new_ordnum == 25:
        thresh = 4000
        bgthresh = 2000
    else:
        thresh = 1500
        bgthresh = 1000

    # fix more fails, ie mask out regions around super bright / saturated lines
    # TODO: make this a predefined set of masks, so that the if statement does not have to be evaluated all the time
    if new_ordnum == 2:
        mask = np.zeros(len(data), dtype='bool')
        mask[3040:3110] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 3:
        mask = np.zeros(len(data), dtype='bool')
        mask[2300:2600] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 8:
        mask = np.zeros(len(data), dtype='bool')
        mask[1670:1860] = True
        mask[3560:3660] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 9:
        mask = np.zeros(len(data), dtype='bool')
        mask[1180:1420] = True
        mask[1560:1850] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 10:
        mask = np.zeros(len(data), dtype='bool')
        mask[2270:2430] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 11:
        mask = np.zeros(len(data), dtype='bool')
        mask[3000:3400] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 12:
        mask = np.zeros(len(data), dtype='bool')
        mask[540:1100] = True
        mask[2930:3220] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 13:
        mask = np.zeros(len(data), dtype='bool')
        mask[470:830] = True
        mask[2130:2330] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 15:
        mask = np.zeros(len(data), dtype='bool')
        mask[2630:2810] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 16:
        mask = np.zeros(len(data), dtype='bool')
        mask[2400:2800] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 17:
        mask = np.zeros(len(data), dtype='bool')
        mask[3060:3160] = True
        mask[3260:3370] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 18:
        mask = np.zeros(len(data), dtype='bool')
        mask[800:960] = True
        mask[1090:1320] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 19:
        mask = np.zeros(len(data), dtype='bool')
        mask[1850:2080] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 23:
        mask = np.zeros(len(data), dtype='bool')
        mask[1325:1400] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 24:
        mask = np.zeros(len(data), dtype='bool')
        mask[1900:2010] = True
        data[mask] = 0
        data2[mask] = 0

    fitted_line_pos2 = fit_emission_lines(data2, return_all_pars=False, varbeta=False, timit=False, verbose=False,
                                          thresh=thresh, bgthresh=bgthresh)

    for i,checkval in enumerate(fitted_line_pos2):
        #print(i)
        q = np.logical_and(ord_pix > checkval + wshift - search_region_size, ord_pix < checkval + wshift + search_region_size)
        found = np.sum(q)
        if found > 0:
            if found == 1:
                print('OK')
                lam = np.append(lam, ord_wl[q])
                #x = np.append(x, ord_pix[q])
                x = np.append(x, checkval)
                order = np.append(order, new_ordnum)
                ref_wl = np.append(ref_wl, ord_wl[np.argwhere(q==True)])
            else:
                print('ERROR')

    # now enter ordix manually
    xx = np.arange(4112)
    x = fitted_line_pos2[ordix]
    lam = ord_wl
    order_fit = np.poly1d(np.polyfit(x, lam, 2))
    plt.scatter(x, lam, color='r')
    plt.plot(xx, order_fit(xx))

    outfile = open(outfn, 'a')

    for i, (xpos, wlpos) in enumerate(zip(x, lam)):
        outfile.write("   %3d             %2d                 %3d           %11.6f     %11.6f\n" % (i + 1, new_ordnum, mord, xpos, wlpos))


    outfile.close()
    
    
    
    
####################################################################################################################################
    
    

# ordix = np.argwhere(m==66).T[0]

type = 'gaussian_offset_slope'

plot_osf = 10
plot_os_grid = np.linspace(xrange[0],xrange[-1],plot_osf * (len(xrange)-1) + 1)
plt.plot(data,'b.-')
plt.plot(xx[xrange],data[xrange],'y.-')
plt.xlim(int(np.round(xguess, 0)) - 7, int(np.round(xguess, 0)) + 7)
plt.ylim(-50,200)
plt.plot(plot_os_grid, gaussian_with_offset_and_slope(plot_os_grid, *guess),'r--', label='initial guess')
plt.plot(plot_os_grid, gaussian_with_offset_and_slope(plot_os_grid, *popt),'g-', label='best fit')
plt.axvline(popt[0], color='g', linestyle=':')
plt.legend()
plt.title(type)
plt.xlabel('pixel')
plt.ylabel('counts')
plt.text(3292, 130, 'mu = '+str(np.round(popt[0],4)))
plt.savefig(path + 'linefit_'+type+'.eps')






####################################################################################################################################
path = '/Users/christoph/OneDrive - UNSW/dispsol_tests/20180917/'
indfib_thflux_so = pyfits.getdata(path + 'ARC_ThAr_17sep30080_optimal3a_extracted_with_slope_and_offset.fits')
fibname = ['S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09', '03', '08', '19', '02', 'S4', 'S3', 'S1']
for i in range(24):
    print('fibre i=',i)
    thflux = indfib_thflux_so[:,i,:]
    p_air_wl, p_vac_wl = get_dispsol_from_known_lines(thflux, fibre=fibname[i], fitwidth=4, satmask=None, lamptype='thar', return_all_pars=False, 
                                                      deg_spectral=7, deg_spatial=7, polytype='chebyshev', return_full=True, savetable=True, outpath=path, debug_level=1, timit=False)
    pyfits.writeto(path + 'veloce_thar_dispsol_from_17sep30080_fib'+fibname[i]+'_air.fits', p_air_wl)
    pyfits.writeto(path + 'veloce_thar_dispsol_from_17sep30080_fib'+fibname[i]+'_vac.fits', p_vac_wl)
####################################################################################################################################

####################################################################################################################################
# difference between 2-dim wl-solutions of individual fibres
central_wl = pyfits.getdata(path + 'veloce_thar_dispsol_from_17sep30080_fib'+fibname[11]+'_air.fits')     
wl_near_LFC = pyfits.getdata(path + 'veloce_thar_dispsol_from_17sep30080_fib'+fibname[23]+'_air.fits')         
        
for i in range(24):
    filename = path + 'veloce_thar_dispsol_from_17sep30080_fib'+fibname[i]+'_air.fits'
    wl = pyfits.getdata(filename)
    if i !=11:
        plt.plot(wl[38,:] - wl_near_LFC[38,:], label='fibre '+fibname[i])
    else:
        plt.plot(wl[38,:] - wl_near_LFC[38,:], 'k', label='fibre '+fibname[i])
####################################################################################################################################
    
####################################################################################################################################
# difference between 1-dim wl-solutions of individual fibres
xx = np.arange(4112)
ref_filename = path + 'thar_lines_fibre_' + fibname[0] + '_as_of_2018-10-30.dat'
linenum, order, m, pix, wlref, vac_wlref = readcol(ref_filename, twod=False, skipline=2)
ix = np.argwhere(order == 21).flatten()
x = pix[ix]
lam = wlref[ix]
order_fit = np.poly1d(np.polyfit(x, lam, 2))
ref_fit = order_fit(xx)
for i in range(24):
    filename = path + 'thar_lines_fibre_' + fibname[i] + '_as_of_2018-10-30.dat'
    linenum, order, m, pix, wlref, vac_wlref = readcol(filename, twod=False, skipline=2)
    ix = np.argwhere(order == 21).flatten()
    x = pix[ix]
    lam = wlref[ix]
    order_fit = np.poly1d(np.polyfit(x, lam, 2))
    plt.plot(xx, order_fit(xx) - ref_fit, label='fibre '+fibname[i])
####################################################################################################################################

####################################################################################################################################
# difference in pixel space (ie peak locations)     produces the 2nd plot in "Translating the wavelength solution from fibre to fibre.docx"
xx = np.arange(4112)
ref_filename = path + 'thar_lines_fibre_' + fibname[0] + '_as_of_2018-10-30.dat'
linenum, order, m, pix, wlref, vac_wlref = readcol(ref_filename, twod=False, skipline=2)
ix = np.argwhere(order == 3).flatten()
ref_x = pix[ix]
ref_wlref = wlref[ix]
order_fit = np.poly1d(np.polyfit(ref_x, ref_wlref, 5))
ref_fit = order_fit(xx)

pixfit_coeffs = []

for i in range(24):
    filename = path + 'thar_lines_fibre_' + fibname[i] + '_as_of_2018-10-30.dat'
    linenum, order, m, pix, wlref, vac_wlref = readcol(filename, twod=False, skipline=2)
    ix = np.argwhere(order == 3).flatten()
    x = pix[ix]
    lam = wlref[ix]
    matched_ref_x = ref_x[np.in1d(ref_wlref, lam)]
    delta_x = x - matched_ref_x
#     plt.plot(matched_ref_x, x - matched_ref_x, 'x-', label='fibre '+fibname[i])
    # perform sanity-check 1D fit (remove only very obvious outliers, as there is quite a bit of scatter)
    test_fit = np.poly1d(np.polyfit(matched_ref_x, delta_x, 1))
    if debug_level >= 2:
        plt.figure()
        plt.plot(matched_ref_x, delta_x, 'x-')
        plt.title('fibre '+fibname[i])
        plt.plot(xx, test_fit(xx))
        plt.xlim(0,4111)
    # remove obvious outliers
    fitres = delta_x - test_fit(matched_ref_x)
    # do single sigma clipping
    goodres,goodix,badix = single_sigma_clip(fitres, 2, return_indices=True)
    #fit again
    pix_fit = np.poly1d(np.polyfit(matched_ref_x[goodix], delta_x[goodix], 1))
    pixfit_coeffs.append(pix_fit)
    #now fit new dispsol to the model-shifted lines
    order_fit = np.poly1d(np.polyfit(ref_x + pix_fit(ref_x), ref_wlref, 5))
    plt.plot(xx, order_fit(xx) - ref_fit, label='fibre '+fibname[i])
    
    
   
####################################################################################################################################

    
#######

# finding ThAr / ThXe regions to mask out
thxespec_fn = '/Volumes/BERGRAID/data/veloce/reduced/20181115/ARC - ThAr_15nov30003_optimal3a_extracted.fits'
thxespec = pyfits.getdata(thxespec_fn)
thxeflux = np.sum(thxespec, axis=1)

tharspec_fn = '/Users/christoph/OneDrive - UNSW/dispsol_tests/20180917/ARC_ThAr_17sep30080_optimal3a_extracted_with_slope_and_offset.fits'
tharspec = pyfits.getdata(tharspec_fn)
tharflux = np.sum(tharspec, axis=1)

linenum, order, m, pix, wlref, vac_wlref, _, _, _, _ = readcol('/Users/christoph/OneDrive - UNSW/linelists/thar_lines_used_in_7x7_fit_as_of_2018-10-19.dat', twod=False, skipline=2)

i=0

i += 1

mord = (i+1) + 64
ix = np.argwhere(m == mord).flatten()
ord_pix = pix[ix]

plt.plot(tharflux[i,:], 'r', label='ThAr')
plt.plot(thxeflux[i,:], 'b', label='ThXe')
plt.title('Order_'+str(i+1).zfill(2) + '   (i='+str(i)+')')
plt.legend()
for p in ord_pix:
    plt.axvline(p, color='gray', linestyle=':')
    
#######
    
    

    
    
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    