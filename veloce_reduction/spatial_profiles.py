'''
Created on 5 Apr. 2018

@author: christoph
'''

from lmfit import Parameters, Model
from lmfit.models import *
from lmfit.minimizer import *
from scipy import ndimage
import matplotlib.pyplot as plt
import time, sys

from .helper_functions import find_maxima, fibmodel, fibmodel_with_amp, offset_pseudo_gausslike, fibmodel_with_amp_and_offset, norm_fibmodel_with_amp, norm_fibmodel_with_amp_and_offset
from .order_tracing import flatten_single_stripe, flatten_single_stripe_from_indices


__all__ =  ['determine_spatial_profiles_single_order', 'fit_stacked_single_fibre_profile', 
            'fit_single_fibre_profile', 'fit_profiles_single_order', 'fit_profiles', 
            'fit_profiles_from_indices', 'make_model_stripes_gausslike', 'older_routines']



def determine_spatial_profiles_single_order(sc, sr, err_sc, ordpol, ordmask=None, model='gausslike', sampling_size=50, return_stats=False, debug_level=0, timit=False):
    """
    Calculate the spatial-direction profiles of the fibres for a single order.
    
    INPUT:
    'sc'             : the flux in the extracted, flattened stripe
    'sr'             : row-indices (ie in spatial direction) of the cutouts in 'sc'
    'err_sc'         : the error in the extracted, flattened stripe
    'ordpol'         : set of polynomial coefficients from P_id for that order (ie p = P_id[ord])
    'ordmask'        : gives user the option to provide a mask (eg from "find_stripes")
    'model'          : which model do you want to use for the fitting? 
                       ['gaussian', 'gausslike', 'lorentz', 'moffat', 'voigt', 'pseudo(voigt)', 'offset_pseudo', 'pearson7', 'studentst', 'breitwigner', 'lognormal', 'dampedosc', 'dampedharmosc', 'expgauss', 'skewgauss', 'donaich', 'all']
    'sampling_size'  : how many pixels (in dispersion direction) either side of current i-th pixel do you want to consider? 
                       (ie stack profiles for a total of 2*sampling_size+1 pixels...)
    'RON'            : read-out noise per pixel
    'return_stats'   : boolean - do you want to return goodness-of-fit statistics (ie AIC, BIC, CHISQ and REDCHISQ)?
    'debug_level'    : for debugging...
    'timit'          : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'colfits'        : instance of "best_values" from "lmfit" fitting of order profiles
    """
    
    if timit:
        start_time = time.time()
    if debug_level >= 1:
        print('Fitting fibre profiles for one order...') 
    #loop over all columns for one order and do the profile fitting
    #'colfits' is a dictionary, that has npix keys. Each key is an instance of the 'ModelResult'-class from the 'lmfit' package
    colfits = {}
    npix = sc.shape[1]
    
    if ordmask is None:
        ordmask = np.ones(npix, dtype='bool')
    
    
    for i in range(npix):
    #for i in range(2000,2500,1):
        
        if debug_level >= 1:
            print('i = ',str(i))
        
        #add pixel column number to output
        if model.lower() != 'all':
            try:
                colfits['pixnum'].append(i)
            except KeyError:
                colfits['pixnum'] = [i]
        else:
            #model_lib = ['gaussian', 'gausslike', 'lorentz', 'moffat', 'voigt', 'pseudo', 'offset_pseudo', 'pearson7', 'studentst', 'breitwigner', 'lognormal', 'dampedosc', 'dampedharmosc', 'skewgauss', 'donaich']
            model_lib = ['gaussian', 'gausslike', 'moffat', 'pseudo', 'offset_pseudo', 'lognormal', 'skewgauss', 'offset_pseudo_gausslike']
            all_fit_results = {}
        
        #fail-check variable    
        fu = 0
        
        #check if that particular cutout falls fully onto CCD
        checkprod = np.product(sr[1:,i].astype(float))    #exclude the first row number, as that can legitimately be zero
        #NOTE: This also covers row numbers > ny, as in these cases 'sr' is set to zero in "flatten_single_stripe(_from_indices)"
        if ordmask[i]==False:
            fu = 1
            print('WARNING: this pixel column is masked out due to low signal!!!')
        elif checkprod == 0:
            fu = 1
            checksum = np.sum(sr[:,i])
            if checksum == 0:
                print('WARNING: the entire cutout lies outside the chip!!!')
                #best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
            else:
                print('WARNING: parts of the cutout lie outside the chip!!!')
                #best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
        else:    
            #this is the NORMAL case, where the entire cutout lies on the chip
            grid = []
            # data = []
            normdata = []
            # errors = []
            weights = []
            refpos = ordpol(i)
            for j in np.arange(np.max([0,i-sampling_size]),np.min([npix-1,i+sampling_size])+1):
                grid.append(sr[:,j] - ordpol(j) + refpos)
                #data.append(sc[:,j])
                normdata.append(sc[:,j]/np.sum(sc[:,j]))
                #errors.append(np.sqrt(sc[:,j] + RON**2))
                #using relative errors for weights in the fit
                normerr = (err_sc[:,j]/sc[:,j]) / np.sum(sc[:,j])
                pix_w = 1./(normerr*normerr)  
                pix_w[np.isinf(pix_w)] = 0.
                weights.append(pix_w)
                ### initially I thought this was clearly rubbish as it down-weights the central parts
                ### and that we really want to use the relative errors, ie w_i = 1/(relerr_i)**2
                ### HOWEVER: this is not true, and the optimal extraction linalg routine requires absolute errors!!!    
                #weights.append(1./((np.sqrt(sc[:,j] + RON**2)) / sc[:,j])**2)
                if debug_level >= 2:
                    #plt.plot(sr[:,j] - ordpol(j),sc[:,j],'.')
                    plt.plot(sr[:,j] - ordpol(j),sc[:,j]/np.sum(sc[:,j]),'.')
                    #plt.xlim(-5,5)
                    plt.xlim(-sc.shape[0]/2,sc.shape[0]/2)
                
            # data = np.array(data)
            normdata = np.array(normdata)
            weights = np.array(weights)
            grid = np.array(grid)
            # data = data[grid.argsort()]
            normdata = normdata[grid.argsort()]
            weights = weights[grid.argsort()]
            grid = grid[grid.argsort()]
            
            #adjust to flux level in actual pixel position
            normdata = normdata * np.sum(sc[:,i])
            
            
            #perform the actual fit
            if model.lower() != 'all':
                fit_result = fit_stacked_single_fibre_profile(grid,normdata,weights=weights,pos=refpos,model=model,debug_level=debug_level)
            else:
                for m in model_lib:
                    fit_result = fit_stacked_single_fibre_profile(grid,normdata,weights=weights,pos=refpos,model=m,debug_level=debug_level)
                    all_fit_results[m] = fit_result
            
            
                
        #fill output structure 
        #for single user-selected model
        if model.lower() != 'all':    
            #bad pixel columns
            if fu == 1:
                #now the problem is that the fit has not actually been performed, so "fit_results" is undefined
                #We therefore need to run a dummy fit to get at least the parameter names
                fit_result = fit_stacked_single_fibre_profile(np.arange(50),fibmodel_with_amp(np.arange(50),25,1,1,2),model=model,nofit=True,debug_level=0)
                for keyname in fit_result.param_names:
                    try:
                        colfits[keyname].append(-1.)
                    except KeyError:
                        colfits[keyname] = [-1.]
                if return_stats:
                    try:
                        colfits['aic'].append(-1.)
                        colfits['bic'].append(-1.)
                        colfits['chi2'].append(-1.)
                        colfits['chi2red'].append(-1.)
                    except KeyError:
                        colfits['aic'] = [-1.]
                        colfits['bic'] = [-1.]   
                        colfits['chi2'] = [-1.]   
                        colfits['chi2red'] = [-1.] 
            #good pixel columns
            else:
                for keyname in fit_result.best_values.keys():
                    try:
                        colfits[keyname].append(fit_result.best_values[keyname])
                    except KeyError:
                        colfits[keyname] = [fit_result.best_values[keyname]]     
                if return_stats:
                    try:
                        colfits['aic'].append(fit_result.aic)
                        colfits['bic'].append(fit_result.bic)
                        colfits['chi2'].append(fit_result.chisqr)
                        colfits['chi2red'].append(fit_result.redchi)
                    except KeyError:
                        colfits['aic'] = [fit_result.aic]
                        colfits['bic'] = [fit_result.bic]   
                        colfits['chi2'] = [fit_result.chisqr]   
                        colfits['chi2red'] = [fit_result.redchi]      
        
        # for all models (comparison test only)
        else:
            for m in model_lib: 
                #bad pixel columns
                if fu == 1:
                    #now the problem is that all_fit_results is still an empty structure, ie we cannot loop over its keys
                    #We therefore need to run a dummy fit to get at least the parameter names
                    fit_result = fit_stacked_single_fibre_profile(np.arange(50),fibmodel_with_amp(np.arange(50),25,1,1,2),model=m,nofit=True,debug_level=0)
                    for keyname in fit_result.param_names:
                        try:
                            colfits[m][keyname].append(-1.)
                        except KeyError:
                            try:
                                colfits[m][keyname] = [-1.]
                            except KeyError:
                                colfits[m] = {}
                                colfits[m][keyname] = [-1.]
                    if return_stats: 
                        try:
                            colfits[m]['aic'].append(-1.)
                            colfits[m]['bic'].append(-1.)
                            colfits[m]['chi2'].append(-1.)
                            colfits[m]['chi2red'].append(-1.)
                        except KeyError:
                            colfits[m]['aic'] = [-1.]
                            colfits[m]['bic'] = [-1.]   
                            colfits[m]['chi2'] = [-1.]   
                            colfits[m]['chi2red'] = [-1.] 
                #good pixel columns
                else:
                    for keyname in all_fit_results[m].best_values.keys():
                        try:
                            colfits[m][keyname].append(all_fit_results[m].best_values[keyname])
                        except KeyError:
                            try:
                                colfits[m][keyname] = [all_fit_results[m].best_values[keyname]]  
                            except KeyError:
                                colfits[m] = {}
                                colfits[m][keyname] = [all_fit_results[m].best_values[keyname]]  
                    if return_stats: 
                        try:
                            colfits[m]['aic'].append(all_fit_results[m].aic)
                            colfits[m]['bic'].append(all_fit_results[m].bic)
                            colfits[m]['chi2'].append(all_fit_results[m].chisqr)
                            colfits[m]['chi2red'].append(all_fit_results[m].redchi)
                        except KeyError:
                            colfits[m]['aic'] = [all_fit_results[m].aic]
                            colfits[m]['bic'] = [all_fit_results[m].bic]   
                            colfits[m]['chi2'] = [all_fit_results[m].chisqr]   
                            colfits[m]['chi2red'] = [all_fit_results[m].redchi]  
                        
      
    
    if timit:
        print('Elapsed time for fitting profiles to a single order: '+np.round(time.time() - start_time,2).astype(str)+' seconds...')
        
    return colfits





def fit_stacked_single_fibre_profile(grid, data, weights=None, pos=None, model='gausslike', method='leastsq', fix_posns=False, offset=False, norm=False, nofit=False, timit=False, debug_level=0):
    """
    Fit a single fibre profile in spatial direction. Sub-pixel sampling is achieved by stacking the (input) data for multiple pixel columns.
    
    INPUT:
    'grid'           : the grid for fitting
    'data'           : the data points for fitting
    'weights'        : weights for the data during the fitting process
    'pos'            : starting guess for the peak position
    'model'          : which model do you want to use for the fitting? 
                       ['gaussian', 'gausslike', 'lorentz', 'moffat', 'voigt', 'pseudo(voigt)', 'offset_pseudo', 'pearson7',
                        'studentst', 'breitwigner', 'lognormal', 'dampedosc', 'dampedharmosc', 'expgauss', 'skewgauss', 'donaich', 'offset_pseudo', 'offset_pseudo_gausslike', 'all'] 
    'method'         : optimisation method (I've only tried 'leastsq'...)
    'fix_posns'      : boolean - do you want to fix the peak position to the initial value
    'offset'         : boolean - do you want to fit an offset?
    'norm'           : boolean - do you want to normalise the fitted function
    'nofit'          : boolean - set to TRUE if you just want to return dummy instances
    'timit'          : boolean - timing the run time...
    'debug_level'    : for debugging...
    
    OUTPUT:
    'result'         : results from "lmfit" fitting of order profiles
    """
    #print('OK, pos: ',pos)
    if timit:
        start_time = time.time()
    
    #initial guess for the locations of the individual fibre profiles from location of maxima 
    maxix = np.where(data == np.max(data))[0]
    maxval = data[maxix]
    
    #initial guesses in the right format for function "fibmodel_with_amp"
    if pos == None:
        guess = np.array([grid[maxix[0]], .7, maxval[0], 2.]).flatten()
    else:
        guess = np.array([pos, .7, maxval[0], 2.]).flatten()
    if offset:
        guess = np.append(guess,np.median(data))
       
    #create model function for fitting with LMFIT
    if not norm:
        if not offset:
            if model.lower() == 'gausslike':
                mod = Model(fibmodel_with_amp)
            if model.lower() in ('gauss','gaussian'):
                mod = GaussianModel()
            if model.lower() in ('lorentz','lorentzian'):
                mod = LorentzianModel()
            if model.lower() == 'voigt':
                mod = VoigtModel()
            if model.lower() in ('pseudo','pseudovoigt'):
                mod = PseudoVoigtModel()
            if model.lower() == 'offset_pseudo':
                gmod = GaussianModel(prefix='G_')
                lmod = LorentzianModel(prefix='L_')
                mod = gmod + lmod
            if model.lower() == 'offset_pseudo_gausslike':
                mod = Model(offset_pseudo_gausslike)
            if model.lower() == 'moffat':
                mod = MoffatModel()
            if model.lower() in ('pearson', 'pearson7'):
                mod = Pearson7Model(nan_policy='omit')
            if model.lower() in ('student', 'students', 'studentst'):
                mod = StudentsTModel()
            if model.lower() == 'breitwigner':
                mod = BreitWignerModel()
            if model.lower() == 'lognormal':
                mod = LognormalModel()
            if model.lower() == 'dampedosc':
                mod = DampedOscillatorModel()
            if model.lower() == 'dampedharmosc':
                mod = DampedHarmonicOscillatorModel()
            if model.lower() == 'expgauss':
                mod = ExponentialGaussianModel(nan_policy='omit')
            if model.lower() == 'skewgauss':
                mod = SkewedGaussianModel()
            if model.lower() == 'donaich':
                mod = DonaichModel()
            #guessmodel = fibmodel_with_amp(grid,*guess)
        else:
            mod = Model(fibmodel_with_amp_and_offset)
            #guessmodel = fibmodel_with_amp_and_offset(grid,*guess)
    else:
        if not offset:
            mod = Model(norm_fibmodel_with_amp)
            #guessmodel = norm_fibmodel_with_amp(grid,*guess)
        else:
            mod = Model(norm_fibmodel_with_amp_and_offset)
            #guessmodel = norm_fibmodel_with_amp_and_offset(grid,*guess)  
    
    #Return just the parameter names for the dummy function call
    if nofit:
        return mod
    
    #create instance of Parameters-class needed for fitting with LMFIT
    if model.lower() in ('gausslike', 'offset_pseudo', 'offset_pseudo_gausslike'):
        parms = Parameters()
    else:
        parms = mod.guess(data,x=grid)
    
    #fill Parameters() instance
    if model.lower() in ('gauss','gaussian'):
        parms['amplitude'].set(min=0.)
        parms['sigma'].set(min=0.)
    if model.lower() == 'gausslike':
        if fix_posns:
            parms.add('mu', guess[0], vary=False)
        else:
            parms.add('mu', guess[0], min=guess[0]-3, max=guess[0]+3)
        parms.add('sigma', guess[1], min=0.2, max=2.)
        parms.add('amp', guess[2], min=0.)
        parms.add('beta', guess[3], min=1., max=4.)
        if offset:
            parms.add('offset',guess[4], min=0., max=65535.)
            #parms.add('offset', guess[4], min=0.)
            #parms.add('slope', guess[5], min=-0.5, max=0.5)
    if model.lower() == 'moffat':
        parms['amplitude'].set(min=0.)
        parms['sigma'].set(min=0.)
        parms['beta'].set(min=0.)
    if model.lower() in ('pseudo','pseudovoigt'):
        #parms['fraction'].set(0.5,min=0.,max=1.)
        parms['amplitude'].set(min=0.)
        parms['sigma'].set(min=0.)
    if model.lower() == 'lognormal':
        parms['sigma'].set(value=1e-4, vary=True, expr='')
        parms['center'].set(value=np.log(guess[0]), vary=True, expr='')
        parms['amplitude'].set(1., vary=True, min=0., expr='')
    if model.lower() == 'dampedosc':
        parms['sigma'].set(1e-4, vary=True, expr='')
        parms['amplitude'].set(1e-4, vary=True, min=0., expr='')
    if model.lower() == 'offset_pseudo':
        parms = gmod.guess(data,x=grid)
        parms.update(lmod.guess(data,x=grid))
        parms['G_amplitude'].set(parms['G_amplitude']/2., min=0., vary=True)
        parms['L_amplitude'].set(parms['L_amplitude']/2., min=0., vary=True)
    if model.lower() == 'skewgauss':
        parms['amplitude'].set(min=0.)
        parms['sigma'].set(min=0.)
    if model.lower() == 'offset_pseudo_gausslike':
        parms.add('G_amplitude', guess[2], min=0.)
        parms.add('L_amplitude', guess[2], min=0.)
        parms.add('G_center', guess[0], min=guess[0]-3, max=guess[0]+3)
        parms.add('L_center', guess[0], min=guess[0]-3, max=guess[0]+3)
        parms.add('G_sigma', guess[1], min=0.1, max=10.)
        parms.add('L_sigma', guess[1], min=0.1, max=10.)
        parms.add('beta', guess[3], min=1., max=4.)
        
        
    #perform fit
    if not nofit:
        result = mod.fit(data,parms,x=grid,weights=weights,method=method)
      
    
    if debug_level >= 2 and not nofit:
        plot_osf = 10
        plot_os_grid = np.linspace(grid[0],grid[-1],plot_osf * (len(grid)-1) + 1)
        #plot_os_data = np.interp(plot_os_grid, grid, data)
        
        guessmodel = mod.eval(result.init_params,x=plot_os_grid)
        bestmodel = mod.eval(result.params,x=plot_os_grid)
#         if not norm:
#             if not offset:
#                 guessmodel = fibmodel_with_amp(plot_os_grid,*guess)
#                 bestparms = np.array([result.best_values['mu'], result.best_values['sigma'], result.best_values['amp'], result.best_values['beta']])
#                 bestmodel = fibmodel_with_amp(plot_os_grid,*bestparms)
#             else:
#                 guessmodel = fibmodel_with_amp_and_offset(plot_os_grid,*guess)
#                 bestparms = np.array([result.best_values['mu'], result.best_values['sigma'], result.best_values['amp'], result.best_values['beta'], result.best_values['offset']])
#                 bestmodel = fibmodel_with_amp_and_offset(plot_os_grid,*bestparms)
#         else:
#             if not offset:
#                 guessmodel = norm_fibmodel_with_amp(plot_os_grid,*guess)
#                 bestparms = np.array([result.best_values['mu'], result.best_values['sigma'], result.best_values['amp'], result.best_values['beta']])
#                 bestmodel = norm_fibmodel_with_amp(plot_os_grid,*bestparms)
#             else:
#                 guessmodel = norm_fibmodel_with_amp_and_offset(plot_os_grid,*guess)
#                 bestparms = np.array([result.best_values['mu'], result.best_values['sigma'], result.best_values['amp'], result.best_values['beta'], result.best_values['offset']])
#                 bestmodel = norm_fibmodel_with_amp_and_offset(plot_os_grid,*bestparms)
        plt.figure()
        plt.title('model = '+model.title())
        plt.xlabel('pixel number (cross-disp. direction)')
        plt.plot(grid, data, 'b.')
        if model.lower() == 'gausslike':
            plt.xlim(result.best_values['mu']-3,result.best_values['mu']+3)
        else:    
            #plt.xlim(result.best_values['center']-3,result.best_values['center']+3)
            plt.xlim(grid[maxix[0]]-3,grid[maxix[0]]+3)
        plt.plot(plot_os_grid, guessmodel, 'k--', label='initial guess')
        plt.plot(plot_os_grid, bestmodel, 'r-', label='best-fit model')     
        plt.legend()
    
    if timit:
        print(time.time() - start_time, ' seconds')
    
    return result





def fit_single_fibre_profile(grid, data, pos=None, osf=1, fix_posns=False, method='leastsq', offset=False, debug_level=0, timit=False):
    
    #print('OK, pos: ',pos)
    if timit:
        start_time = time.time()
    
    if pos == None:
        #initial guess for the locations of the individual fibre profiles from location of maxima
        #maxix,maxval = find_maxima(data, return_values=1)
        maxix = np.where(data == np.max(data))[0]
        maxval = data[maxix]
    else:
        maxval = data[np.int(np.rint(pos-grid[0]))]
    
    # go to oversampled grid (the number of grid points should be: n_os = ((n_orig-1)*osf)+1
    if osf != 1:
        os_grid = np.linspace(grid[0],grid[-1],osf * (len(grid)-1) + 1)
        os_data = np.interp(os_grid, grid, data)
        grid = os_grid
        data = os_data
    
    #initial guesses in the right format for function "fibmodel_with_amp"
    if pos == None:
        guess = np.array([maxix[0]+grid[0], .7, maxval[0], 2.]).flatten()
    else:
        guess = np.array([pos, .7, maxval, 2.]).flatten()
    if offset:
        guess = np.append(guess,np.median(data))
       
    #create model function for fitting with LMFIT
    #model = Model(nineteen_fib_model_explicit)
    if not offset:
        model = Model(fibmodel_with_amp)
        guessmodel = fibmodel_with_amp(grid,*guess)
    else:
        model = Model(fibmodel_with_amp_and_offset)
        guessmodel = fibmodel_with_amp_and_offset(grid,*guess)
    
    #create instance of Parameters-class needed for fitting with LMFIT
    parms = Parameters()
    if fix_posns:
        parms.add('mu', guess[0], vary=False)
    else:
        parms.add('mu', guess[0], min=guess[0]-3*osf, max=guess[0]+3*osf)
    parms.add('sigma', guess[1], min=0.5, max=1.)
    parms.add('amp', guess[2], min=0.)
    parms.add('beta', guess[3], min=1., max=4.)
    if offset:
        parms.add('offset',guess[4], min=0., max=65535.)
    #parms.add('offset', guess[4], min=0.)
    #parms.add('slope', guess[5], min=-0.5, max=0.5)
    
    #perform fit
    result = model.fit(data,parms,xarr=grid,method=method)
    
    if debug_level >= 1:
        plot_osf = 10
        plot_os_grid = np.linspace(grid[0],grid[-1],plot_osf * (len(grid)-1) + 1)
        #plot_os_data = np.interp(plot_os_grid, grid, data)
        guessmodel = fibmodel_with_amp(plot_os_grid,*guess)
        bestparms = np.array([result.best_values['mu'], result.best_values['sigma'], result.best_values['amp'], result.best_values['beta']])
        bestmodel = fibmodel_with_amp(plot_os_grid,*bestparms)
        plt.plot(grid, data, 'bx')
        plt.plot(plot_os_grid, guessmodel, 'r--')
        plt.plot(plot_os_grid, bestmodel, 'g-')     
    
    if timit:
        print(time.time() - start_time, ' seconds')
    
    return result





def fit_profiles_single_order(stripe_rows, stripe_columns, ordpol, osf=1, method='leastsq', offset=False, timit=False, silent=False):
    
    if not silent:
        choice = None
        while choice is None:
            choice = raw_input("WARNING: Fitting the fibre profiles for an entire order currently takes more than 2 hours. Do you want to continue? [y/n]: ")
            if choice not in ('y','n'):
                print('Invalid input! Please try again...')
                choice = None
    else:
        choice = 'y'
            
    if choice == 'n':
        print('OK, stopping script...')
        sys.exit()
    else:         
        
        if timit:
            start_time = time.time()
        print('Fitting fibre profiles for one order...') 
        #loop over all columns for one order and do the profile fitting
        #'colfits' is a dictionary, that has 4096 keys. Each key is an instance of the 'ModelResult'-class from the 'lmfit' package
        colfits = {}
        npix = stripe_columns.shape[1]
        fiblocs = np.poly1d(ordpol)(np.arange(npix))
        #starting_values = None
        for i in range(npix):
            #fit_result = fitfib_single_cutout(stripe_rows[:,i], stripe_columns[:,i], osf=osf, method=method)
            if not silent:
                print('i = ',str(i))
            fu = 0
            #check if that particular cutout falls fully onto CCD
            checkprod = np.product(stripe_rows[1:,i].astype(float))    #exclude the first row number, as that can legitimately be zero
            if checkprod == 0:
                fu = 1
                checksum = np.sum(stripe_rows[:,i])
                if checksum == 0:
                    print('WARNING: the entire cutout lies outside the chip!!!')
                    #fit_result = fit_single_fibre_profile(np.arange(len(stripe_rows[:,i])),stripe_columns[:,i],guess=None)
                    best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
                else:
                    print('WARNING: parts of the cutout lie outside the chip!!!')
                    #fit_result = fit_single_fibre_profile(np.arange(len(stripe_rows[:,i])),stripe_columns[:,i],guess=None)
                    best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
            else:    
                #fit_result = fit_single_fibre_profile(stripe_rows[:,i],stripe_columns[:,i],guess=None,timit=1)
                fit_result = fit_single_fibre_profile(stripe_rows[:,i],stripe_columns[:,i]-1.,pos=fiblocs[i],offset=offset,timit=0)     #the minus 1 is necessary because we added an offset to start with, due to stripes needing that
            #starting_values = fit_result.best_values
            #colfits['col_'+str(i)] = fit_result
            if fu == 1:
                for keyname in best_values.keys():
                    try:
                        colfits[keyname].append(best_values[keyname])
                    except KeyError:
                        colfits[keyname] = [best_values[keyname]]
            else:
                for keyname in fit_result.best_values.keys():
                    try:
                        colfits[keyname].append(fit_result.best_values[keyname])
                    except KeyError:
                        colfits[keyname] = [fit_result.best_values[keyname]]
    
    if timit:
        print('Elapsed time for fitting profiles to a single order: '+str(time.time() - start_time)+' seconds...')
    
    return colfits





def fit_profiles(P_id, stripes, err_stripes, mask=None, stacking=True, slit_height=25, model='gausslike', return_stats=False, timit=False):
    """
    This routine determines the profiles of the fibres in spatial direction. This is an extremely crucial step, as the pre-defined profiles
    are then used during the optimal extraction, as well as during the determination of the relative fibre intensities!!!
    
    INPUT:
    'P_id'          : dictionary of the form of {order: np.poly1d, ...} (as returned by "identify_stripes")
    'stripes'       : dictionary containing the flux in the extracted stripes (keys = orders)
    'err_stripes'   : dictionary containing the errors in the extracted stripes (keys = orders)
    'mask'          : dictionary of boolean masks (keys = orders) from "find_stripes" (masking out regions of very low signal)
    'stacking'      : boolean - do you want to stack the profiles from multiple pixel-columns (in order to achieve sub-pixel sampling)?
    'slit_height'   : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'model'         : the name of the mathematical model used to describe the profile of an individual fibre profile
    'return_stats'  : boolean - do you want to include some goodness-of-fit statistics in the output (ie AIC, BIC, CHISQ and REDCHISQ)?
    'timit'         : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'fibre_profiles'  : dictionary (keys=orders) containing the calculated spatial-direction fibre profiles
    """
    
    print('Fitting fibre profiles...')
    
    if timit:
        start_time = time.time()
    
    #create "global" parameter dictionary for entire chip
    fibre_profiles = {}
    #loop over all orders
    for ord in sorted(list(P_id.keys())):
        print('OK, now processing '+str(ord))
        
        ordpol = P_id[ord]
        
        # define stripe
        stripe = stripes[ord]
        err_stripe = err_stripes[ord]
        # find the "order-box"
        sc,sr = flatten_single_stripe(stripe, slit_height=slit_height, timit=False)
        err_sc,err_sr = flatten_single_stripe(err_stripe, slit_height=slit_height, timit=False)
        
        npix = sc.shape[1]
        if mask is None:
            mask = {}
            mask[ord] = np.ones(npix, dtype='bool')
        
        # fit profile for single order and save result in "global" parameter dictionary for entire chip
        if stacking:
            colfits = determine_spatial_profiles_single_order(sc, sr, err_sc, ordpol, ordmask=mask[ord], model=model, return_stats=return_stats, timit=timit)
        else:
            colfits = fit_profiles_single_order(sr,sc,ordpol,osf=1,silent=True,timit=timit)
        fibre_profiles[ord] = colfits
    
    if timit:
        print('Time elapsed: '+str(int(time.time() - start_time))+' seconds...')  
          
    return fibre_profiles





def fit_profiles_from_indices(P_id, img, err_img, stripe_indices, mask=None, stacking=True, slit_height=25, model='gausslike', return_stats=False, timit=False):
    """
    This routine determines the profiles of the fibres in spatial direction. This is an extremely crucial step, as the pre-defined profiles are then used during
    the optimal extraction, as well as during the determination of the relative fibre intensities!!!
    
    CLONE OF "fit_profiles", but using stripe-indices, rather than stripes...
    
    INPUT:
    'P_id'          : dictionary of the form of {order: np.poly1d, ...} (as returned by "identify_stripes")
    'img'           : 2-dim input array/image
    'err_img'       : estimated uncertainties in the 2-dim input array/image
    'mask'          : dictionary of boolean masks (keys = orders) from "find_stripes" (masking out regions of very low signal)
    'stacking'      : boolean - do you want to stack the profiles from multiple pixel-columns (in order to achieve sub-pixel sampling)?
    'slit_height'   : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'model'         : the name of the mathematical model used to describe the profile of an individual fibre profile
    'return_stats'  : boolean - do you want to include some goodness-of-fit statistics in the output (ie AIC, BIC, CHISQ and REDCHISQ)?
    'timit'         : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'fibre_profiles'  : dictionary (keys=orders) containing the calculated spatial-direction fibre profiles
    """
    
    
    print('Fitting fibre profiles...')
    
    if timit:
        start_time = time.time()
        
    #create "global" parameter dictionary for entire chip
    fibre_profiles = {}
    #loop over all orders
    for ord in sorted(list(P_id.keys())):
        print('OK, now processing '+str(ord))
        
        ordpol = P_id[ord]
        
        # define stripe
        indices = stripe_indices[ord]
        # find the "order-box"
        sc,sr = flatten_single_stripe_from_indices(img, indices, slit_height=slit_height, timit=False)
        err_sc,err_sr = flatten_single_stripe_from_indices(err_img, indices, slit_height=slit_height, timit=False)
        
        npix = sc.shape[1]
        if mask is None:
            mask = {}
            mask[ord] = np.ones(npix, dtype='bool')
        
        # fit profile for single order and save result in "global" parameter dictionary for entire chip
        if stacking:
            colfits = determine_spatial_profiles_single_order(sc, sr, err_sc, ordpol, ordmask=mask[ord], model=model, return_stats=return_stats, timit=timit)
        else:
            colfits = fit_profiles_single_order(sr,sc,ordpol,osf=1,silent=True,timit=timit)
        fibre_profiles[ord] = colfits
    
    if timit:
        print('Time elapsed: '+str(int(time.time() - start_time))+' seconds...')  
          
    return fibre_profiles





def make_model_stripes_gausslike(fibre_profiles, flat, err_img, stripe_indices, mask, degpol=5, slit_height=10, return_fitpars=False, debug_level=0, timit=False):
    """
    Using the fibre_profiles dictionary from "fit_profiles(_from_indices)", we enforce that the parameters describing the spatial fibre profiles
    are only smoothly varying as a function of pixel number in dispersion direction by fitting a low-level polynomial to the fitted values of the parameters.
    
    We construct two sets of stripes as output:
    (1) 'fitted_stripes' - by using the fibre_profiles from each pixel column for each order
    (2) 'model_stripes'  - by using the "smoothly-varying-enforced" parameters
    
    INPUT:
    'fibre_profiles'  : dictionary (keys=orders) containing the calculated spatial-direction fibre profiles
    'flat'            : the 2-dim master white / flat-field image
    'err_img'         : the 2-dim array of the corresponding uncertainties
    'stripe_indices'  : dictionary (keys = orders) containing the indices of the pixels that are identified as the "stripes" (ie the to-be-extracted regions centred on the orders)
    'mask'            : dictionary of boolean masks (keys = orders) from "find_stripes" (masking out regions of very low signal)
    'slit_height'     : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'return_fitpars'  : boolean - do you want to return the best-fit polynomial coefficients for the parameters as well?
    'debug_level'     : for debugging...
    'timit'           : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'fitted_stripes'  : dictionary (keys=orders) containing the reconstructed model using the best-fit solutions to the individual-pixel-column profile fits
    'model_stripes'   : dictionary (keys=orders) containing the reconstructed model using the "smoothly-varying-enforced" parameters
    
    TODO:
    make this more generic to include different analytical models for the profile shapes
    """
    
    if timit:
        start_time = time.time()
    
    #initialize the dictionaries
    fitted_stripes = {}
    model_stripes = {}
    if return_fitpars:
        fitpars = {}
    
    #loop over all orders
    for ord in sorted(list(fibre_profiles.keys())):
        if debug_level >= 1:
            print('Creating model for ',ord)
            
        # initialize the dictionaries for each order
        fitted_stripes[ord] = {}
        model_stripes[ord] = {}
        if return_fitpars:
            fitpars[ord] = {}
        
        # fill "fitted_stripes" dictionary
        parms = np.array([fibre_profiles[ord]['mu'],fibre_profiles[ord]['sigma'],fibre_profiles[ord]['amp'],fibre_profiles[ord]['beta']])
        indices = stripe_indices[ord]
        sc,sr = flatten_single_stripe_from_indices(flat, indices, slit_height=slit_height, timit=False)
        err_sc,err_sr = flatten_single_stripe_from_indices(err_img, indices, slit_height=slit_height, timit=False)
        NY,NX = sc.shape
        xx = np.arange(NX)
        fitted_stripes[ord] = fibmodel_with_amp(sr,*parms)
        
        ### NOW ENFORCE SMOOTHLY VARYING PROFILES!!!
        # get weights for fitting of smooth function to parameters across order 
        collapsed_signal = np.sum(sc,axis=0)     #BTW this is a quick-and-dirty style extracted spectrum 
        #collapsed_error = np.sqrt(collapsed_signal + NY*RON**2)
        collapsed_error = np.sqrt(np.sum(err_sc**2,axis=0))
        #use relative errors here, otherwise the order centres will get down-weighted
        w = 1./((collapsed_error / collapsed_signal)**2)   
        
        #perform the polynomial fits to the parameters across the dispersion direction (MASKS for fitting comes from "find_stripes()")
        p_mu = np.poly1d(np.polyfit(xx[mask[ord]], np.array(fibre_profiles[ord]['mu'])[mask[ord]], degpol, w=w[mask[ord]]))
        p_sigma = np.poly1d(np.polyfit(xx[mask[ord]], np.array(fibre_profiles[ord]['sigma'])[mask[ord]], degpol, w=w[mask[ord]]))
        p_amp = np.poly1d(np.polyfit(xx[mask[ord]], np.array(fibre_profiles[ord]['amp'])[mask[ord]], degpol, w=w[mask[ord]]))
        p_beta = np.poly1d(np.polyfit(xx[mask[ord]], np.array(fibre_profiles[ord]['beta'])[mask[ord]], degpol, w=w[mask[ord]]))
        if return_fitpars:
            fitpars[ord]['p_mu'] = p_mu
            fitpars[ord]['p_sigma'] = p_sigma
            fitpars[ord]['p_amp'] = p_amp
            fitpars[ord]['p_beta'] = p_beta 
        
        # fill "model_stripes" dictionary
        smooth_parms = np.array([p_mu(xx),p_sigma(xx),p_amp(xx),p_beta(xx)])
        model_stripes[ord] = fibmodel_with_amp(sr,*smooth_parms)
     
    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds...')   
        
    if return_fitpars:
        return fitted_stripes, model_stripes, fitpars
    else:
        return fitted_stripes, model_stripes





def older_routines():
#
#
# def fit_stacked_single_fibre_profile_v1(grid, data, pos=None, model='gausslike', fix_posns=False, method='leastsq', offset=False, norm=False, timit=False, debug_level=0):
#     """
#
#     OLDER VERSION:
#     NOT CURRENTLY USED !!!!!
#
#
#     INPUT:
#     'grid'           : the grid for fitting
#     'data'           : the data points for fitting
#     'pos'            : starting guess for the peak position
#     'model'          : which model do you want to use for the fitting? ['gauss(ian)', 'gausslike', 'lorentz(ian)', 'voigt', 'pseudo(voigt)', 'moffat', 'all']
#     'fix_posns'      : boolean - do you want to fix the peak position to the initial value
#     'method'         : optimisation method (I've only tried 'leastsq'...)
#     'offset'         : boolean - do you want to fit an offset?
#     'norm'           : boolean - do you want to normalise the fitted function
#     'timit'          : boolean - timing the run time...
#     'debug_level'    : for debugging...
#
#     OUTPUT:
#     'result'         : results from "lmfit" fitting of order profiles
#     """
#     #print('OK, pos: ',pos)
#     if timit:
#         start_time = time.time()
#
#     #initial guess for the locations of the individual fibre profiles from location of maxima
#     maxix = np.where(data == np.max(data))[0]
#     maxval = data[maxix]
#
#     #initial guesses in the right format for function "fibmodel_with_amp"
#     if pos == None:
#         guess = np.array([grid[maxix[0]], .7, maxval[0], 2.]).flatten()
#     else:
#         guess = np.array([pos, .7, maxval[0], 2.]).flatten()
#     if offset:
#         guess = np.append(guess,np.median(data))
#
#     #create model function for fitting with LMFIT
#     if model.lower() != 'all':
#         if not norm:
#             if not offset:
#                 if model.lower() == 'gausslike':
#                     mod = Model(fibmodel_with_amp)
#                 if model.lower() in ('gauss','gaussian'):
#                     mod = GaussianModel()
#                 if model.lower() in ('lorentz','lorentzian'):
#                     mod = LorentzianModel()
#                 if model.lower() == ('voigt'):
#                     mod = VoigtModel()
#                 if model.lower() in ('pseudo','pseudovoigt'):
#                     mod = PseudoVoigtModel()
#                 if model.lower() == 'moffat':
#                     mod = MoffatModel()
#                 #guessmodel = fibmodel_with_amp(grid,*guess)
#             else:
#                 mod = Model(fibmodel_with_amp_and_offset)
#                 #guessmodel = fibmodel_with_amp_and_offset(grid,*guess)
#         else:
#             if not offset:
#                 mod = Model(norm_fibmodel_with_amp)
#                 #guessmodel = norm_fibmodel_with_amp(grid,*guess)
#             else:
#                 mod = Model(norm_fibmodel_with_amp_and_offset)
#                 #guessmodel = norm_fibmodel_with_amp_and_offset(grid,*guess)
#
#         #create instance of Parameters-class needed for fitting with LMFIT
#         parms = Parameters()
#
#         if model.lower() == 'gausslike':
#             if fix_posns:
#                 parms.add('mu', guess[0], vary=False)
#             else:
#                 parms.add('mu', guess[0], min=guess[0]-3, max=guess[0]+3)
#             parms.add('sigma', guess[1], min=0.2, max=2.)
#             parms.add('amp', guess[2], min=0.)
#             parms.add('beta', guess[3], min=1., max=4.)
#             if offset:
#                 parms.add('offset',guess[4], min=0., max=65535.)
#                 #parms.add('offset', guess[4], min=0.)
#                 #parms.add('slope', guess[5], min=-0.5, max=0.5)
#         else:
#             parms = mod.guess(data,x=grid)
#             #parms = model.make_params(sigma=guess[1], center=guess[0], amplitude=guess[2])
#
#         #perform fit
#         result = mod.fit(data,parms,x=grid,method=method)
#
#     #this is for the model='all' case, ie for model selection
#     else:
#         #create Model() instances
#         mod_gauss = GaussianModel()
#         mod_gausslike = Model(fibmodel_with_amp)
#         mod_lorentz = LorentzianModel()
#         mod_voigt = VoigtModel()
#         mod_pseudo = PseudoVoigtModel()
#         mod_moffat = MoffatModel()
#         #create instances of Parameters()
#         parms_gauss = Parameters()
#         parms_gausslike = Parameters()
#         parms_lorentz = Parameters()
#         parms_voigt = Parameters()
#         parms_pseudo = Parameters()
#         parms_moffat = Parameters()
#         #fill them with initial guesses
#         parms_gauss = mod_gauss.guess(data,x=grid)
#         parms_lorentz = mod_lorentz.guess(data,x=grid)
#         parms_voigt = mod_voigt.guess(data,x=grid)
#         parms_pseudo = mod_pseudo.guess(data,x=grid)
#         parms_moffat = mod_moffat.guess(data,x=grid)
#         #gausslike is a special case b/c it is a user-defined function [WARNING: NORM or OFFSET not implemented yet!!!]
#         parms_gausslike.add('mu', guess[0], min=guess[0]-3, max=guess[0]+3)
#         parms_gausslike.add('sigma', guess[1], min=0.2, max=2.)
#         parms_gausslike.add('amp', guess[2], min=0.)
#         parms_gausslike.add('beta', guess[3], min=1., max=4.)
#
#         #perform fit
#         result_gauss = mod_gauss.fit(data,parms_gauss,x=grid,method=method)
#         result_gausslike = mod_gausslike.fit(data,parms_gausslike,x=grid,method=method)
#         result_lorentz = mod_lorentz.fit(data,parms_lorentz,x=grid,method=method)
#         result_voigt = mod_voigt.fit(data,parms_voigt,x=grid,method=method)
#         result_pseudo = mod_pseudo.fit(data,parms_pseudo,x=grid,method=method)
#         result_moffat = mod_moffat.fit(data,parms_moffat,x=grid,method=method)
#
#         #save everything in one dictionary
#         result = {'gauss':result_gauss, 'gausslike':result_gausslike, 'lorentz':result_lorentz, 'voigt':result_voigt, 'pseudo':result_pseudo, 'moffat':result_moffat}
#
#
#     if debug_level >= 1 and model.lower() != 'all':
#         plot_osf = 10
#         plot_os_grid = np.linspace(grid[0],grid[-1],plot_osf * (len(grid)-1) + 1)
#         #plot_os_data = np.interp(plot_os_grid, grid, data)
#
#         guessmodel = mod.eval(result.init_params,x=plot_os_grid)
#         bestmodel = mod.eval(result.params,x=plot_os_grid)
# #         if not norm:
# #             if not offset:
# #                 guessmodel = fibmodel_with_amp(plot_os_grid,*guess)
# #                 bestparms = np.array([result.best_values['mu'], result.best_values['sigma'], result.best_values['amp'], result.best_values['beta']])
# #                 bestmodel = fibmodel_with_amp(plot_os_grid,*bestparms)
# #             else:
# #                 guessmodel = fibmodel_with_amp_and_offset(plot_os_grid,*guess)
# #                 bestparms = np.array([result.best_values['mu'], result.best_values['sigma'], result.best_values['amp'], result.best_values['beta'], result.best_values['offset']])
# #                 bestmodel = fibmodel_with_amp_and_offset(plot_os_grid,*bestparms)
# #         else:
# #             if not offset:
# #                 guessmodel = norm_fibmodel_with_amp(plot_os_grid,*guess)
# #                 bestparms = np.array([result.best_values['mu'], result.best_values['sigma'], result.best_values['amp'], result.best_values['beta']])
# #                 bestmodel = norm_fibmodel_with_amp(plot_os_grid,*bestparms)
# #             else:
# #                 guessmodel = norm_fibmodel_with_amp_and_offset(plot_os_grid,*guess)
# #                 bestparms = np.array([result.best_values['mu'], result.best_values['sigma'], result.best_values['amp'], result.best_values['beta'], result.best_values['offset']])
# #                 bestmodel = norm_fibmodel_with_amp_and_offset(plot_os_grid,*bestparms)
#         plt.figure()
#         plt.title('model = '+model.title())
#         plt.xlabel('pixel number (cross-disp. direction)')
#         plt.plot(grid, data, 'b.')
#         if model.lower() == 'gausslike':
#             plt.xlim(result.best_values['mu']-3,result.best_values['mu']+3)
#         else:
#             plt.xlim(result.best_values['center']-3,result.best_values['center']+3)
#         plt.plot(plot_os_grid, guessmodel, 'k--', label='initial guess')
#         plt.plot(plot_os_grid, bestmodel, 'r-', label='best-fit model')
#         plt.legend()
#
#     if timit:
#         print(time.time() - start_time, ' seconds')
#
#     return result
#
# def fitfib_single_cutout(grid,data,osf=1,method='leastsq',debug_level=0):
#     """
#     NOT CURRENTLY USED
#     """
#     #timing test
#     start_time = time.time()
#
#     nfib = 19
#
#     #initial guess for the locations of the individual fibre profiles from location of maxima
#     #NAIVE METHOD:
#     maxix,maxval = find_maxima(data, return_values=1)
#
#     if len(maxix) == len(maxix):
#     #if len(maxix) != nfib:
#         # smooth image slightly for noise reduction
#         filtered_data = ndimage.gaussian_filter(data, 2.)
#         #filtered_data = ndimage.gaussian_filter(data, 3.)
#         maxval = np.r_[[.3]*6,[.7]*3,1,[.7]*3,[.3]*6] * np.max(maxval)
#
#         #ALTERNATIVE 1: FIND HIGHEST MAXIMUM AND GO 2.5 PIXELS TO EACH SIDE FOR NEIGHBOURING FIBRES
#         print('Did not find exactly 19 peaks! Using smoothed data to determine starting values for peak locations...')
#         #top_max = np.where(filtered_data == np.max(filtered_data))[0]
#         top_max = np.where(filtered_data == np.max(filtered_data))[0]
#         ##top_max = np.where(data == np.max(data))[0]
#
# #         #ALTERNATIVE 2:
# #         #do a cross-correlation between the data and a "standard profile model" (with highest peak at grid[len(grid)/2] ) to find the rough position of the central peak, then go either side to get locations of other peaks
# #         print('Did not find exactly 19 peaks! Performing a cross-correlation to determine starting values for peak locations...')
# #         stdguess = np.append((np.array([np.arange(-22.5,22.51,2.5)+grid[len(grid)/2], maxval]).flatten()),[.7,2.])
# #         stdmod = nineteen_fib_model_explicit_onesig_onebeta(grid,*stdguess)
# #         xc = np.correlate(data,stdmod,mode='same')
# #         #now fit gaussian + offset to CCF
# #         pguess = [grid[len(grid)/2], 17., 2.7e7, 0.]
# #         popt,pcov = curve_fit(gaussian_with_offset,grid,xc,p0=pguess)
# #         ccf_fit = gaussian_with_offset(grid, *popt)
# #         shift = grid[len(grid)/2] - popt[0]
# #
# #         if debug_level > 0:
# #             plt.figure()
# #             plt.title('CCF')
# #             plt.plot(grid,xc)
# #             plt.plot(grid,gaussian_with_offset(grid,*pguess),'r--')
# #             plt.plot(grid,gaussian_with_offset(grid,*popt),'g-')
# #             plt.show()
# #
# #         top_max = popt[0]
#
#
#         #for ALTERNATIVE 1:
#         maxix = np.arange(-22.5,22.51,2.5) + top_max
#         #for ALTERNATIVE 2:
#         #maxix = np.arange(-22.5,22.51,2.5) + top_max - grid[0]
#         #maxval = np.r_[[.3]*6,[.7]*3,1,[.7]*3,[.3]*6] * filtered_data[top_max]     #already done above now
#
#     # go to oversampled grid (the number of grid points should be: n_os = ((n_orig-1)*osf)+1
#     if osf != 1:
#         os_grid = np.linspace(grid[0],grid[-1],osf * (len(grid)-1) + 1)
#         os_data = np.interp(os_grid, grid, data)
#         grid = os_grid
#         data = os_data
#
#     #nur zur Erinnerung...
#     #def fibmodel(xarr, mu, fwhm, beta=2, alpha=0, norm=0):
#
# #     os_prof = np.zeros((len(maxix),len(os_grid)))
# #     for i in range(len(maxix)):
# #         os_prof[i,:] = fibmodel(os_grid, maxix[i]+col_rows[0], sigma=1.5, norm=1) * maxval[i]
# #     os_prof_sum = np.sum(os_prof, axis=0)
#
#     #initial guesses in the right format for function "multi_fib_model"
#     #guess = np.array([maxix+grid[0], [0.7]*nfib, maxval, [2.]*nfib]).flatten()
#     guess = np.append((np.array([maxix+grid[0], maxval]).flatten()),[.7,2.])
#
#     #create model function for fitting with LMFIT
#     #model = Model(nineteen_fib_model_explicit)
#     model = Model(nineteen_fib_model_explicit_onesig_onebeta)
#     #create instance of Parameters-class needed for fitting with LMFIT
#     parms = Parameters()
#     for i in range(nfib):
#         parms.add('mu'+str(i+1), guess[i], min=guess[i]-3*osf, max=guess[i]+3*osf)
#         #parms.add('sigma'+str(i+1), guess[nfib+i], min=0.)
#         parms.add('amp'+str(i+1), guess[nfib+i], min=0.)
#         #parms.add('beta'+str(i+1), guess[3*nfib+i], min=1.5, max=3.)
#     parms.add('sigma', guess[2*nfib], min=0.)
#     parms.add('beta', guess[2*nfib+1], min=1.5, max=3.)
#     #perform fit
#     result = model.fit(data,parms,x=grid,method=method)
#
#     print(time.time() - start_time, ' seconds')
#
#     return result
#
# def multi_fib_model_star(x,*p):
#     """
#     NOT CURRENTLY USED
#     """
#     #determine number of fibres
#     #nfib = len(p)/4
#     nfib = 1
#
#     #fill input-arrays for function "fibmodel"
#     if nfib == 1:
#         mu = p[0]
#         sigma = p[1]
#         amp = p[2]
#         beta = p[3]
#     else:
#         mu = p[:nfib]
#         sigma = p[nfib:nfib*2]
#         amp = p[nfib*2:nfib*3]
#         beta = p[nfib*3:]
#
#     if nfib == 1:
#         model = fibmodel(x, mu, sigma, beta=beta, norm=1) * amp
#     else:
#         single_models = np.zeros((nfib, len(x)))
#         for i in range(nfib):
#             single_models[i,:] = fibmodel(x, mu[i], sigma[i], beta=beta[i], norm=1) * amp[i]
#         model = np.sum(single_models, axis=0)
#
#     return model
#
# def multi_fib_model(x,p):
#     """
#     NOT CURRENTLY USED
#     """
#     #determine number of fibres
#     nfib = len(p)/4
#     #nfib = 19
#
#     #fill input-arrays for function "fibmodel"
#     if nfib == 1:
#         mu = p[0]
#         sigma = p[1]
#         amp = p[2]
#         beta = p[3]
#     else:
#         mu = p[:nfib]
#         sigma = p[nfib:nfib*2]
#         amp = p[nfib*2:nfib*3]
#         beta = p[nfib*3:]
#
# #     print('nfib = ',nfib)
# #     print('mu = ',mu)
# #     print('sigma = ',sigma)
# #     print('amp = ',amp)
# #     print('beta = ',beta)
# #     return 1
# #    print('beta = ',beta)
# #     #check if all the arrays provided have the same length
# #     if len(fwhm) != len(mu):
# #         print('ERROR: "fwhm" and "mu" must have the same length!!!')
# #         quit()
# #     if len(amp) != len(mu):
# #         print('ERROR: "amp" and "mu" must have the same length!!!')
# #         quit()
# #     if len(fwhm) != len(amp):
# #         print('ERROR: "fwhm" and "amp" must have the same length!!!')
# #         quit()
#
# #     #determine number of fibres
# #     if isinstance(mu, collections.Sequence):
# #         nfib = len(mu)
# #     else:
# #         nfib = 1
#
#     if nfib == 1:
#         model = fibmodel(x, mu, sigma, beta=beta, norm=1) * amp
#     else:
#         single_models = np.zeros((nfib, len(x)))
#         for i in range(nfib):
#             single_models[i,:] = fibmodel(x, mu[i], sigma[i], beta=beta[i], norm=1) * amp[i]
#         model = np.sum(single_models, axis=0)
#
#     return model
#
# def nineteen_fib_model_explicit(x, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9, mu10, mu11, mu12, mu13, mu14, mu15, mu16, mu17, mu18, mu19,
#                                 sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7, sigma8, sigma9, sigma10, sigma11, sigma12, sigma13, sigma14, sigma15, sigma16, sigma17, sigma18, sigma19,
#                                 amp1, amp2, amp3, amp4, amp5, amp6, amp7, amp8, amp9, amp10, amp11, amp12, amp13, amp14, amp15, amp16, amp17, amp18, amp19,
#                                 beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8, beta9, beta10, beta11, beta12, beta13, beta14, beta15, beta16, beta17, beta18, beta19):
#     """
#     NOT CURRENTLY USED
#     """
#
#     nfib=19
#
#     mu = np.array([mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9, mu10, mu11, mu12, mu13, mu14, mu15, mu16, mu17, mu18, mu19])
#     sigma= np.array([sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7, sigma8, sigma9, sigma10, sigma11, sigma12, sigma13, sigma14, sigma15, sigma16, sigma17, sigma18, sigma19])
#     amp = np.array([amp1, amp2, amp3, amp4, amp5, amp6, amp7, amp8, amp9, amp10, amp11, amp12, amp13, amp14, amp15, amp16, amp17, amp18, amp19])
#     beta = np.array([beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8, beta9, beta10, beta11, beta12, beta13, beta14, beta15, beta16, beta17, beta18, beta19])
#
#     single_models = np.zeros((nfib, len(x)))
#     for i in range(nfib):
#         single_models[i,:] = fibmodel(x, mu[i], sigma[i], beta=beta[i], norm=0) * amp[i]
#     model = np.sum(single_models, axis=0)
#
#     return model
#
# def nineteen_fib_model_explicit_onesig_onebeta(x, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9, mu10, mu11, mu12, mu13, mu14, mu15, mu16, mu17, mu18, mu19,
#                                                amp1, amp2, amp3, amp4, amp5, amp6, amp7, amp8, amp9, amp10, amp11, amp12, amp13, amp14, amp15, amp16, amp17, amp18, amp19, sigma, beta):
#     """
#     NOT CURRENTLY USED
#     """
#
#     nfib=19
#
#     mu = np.array([mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9, mu10, mu11, mu12, mu13, mu14, mu15, mu16, mu17, mu18, mu19])
#     #sigma= np.array([sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7, sigma8, sigma9, sigma10, sigma11, sigma12, sigma13, sigma14, sigma15, sigma16, sigma17, sigma18, sigma19])
#     amp = np.array([amp1, amp2, amp3, amp4, amp5, amp6, amp7, amp8, amp9, amp10, amp11, amp12, amp13, amp14, amp15, amp16, amp17, amp18, amp19])
#     #beta = np.array([beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8, beta9, beta10, beta11, beta12, beta13, beta14, beta15, beta16, beta17, beta18, beta19])
#
#     single_models = np.zeros((nfib, len(x)))
#     for i in range(nfib):
#         single_models[i,:] = fibmodel(x, mu[i], sigma, beta=beta, norm=0) * amp[i]
#     model = np.sum(single_models, axis=0)
#
#     return model
#
#
    return
