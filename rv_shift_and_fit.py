'''
Created on 12 Jul. 2019

@author: christoph
'''
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as op



##########################################################################################
### CMB - 15/11/2017                                                                   ###
### The following functions are based on the RV parts from Mike Ireland's "pymfe"      ###
### I also made them stand-alone routines rather than part of an object-oriented class ###
##########################################################################################    
def rv_shift_resid(parms, wl, f, err, spline_ref, return_spect=False):
    """Find the residuals to a fit of a (subsampled) reference spectrum to an 
    observed spectrum. 
    
    The function for parameters p[0] through p[3] is:
    
    .. math::
        y(x) = Ref[ wave(x) * (1 - p[0]/c) ] * exp(p[1] * x^2 + p[2] * x + p[3])
    
    Here "Ref" is a function f(wave)
    
    Parameters
    ----------        
    params: array-like
    wl: float array
        Wavelengths for the observed spectrum.        
    f: float array
       The observed spectra     
    err: float array
        standard deviation of the input spectra 
    spline_ref: InterpolatedUnivariateSpline instance
        For interpolating the reference spectrum
    return_spect: boolean
        Whether to return the fitted spectrum or the residuals.
    wave_ref: float array
        The wavelengths of the reference spectrum
    ref: float array
        The reference spectrum
    
    Returns
    -------
    resid: float array
        The fit residuals
    """
    
    # speed of light in m/s
    c = 2.99792458e8
    
    ny = len(f)
    
    # CMB change: necessary to make xx go smoothly from -0.5 to 0.5, rather than a step function (step at ny//2) from -1.0 to 0.0      
    #xx = (np.arange(ny)-ny//2)/ny
    xx = (np.arange(ny) - ny//2) / float(ny)
    norm = np.exp(parms[1]*xx*xx + parms[2]*xx + parms[3])     
    
    # Lets get this sign correct. A redshift (positive radial velocity) means that a given wavelength for the reference corresponds to a longer  
    # wavelength for the target, which in turn means that the target wavelength has to be interpolated onto shorter wavelengths for the reference.
#     fitted_spect = spline_ref(wl * (1.0 - parms[0]/c)) * norm
#     fitted_spect = spline_ref(wl * np.sqrt((1 + parms[0] / c) / (1 - parms[0] / c))) * norm   # NOPE! that's the wrong way around...
    fitted_spect = spline_ref(wl * np.sqrt((1 - parms[0] / c) / (1 + parms[0] / c))) * norm
    
    if return_spect:
        return fitted_spect
    else:
        return (fitted_spect - f)/err



def rv_shift_chi2(parms, wl, f, err, spline_ref):
    """Find the chi-squared for an RV fit. Just a wrapper for rv_shift_resid,
    so the docstring is cut and paste!
    
    The function for parameters p[0] through p[3] is:
    
    .. math::
        y(x) = Ref[ wave(x) * (1 - p[0]/c) ] * exp(p[1] * x^2 + p[2] * x + p[3])
    
    Here "Ref" is a function f(wave)
     
    Parameters
    ----------
    
    params: 
        ...
    wl: float array
        Wavelengths for the observed spectrum.
    f: float array
        The observed spectrum
    err: 
        ...
    spline_ref: 
        ...
        
    Returns
    -------
    chi2:
        The fit chi-squared
    """
    return np.sum(rv_shift_resid(parms, wl, f, err, spline_ref)**2)



def rv_shift_jac(parms, wl, f, err, spline_ref):
    """Explicit Jacobian function for rv_shift_resid. 
    
    This is not a completely analytic solution, but without it there seems to be 
    numerical instability.
    
    The key equations are:
    
    .. math:: f(x) = R( \lambda(x)  (1 - p_0/c) ) \times \exp(p_1 x^2 + p_2 x + p_3)
    
       g(x) = (f(x) - d(x))/\sigma(x)
    
       \frac{dg}{dp_0}(x) \approx  [f(x + 1 m/s) -f(x) ]/\sigma(x)
    
       \frac{dg}{dp_1}(x) = x^2 f(x) / \sigma(x)
    
       \frac{dg}{dp_2}(x) = x f(x) / \sigma(x)
    
       \frac{dg}{dp_3}(x) = f(x) / \sigma(x)
    
    Parameters
    ----------
    
    params: float array
    wl: float array
        Wavelengths for the observed spectrum.
    f: float array
        The observed spectrum
    err: float array
        Error in the observed spectrum
    spline_ref: 
        ...
        
    Returns
    -------
    jac: 
        The Jacobian.
    """
    # speed of light in m/s
    c = 2.99792458e8
    
    ny = len(f)
    
    # CMB change: necessary to make xx go smoothly from -0.5 to 0.5, rather than a step function (step at ny//2) from -1.0 to 0.0      
    #xx = (np.arange(ny)-ny//2)/ny
    xx = (np.arange(ny) - ny//2) / float(ny)
    norm = np.exp(parms[1]*xx*xx + parms[2]*xx + parms[3])     
    
    # Lets get this sign correct. A redshift (positive radial velocity) means that a given wavelength for the reference corresponds to a longer  
    # wavelength for the target, which in turn means that the target wavelength has to be interpolated onto shorter wavelengths for the reference.
    fitted_spect = spline_ref(wl * (1.0 - parms[0]/c)) * norm
#     fitted_spect = spline_ref(wave * np.sqrt((1 + parms[0] / c) / (1 - parms[0] / c))) * norm
#     fitted_spect = spline_ref(wave * np.sqrt((1 - parms[0] / c) / (1 + parms[0] / c))) * norm
    
    #The Jacobian is the derivative of fitted_spect/err with respect to p[0] through p[3]
    jac = np.empty((ny,4))
    jac[:,3] = fitted_spect / err
    jac[:,2] = fitted_spect*xx / err
    jac[:,1] = fitted_spect*xx*xx / err     
#     jac[:,0] = (spline_ref(wave*(1.0 - (parms[0] + 1.0)/const.c.si.value))*norm - fitted_spect)/spect_sdev
    jac[:,0] = ((spline_ref(wl * (1.0 - (parms[0] + 1.0)/c)) * norm) - fitted_spect) / err
    
    return jac



def calculate_rv_shift(f0, wl0, f, wl, err, bc=0, bc0=0, return_fitted_spects=False, bad_threshold=10):
    """Calculates the Radial Velocity of each spectrum
    
    The radial velocity shift of the reference spectrum required
    to match the flux in each order for one input spectrum is calculated.
    
    The input flux to this method should be flat-fielded data, which are then fitted with 
    a barycentrically corrected reference spectrum :math:`R(\lambda)`, according to 
    the following equation:

    .. math::
        f(x) = R( \lambda(x)  (1 - p_0/c) ) \\times \exp(p_1 x^2 + p_2 x + p_3)

    The first term in this equation is simply the velocity corrected spectrum, based on a 
    the arc-lamp derived reference wavelength scale :math:`\lambda(x)` for pixels coordinates x.
    The second term in the equation is a continuum normalisation - a shifted Gaussian was 
    chosen as a function that is non-zero everywhere. The scipy.optimize.leastsq function is used
    to find the best fitting set fof parameters :math:`p_0` through to :math`p_3`. 

    The reference spectrum function :math:`R(\lambda)` is created using a wavelength grid 
    which is over-sampled with respect to the data by a factor of 2. Individual fitted 
    wavelengths are then found by cubic spline interpolation on this :math:`R_j(\lambda_j)` 
    discrete grid.
    
    Parameters
    ----------
    wl0: 3D np.array(float)
        Wavelength coordinate map of form (Order, Wavelength/pixel*2+2), 
        where the wavelength scale has been interpolated.
    f0: 3D np.array(float)
        Reference spectrum of form (Order, Flux/pixel*2+2), 
        where the flux scale has been interpolated.
    f: 3D np.array(float)
        Fluxes of form (Order, Fibre, Flux/pixel)
    wl: 3D np.array(float)
        Wavelength coordinate map of form (Order, Fibre, Wavelength/pixel)   
    err: 3D np.array(float)
        Variance of form (Order, Fibre, Error/pixel)    
     

    Returns
    -------
    rvs: 2D np.array(float)
        Radial velocities of format (Observation, Order)
    rv_sigs: 2D np.array(float)
        Radial velocity sigmas of format (Observation, Order)

    TODO:
    oversample the template spectrum before regridding onto bc-shifted wl-grid of observation
    """
    
    nm = f.shape[0]
    nf = f.shape[1]
    npix = f.shape[2]
    
    # speed of light in m/s
    c = 2.99792458e8
    
    # initialise output arrays
    rvs = np.zeros( (nm,nf) )
    rv_sigs = np.zeros( (nm,nf) )
    redchi2_arr = np.zeros( (nm,nf) )
#     thefit_100 = np.zeros( (nm,nf) )

    initp = np.zeros(4)
    initp[3] = 0.5

    if return_fitted_spects:
        fitted_spects = np.empty(f.shape)
    
    # shift wavelength arrays by the respective barycentric corrections
#     wl_bcc = (1 + bc / c) * wave
    wl_bcc = wl * np.sqrt((1 + bc / c) / (1 - bc / c))
#     wl0_bcc = (1 + bc0 / c) * wave_ref
    wl0_bcc = wl0 * np.sqrt((1 + bc0 / c) / (1 - bc0 / c))
    
    print("Order ")
    
    # loop over all orders (skipping first order, as wl-solution is crap!)
    for o in range(1,nm):

        print(str(o+1)),
        # Start with initial guess of no intrinsic RV for the target.
        nbad = 0
        
        # loop over all fibres
        for fib in range(nf):
            
            spl_ref = interp.InterpolatedUnivariateSpline(wl0_bcc[o,fib,::-1], f0[o,fib,::-1], k=3)
            args = (wl_bcc[o,fib,:], f[o,fib,:], err[o,fib,:], spl_ref)
            
            # Remove edge effects in a slightly dodgy way by making their error bars infinity and hence giving them zero weights (20 pixels is about 30km/s) 
            args[2][:20] = np.inf
            args[2][-20:] = np.inf
            
            # calculate the model and residuals starting with initial parameters
            the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True, xtol=1e-6)
            # the_fit[0] are the best-fit parms
            # the_fit[1] is the covariance matrix
            # the_fit[2] is auxiliary information on the fit (in form of a dictionary)
            # the_fit[3] is a string message giving information about the cause of failure.
            # the_fit[4] is an integer flag 
            # (if it is equal to 1, 2, 3 or 4, the solution was found. Otherwise, the solution was not found -- see op.leastsq documentation for more info)
            model_before = rv_shift_resid(the_fit[0], *args, return_spect=True)
            resid_before = rv_shift_resid(the_fit[0], *args)
            wbad = np.where(np.abs(resid_before) > bad_threshold)[0]   # I should probably return the abs residuals and then divide them by the error outside "rv_shift_resid"
            nbad += len(wbad)
            
            chi2 = rv_shift_chi2(the_fit[0], *args)
            redchi2 = chi2 / (npix - len(initp))
            
            try:
#                 errorSigma = np.sqrt(chi2 * the_fit[1][0,0])
##                 normalisedValsBefore = normalised(residBefore, errorSigma)
                
                fitted_spec = rv_shift_resid(the_fit[0], *args, return_spect=True)
                
                # make the errors for the "bad" pixels infinity (so as to make the weights zero)
                args[2][np.where(np.abs(resid_before) > bad_threshold)] = np.inf

                the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True, xtol=1e-6)
#                 residAfter = rv_shift_resid( the_fit[0], *args)
                chi2 = rv_shift_chi2(the_fit[0], *args)
                redchi2 = chi2 / (npix - len(initp))
                
#                 errorSigma = np.sqrt(chi2 * the_fit[1][0,0])
##                 normalisedValsAfter = normalised(residAfter, errorSigma)

                redchi2_arr[o,fib] = redchi2

            except:
                pass

            
            # Some outputs for testing
            if return_fitted_spects:
                fitted_spects[o,fib,:] = rv_shift_resid(the_fit[0], *args, return_spect=True)
            
            # Save the fit and the uncertainty (the_fit[0][0] is the RV shift)
            rvs[o,fib] = the_fit[0][0]

            try:
                rv_sigs[o,fib] = np.sqrt(redchi2 * the_fit[1][0,0])
            except:
                rv_sigs[o,fib] = np.nan
                
#             # the_fit[1] is the covariance matrix
#             try:
#                 thefit_100[o,fib] = the_fit[1][0,0]
#             except:
#                 thefit_100[o,fib] = np.NaN

    if return_fitted_spects:
        return rvs, rv_sigs, redchi2_arr, fitted_spects
    else:
        return rvs, rv_sigs, redchi2_arr
    
    
    
def normalised(residVals, errorSigma):
    return residVals/errorSigma


