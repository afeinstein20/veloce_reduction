'''
Created on 3 Aug. 2018

@author: christoph
'''

import numpy as np

__all__ =  ['linalg_extract_column', 'mikes_linalg_extraction']


def linalg_extract_column(z, w, phi, RON=3.3, naive_variance=False, altvar=True):
    
    #create diagonal matrix for weights
    ### XXX maybe use sparse matrix here instead to save memory / speed things up
    w_mat = np.diag(w)

    #create the cross-talk matrix
    #C = np.transpose(phi) @ w_mat @ phi
    C = np.matmul( np.matmul(np.transpose(phi),w_mat), phi )
    

    #compute b
    btemp = phi * np.transpose(np.array([z]))     
    #this reshaping is necessary so that z has shape (4096,1) instead of (4096,), which is needed
    # for the broadcasting (ie singelton expansion) functionality
    b = sum( btemp * np.transpose(np.array([w])) )
    #alternatively: b = sum( btemp / (np.transpose(np.array([err**2]))) )
    

    #compute eta (ie the array of the fibre-intensities (or amplitudes)
    ### XXX check if that can be made more efficient
    C_inv = np.linalg.inv(C)
    #eta =  C_inv @ b
    eta = np.matmul(C_inv,b)
    #the following line does the same thing, but contrary to EVERY piece of literature I find it is NOT faster
    #np.linalg.solve(C, b)
    #OR
    #np.linalg.solve(C, b)
    
    #retrieved = eta * phi
    
    if not naive_variance:
        if altvar:
            #THIS CORRESPONDS TO SHARP & BIRCHALL paragraph 5.2.2
            C_prime = np.matmul(np.transpose(phi),phi)
            C_prime_inv = np.linalg.inv(C_prime)
    #        b_prime = sum( phi * np.transpose(np.array([z-RON*RON])) )     #that z here should be the variance actually, but input weights should be from error array that leaves error bars high when cosmics are removed
            b_prime = sum( phi * np.transpose(np.array([(1./w)-RON*RON])) )
            var = np.matmul(C_prime_inv,b_prime)
        else:
            #THIS CORRESPONDS TO SHARP & BIRCHALL paragraph 5.2.1
            T = np.maximum(np.sum(eta * phi, axis=1), 1e-6)
            T = T[:, np.newaxis]    #add extra axis b/c Python...
            fracs = (eta * phi) / T
            var = np.sum(fracs**2 * (1./w)[:, np.newaxis], axis=0)
            #var = np.dot(fracs.T**2 , (1./w)[:,np.newaxis])   # same as line above, but a bit faster
            
    else:
        #these are the "naive errorbars"
        #convert error from variance to proper error
        eta_err = np.sqrt(abs(eta))
        var = eta_err ** 2
        
    #return np.array([eta, eta_err, retrieved])   
    return np.array([eta, var])   





def mikes_linalg_extraction(col_data, col_inv_var, phi, no=19):
    """
    col_data = z
    col_inv_var = w
    """
    
    #Fill in the "c" matrix and "b" vector from Sharp and Birchall equation 9
    #Simplify things by writing the sum in the computation of "b" as a matrix
    #multiplication. We can do this because we're content to invert the 
    #(small) matrix "c" here. Equation 17 from Sharp and Birchall 
    #doesn't make a lot of sense... so lets just calculate the variance in the
    #simple explicit way.
    col_inv_var_mat = np.reshape(col_inv_var.repeat(no), (len(col_data),no) )     #why do the weights have to be the same for every "object"?
    b_mat = phi * col_inv_var_mat
    c_mat = np.dot(phi.T,phi*col_inv_var_mat)
    pixel_weights = np.dot(b_mat,np.linalg.inv(c_mat))   #pixel weights are the z_ki in M.I.'s description
    f = np.dot(col_data,pixel_weights)   #these are the etas
    var = np.dot(1.0/np.maximum(col_inv_var,1e-12),pixel_weights**2)    # CMB: I don't quite understand this, and I think this is wrong actually...;similar to section 5.2.1 in Sharp & Birchall, just not weighted per pixel
    #if ((i % 5)==1) & (j==ny//2):
    #if (i%5==1) & (j==ny//2):
    #if (j==ny//2):
    #    pdb.set_trace()

    return f,var




