'''
Created on 19 Dec. 2018

@author: christoph
'''

import glob
import time
import numpy as np
import matplotlib.pyplot as plt
# import astropy.io.fits as pyfits

from readcol import readcol
from veloce_reduction.veloce_reduction.helper_functions import find_nearest



# path = '/Volumes/BERGRAID/data/veloce/lfc_peaks/'
# path = '/Users/christoph/data/lfc_peaks/'
# path = '/Users/christoph/OneDrive - UNSW/lfc_peaks/'
#
# files = glob.glob(path + '*olc.nst')
#
# id, y0, x0, mag, err_mag, skymod, niter, chi, sharp, y_err, x_err = readcol(files[0], twod=False, skipline=2)
# id, y, x, mag, err_mag, skymod, niter, chi, sharp, y_err, x_err = readcol(files[1], twod=False, skipline=2)
# x0 = 4112. - x0   # flipped
# x = 4112. - x
# y0 = y0 - 54.     # 53 overscan pixels either side and DAOPHOT counting from 1?
# y = y - 54.
#
# test_x0 = x0[(x0 > 1500) & (x0 < 1800) & (y0 > 1500) & (y0 < 1800)]
# test_y0 = y0[(x0 > 1500) & (x0 < 1800) & (y0 > 1500) & (y0 < 1800)]
# test_x = x[(x > 1500) & (x < 1800) & (y > 1500) & (y < 1800)]
# test_y = y[(x > 1500) & (x < 1800) & (y > 1500) & (y < 1800)]
#
# x0 = test_x0.copy()
# x = test_x.copy()
# y0 = test_y0.copy()
# y = test_y.copy()



def find_affine_transformation_matrix(x, y, x0, y0, nx=4112, ny=4202, eps=0.5, wrt='centre', timit=False):
    '''
    Finds the affine transformation matrix that describes the co-ordinate transformation from (x0,y0) --> (x,y)
    
    x0  - reference x
    y0  - reference y
    x   - observed x
    y   - observed y
    nx  - number of pixels in dispersion direction
    ny  - number of pixels in cross-dispersion direction
    eps - tolerance
    wrt - 'corner' or 'centre'

    '''
    
    if timit:
        start_time = time.time()
        
    assert wrt in ['corner', 'centre'], "'wrt' not set correctly!!! Can only use the corner or the centre of the chip as the origin!"
    
    if wrt == 'centre':
        x -= nx//2
        x0 -= nx//2
        y -= ny//2
        y0 -= ny//2
    
    # get numpy arrays from list of (x,y)-tuples in the shape (2,***)
    ref_peaks_xy_list = [(xpos, ypos) for xpos,ypos in zip(x0,y0)]
    ref_peaks_xy = np.array(ref_peaks_xy_list).T     # not really needed
    obs_peaks_xy_list = [(xpos, ypos) for xpos,ypos in zip(x,y)]
    obs_peaks_xy = np.array(obs_peaks_xy_list).T
    
    #now we need to match the peaks
    good_ref_peaks = []
    good_obs_peaks = []
    for n,refpeak in enumerate(ref_peaks_xy_list):
        # print(n)
        shifted_obs_peaks = obs_peaks_xy - np.expand_dims(np.array(refpeak), axis=1)
        distance = np.sqrt(shifted_obs_peaks[0,:]**2 + shifted_obs_peaks[1,:]**2)

        if np.sum(distance < eps) > 0:
            if np.sum(distance < eps) > 1:
                print('FUGANDA: ',refpeak)
                print('There is probably a cosmic really close to an LFC peak - skipping this peak...')
            else:
                good_ref_peaks.append(refpeak)
                good_obs_peaks.append((obs_peaks_xy[0,distance < eps], obs_peaks_xy[1,distance < eps]))

        # print(n, refpeak, np.sum(distance < eps))

    # go to homogeneous coordinates (ie add a z-component equal to 1, so that we can include translation into the matrix)
#     print(np.array(good_ref_peaks).shape)
#     print(np.squeeze(good_ref_peaks).shape)
#     print(np.array(good_obs_peaks).shape)
#     print(np.squeeze(good_ref_peaks).shape)
    good_ref_peaks_xyz = np.hstack((np.array(good_ref_peaks), np.expand_dims(np.repeat(1, len(good_ref_peaks)), axis=1)))
    good_obs_peaks_xyz = np.hstack((np.squeeze(np.array(good_obs_peaks)), np.expand_dims(np.repeat(1, len(good_obs_peaks)), axis=1)))
    
    # np.linalg.lstsq(r1,r2) solves matrix equation M*r1 = r2  (note that linalg.lstsq wants row-vectors)
    # i.e.: good_obs_peaks_xyz ~= np.dot(good_ref_peaks_xyz, M)
#     M, res, rank, s = np.linalg.lstsq(good_ref_peaks_xyz, good_obs_peaks_xyz, rcond=None)  # I don't quite understand what the rconv does...
    M, res, rank, s = np.linalg.lstsq(good_ref_peaks_xyz, good_obs_peaks_xyz, rcond=-1)      # I don't quite understand what the rconv does...

    if timit:
        print('Time elapsed: ' + str(np.round(time.time() - start_time,1)) + ' seconds')
    
    #return affine transformation matrix  (note this is the transpose of the usual form eg as listed in wikipedia)
    return M



#   if   x' = x * M
# then   x  = x' * M_inv



def unfinished_find_manual_transformation_matrix(x, y, x0, y0, nx=4112, ny=4202, eps=0.5, wrt='centre', timit=False):
    '''
    
    NOT IMPLEMENTED YET, JUST A COPY OF "find_affine_transformation_matrix" for now
    
    Finds the transformation matrix that describes the co-ordinate transformation from (x0,y0) --> (x,y), i.e. 
    (1) - a bulk shift in x
    (2) - a bulk shift in y
    (3) - a solid body rotation (theta)
    (4) - a global plate scale S0
    (5) - a scale factor SA in arbitrary direction along an angle A
    
    x0  - reference x
    y0  - reference y
    x   - observed x
    y   - observed y
    nx  - number of pixels in dispersion direction
    ny  - number of pixels in cross-dispersion direction
    eps - tolerance
    wrt - 'corner' or 'centre'

    TODO: find transformation relative to CCD centre, rather than corner!!!

    '''
    
    print('WARNING: NOT IMPLEMENTED YET, JUST A COPY OF "find_affine_transformation_matrix" for now')
    
#     if timit:
#         start_time = time.time()
#         
#     assert wrt in ['corner', 'centre'], "'wrt' not set correctly!!! Can only use the corner or the centre of the chip as the origin!"
#     
#     if wrt == 'centre':
#         x -= nx//2
#         x0 -= nx//2
#         y -= ny//2
#         y0 -= ny//2
#     
#     # get numpy arrays from list of (x,y)-tuples in the shape (2,***)
#     ref_peaks_xy_list = [(xpos, ypos) for xpos,ypos in zip(x0,y0)]
#     ref_peaks_xy = np.array(ref_peaks_xy_list).T     # not really needed
#     obs_peaks_xy_list = [(xpos, ypos) for xpos,ypos in zip(x,y)]
#     obs_peaks_xy = np.array(obs_peaks_xy_list).T
#     
#     #now we need to match the peaks
#     good_ref_peaks = []
#     good_obs_peaks = []
#     for n,refpeak in enumerate(ref_peaks_xy_list):
#         # print(n)
#         shifted_obs_peaks = obs_peaks_xy - np.expand_dims(np.array(refpeak), axis=1)
#         distance = np.sqrt(shifted_obs_peaks[0,:]**2 + shifted_obs_peaks[1,:]**2)
# 
#         if np.sum(distance < eps) > 0:
#             if np.sum(distance < eps) > 1:
#                 print('FUGANDA: ',refpeak)
#                 print('There is probably a cosmic really close to an LFC peak - skipping this peak...')
#             else:
#                 good_ref_peaks.append(refpeak)
#                 good_obs_peaks.append((obs_peaks_xy[0,distance < eps], obs_peaks_xy[1,distance < eps]))
# 
#         # print(n, refpeak, np.sum(distance < eps))
# 
#     # go to homogeneous coordinates (ie add a z-component equal to 1, so that we can include translation into the matrix)
# #     print(np.array(good_ref_peaks).shape)
# #     print(np.squeeze(good_ref_peaks).shape)
# #     print(np.array(good_obs_peaks).shape)
# #     print(np.squeeze(good_ref_peaks).shape)
#     good_ref_peaks_xyz = np.hstack((np.array(good_ref_peaks), np.expand_dims(np.repeat(1, len(good_ref_peaks)), axis=1)))
#     good_obs_peaks_xyz = np.hstack((np.squeeze(np.array(good_obs_peaks)), np.expand_dims(np.repeat(1, len(good_obs_peaks)), axis=1)))
#     
#     # np.linalg.lstsq(r1,r2) solves matrix equation M*r1 = r2  (note that linalg.lstsq wants row-vectors)
#     # i.e.: good_obs_peaks_xyz ~= np.dot(good_ref_peaks_xyz, M)
# #     M, res, rank, s = np.linalg.lstsq(good_ref_peaks_xyz, good_obs_peaks_xyz, rcond=None)  # I don't quite understand what the rconv does...
#     M, res, rank, s = np.linalg.lstsq(good_ref_peaks_xyz, good_obs_peaks_xyz, rcond=-1)      # I don't quite understand what the rconv does...
# 
#     if timit:
#         print('Time elapsed: ' + str(np.round(time.time() - start_time,1)) + ' seconds')
#     
#     #return affine transformation matrix  (note this is the transpose of the usual form eg as listed in wikipedia)
#     return M
    return



def divide_lfc_peaks_into_orders(x, y, tol=5, lfc_path='/Users/christoph/OneDrive - UNSW/lfc_peaks/'):

    # read rough LFC traces
    pid = np.load(lfc_path + 'lfc_P_id.npy').item()

    peaks = {}

    for o in sorted(pid.keys()):
        y_dist = y - pid[o](x)
        peaks[o] = zip(x[np.abs(y_dist) < tol], y[np.abs(y_dist) < tol])

    return peaks



def get_pixel_phase(lfc_file):
    # read observation LFC peak positions from DAOPHOT output files
    try:
        _, y, x, _, _, _, _, _, _, _, _ = readcol(lfc_file, twod=False, skipline=2)
    except:
        _, y, x, _, _, _, _, _, _ = readcol(lfc_file, twod=False, skipline=2)
    del _
    
    x_pixel_phase = x - np.round(x, 0)
    y_pixel_phase = y - np.round(y, 0)
    
    return x_pixel_phase, y_pixel_phase
    
    
    
def check_transformation_scatter_daophot(lfc_files, M_list=None, nx=4112, ny=4202, wrt='centre', n_sub=1, eps=0.5, return_residuals=True,
                                         ref_obsname='21sep30019', return_M_list=False, return_pixel_phase=False, lfc_path = '/Users/christoph/OneDrive - UNSW/lfc_peaks/'):
    # INPUT:
    # 'lfc_files' - list of lfc files for which to check
    # 'M_list'    - corresponding list of calculated transformation matrices

    # WARNING: They obviously need to be in the same order. See how it's done at the start of "check_all_shifts_with_telemetry"

    if M_list is not None:
        assert len(lfc_files) == len(M_list), 'ERROR: list of files and list of matrices have different lengths!!!'
    else:
        M_list_new = []

    # read reference LFC peak positions
    _, yref, xref, _, _, _, _, _, _, _, _ = readcol(lfc_path + ref_obsname + 'olc.nst', twod=False, skipline=2)
    xref = nx - xref
    yref = yref - 54.  # or 53??? but does not matter for getting the transformation matrix
    if wrt == 'centre':
        xref -= nx//2
        yref -= ny//2
    ref_peaks_xy_list = [(xpos, ypos) for xpos, ypos in zip(xref, yref)]

    all_delta_x_list = []
    all_delta_y_list = []
    if return_pixel_phase:
        xphi_list = []
        yphi_list = []

    # loop over all files
    for i,lfc_file in enumerate(lfc_files):

        print('Processing observation ' + str(i+1) + '/' + str(len(lfc_files)) + '...')

        # read observation LFC peak positions
        try:
            _, y, x, _, _, _, _, _, _, _, _ = readcol(lfc_file, twod=False, skipline=2)
        except:
            _, y, x, _, _, _, _, _, _ = readcol(lfc_file, twod=False, skipline=2)
        del _
        x = nx - x
        y = y - 54.  # or 53??? but does not matter for getting the transformation matrix
        if wrt == 'centre':
            x -= nx//2
            y -= ny//2
        obs_peaks_xy_list = [(xpos, ypos) for xpos, ypos in zip(x, y)]
        obs_peaks_xy = np.array(obs_peaks_xy_list).T

        if M_list is None:
            # NOTE that we do not want to shift to the centre twice, so we hard-code 'corner' here!!! (xref, yref, x, y) are already transformed above!!!
            M = find_affine_transformation_matrix(xref, yref, x, y, timit=True, eps=2., wrt='corner')   # note that within "wavelength_solution" this is called "Minv"
            M_list_new.append(M)
        else:
            M = M_list[i]

        # now we need to match the peaks so we can compare the reference peaks with the (back-)transformed obs peaks
        good_ref_peaks = []
        good_obs_peaks = []
        for n, refpeak in enumerate(ref_peaks_xy_list):
            # print(n)
            shifted_obs_peaks = obs_peaks_xy - np.expand_dims(np.array(refpeak), axis=1)
            distance = np.sqrt(shifted_obs_peaks[0, :] ** 2 + shifted_obs_peaks[1, :] ** 2)

            if np.sum(distance < eps) > 0:
                if np.sum(distance < eps) > 1:
                    print('FUGANDA: ', refpeak)
                    print('There is probably a cosmic really close to an LFC peak - skipping this peak...')
                else:
                    good_ref_peaks.append(refpeak)
                    good_obs_peaks.append((obs_peaks_xy[0, distance < eps], obs_peaks_xy[1, distance < eps]))
            # print(n, refpeak, np.sum(distance < eps))

        # calculate pixel phase as defined by Anderson & King, 2000, PASP, 112:1360
        if return_pixel_phase:
            x_pixel_phase = np.squeeze(good_obs_peaks)[:,0] - np.round(np.squeeze(good_obs_peaks)[:,0], 0)
            y_pixel_phase = np.squeeze(good_obs_peaks)[:,1] - np.round(np.squeeze(good_obs_peaks)[:,1], 0)

        # divide good_ref_peaks into several subsections for a more detailed investigation
        x_step = nx / np.sqrt(n_sub).astype(int)
        y_step = ny / np.sqrt(n_sub).astype(int)
        x_centres = np.arange(0.5 * x_step, (np.sqrt(n_sub).astype(int) + 0.5) * x_step, x_step)
        y_centres = np.arange(0.5 * y_step, (np.sqrt(n_sub).astype(int) + 0.5) * y_step, y_step)
        if wrt == 'centre':
            x_centres -= nx//2
            y_centres -= ny//2
        peak_subsection_id = []
        for refpeak in good_ref_peaks:
            # first, figure out which subsection this particular peak falls into
            xpos = refpeak[0]
            ypos = refpeak[1]
            nearest_x_ix = find_nearest(x_centres, xpos, return_index=True)
            nearest_y_ix = find_nearest(y_centres, ypos, return_index=True)
            # then save that information
            peak_subsection_id.append((nearest_x_ix, nearest_y_ix))

        # give each subsection a label
        subsection_id = []
        for j in range(np.sqrt(n_sub).astype(int)):
            for i in range(np.sqrt(n_sub).astype(int)):
                subsection_id.append((i, j))  # (x,y)

        # # divide chip into several subsections for a more detailed investigation
        # section_masks = []
        # section_indices = []
        # x_step = nx / np.sqrt(n_sub).astype(int)
        # y_step = ny / np.sqrt(n_sub).astype(int)
        # for j in range(np.sqrt(n_sub).astype(int)):
        #     for i in range(np.sqrt(n_sub).astype(int)):
        #         q = np.zeros((ny, nx), dtype='bool')
        #         q[j * y_step : (j+1) * y_step, i * x_step : (i+1) * x_step] = True
        #         section_masks.append(q)
        #         section_indices.append((i,j))   # (x,y)

        # go to homogeneous coordinates (ie add a z-component equal to 1, so that we can include translation into the matrix)
        good_ref_peaks_xyz = np.hstack((np.array(good_ref_peaks), np.expand_dims(np.repeat(1, len(good_ref_peaks)), axis=1)))
        good_obs_peaks_xyz = np.hstack((np.squeeze(np.array(good_obs_peaks)), np.expand_dims(np.repeat(1, len(good_obs_peaks)), axis=1)))

        # calculate transformed co-ordinates (ie the observed peaks transformed back to match the reference peaks)
        xyz_prime = np.dot(good_obs_peaks_xyz, M)
        delta_x = good_ref_peaks_xyz[:, 0] - xyz_prime[:, 0]
        delta_y = good_ref_peaks_xyz[:, 1] - xyz_prime[:, 1]

        delta_x_list = []
        delta_y_list = []
        # loop over all subsections
        for tup in subsection_id:
            # find indices of peaks falling in each subsection
            ix = [i for i, x in enumerate(peak_subsection_id) if x == tup]
            if return_residuals:
                delta_x_list.append(delta_x[ix])
                delta_y_list.append(delta_y[ix])
            else:
                # return difference between ref and obs
                delta_x_list.append(good_ref_peaks_xyz[ix,0] - good_obs_peaks_xyz[ix,0])
                delta_y_list.append(good_ref_peaks_xyz[ix,1] - good_obs_peaks_xyz[ix,1])
                
                
#####   DO THIS IF YOU WANT TO GET A TRANSFORMATION MATRIX FOR EVERY SUBSECTION OF THE CHIP   #####
#                 M = find_affine_transformation_matrix(np.squeeze(good_ref_peaks)[ix,0], np.squeeze(good_ref_peaks)[ix,1], np.squeeze(good_obs_peaks)[ix,0], np.squeeze(good_obs_peaks)[ix,1], timit=True, eps=2., wrt='corner')
#                 good_ref_peaks_xyz = np.hstack((np.array(good_ref_peaks)[ix], np.expand_dims(np.repeat(1, len(ix)), axis=1)))
#                 good_obs_peaks_xyz = np.hstack((np.squeeze(np.array(good_obs_peaks)[ix]), np.expand_dims(np.repeat(1, len(ix)), axis=1)))
#                 xyz_prime = np.dot(good_obs_peaks_xyz, M)
#                 delta_x = good_ref_peaks_xyz[:, 0] - xyz_prime[:, 0]
#                 delta_y = good_ref_peaks_xyz[:, 1] - xyz_prime[:, 1]
#                 plt.plot(delta_x,'.')
#                 print(np.std(delta_x))
#                 sub_M_list.append(M)

        # append to all-files list
        all_delta_x_list.append(delta_x_list)
        all_delta_y_list.append(delta_y_list)

    if M_list is None:
        M_list = M_list_new[:]

    if return_pixel_phase:
        if not return_M_list:
            return all_delta_x_list, all_delta_y_list, x_pixel_phase, y_pixel_phase
        else:
            return all_delta_x_list, all_delta_y_list, x_pixel_phase, y_pixel_phase, M_list
    else:
        if not return_M_list:
            return all_delta_x_list, all_delta_y_list
        else:
            return all_delta_x_list, all_delta_y_list, M_list



def unfinished_check_transformation_scatter_xcorr(lfc_surface_files, M_list=None, nx=4112, ny=4202, n_sub=1, eps=0.5, return_residuals=True,
                                       ref_obsname='21sep30019', lfc_surface_path = '/Users/christoph/OneDrive - UNSW/dispsol/laser_offsets/relto_21sep30019/'):
    # INPUT:
    # 'lfc_surface_files' - list of files containing the shift, slope and 2nd order coefficients of Duncan's xcorr surfaces
    # 'M_list'    - corresponding list of calculated transformation matrices

    # WARNING: They obviously need to be in the same order. See how it's done at the start of "check_all_shifts_with_telemetry"

    print('WARNING: still under development!!!')

#     if M_list is not None:
#         assert len(lfc_files) == len(M_list), 'ERROR: list of files and list of matrices have different lengths!!!'
# 
#     # read reference LFC peak positions
#     _, yref, xref, _, _, _, _, _, _, _, _ = readcol(lfc_path + ref_obsname + 'olc.nst', twod=False, skipline=2)
#     xref = nx - xref
#     yref = yref - 54.  # or 53??? but does not matter for getting the transformation matrix
#     ref_peaks_xy_list = [(xpos, ypos) for xpos, ypos in zip(xref, yref)]
# 
#     all_delta_x_list = []
#     all_delta_y_list = []
# 
#     # loop over all files
#     for i,lfc_file in enumerate(lfc_files):
# 
#         print('Processing observation ' + str(i+1) + '/' + str(len(lfc_files)) + '...')
# 
#         # read observation LFC peak positions
#         try:
#             _, y, x, _, _, _, _, _, _, _, _ = readcol(lfc_file, twod=False, skipline=2)
#         except:
#             _, y, x, _, _, _, _, _, _ = readcol(lfc_file, twod=False, skipline=2)
#         del _
#         x = nx - x
#         y = y - 54.  # or 53??? but does not matter for getting the transformation matrix
#         obs_peaks_xy_list = [(xpos, ypos) for xpos, ypos in zip(x, y)]
#         obs_peaks_xy = np.array(obs_peaks_xy_list).T
# 
#         if M_list is None:
#             M = find_affine_transformation_matrix(xref, yref, x, y, timit=True, eps=2.)   # note that within "wavelength_solution" this is called Minv
#         else:
#             M = M_list[i]
# 
#         # now we need to match the peaks so we can compare the reference peaks with the (back-)transformed obs peaks
#         good_ref_peaks = []
#         good_obs_peaks = []
#         for n, refpeak in enumerate(ref_peaks_xy_list):
#             # print(n)
#             shifted_obs_peaks = obs_peaks_xy - np.expand_dims(np.array(refpeak), axis=1)
#             distance = np.sqrt(shifted_obs_peaks[0, :] ** 2 + shifted_obs_peaks[1, :] ** 2)
# 
#             if np.sum(distance < eps) > 0:
#                 if np.sum(distance < eps) > 1:
#                     print('FUGANDA: ', refpeak)
#                     print('There is probably a cosmic really close to an LFC peak - skipping this peak...')
#                 else:
#                     good_ref_peaks.append(refpeak)
#                     good_obs_peaks.append((obs_peaks_xy[0, distance < eps], obs_peaks_xy[1, distance < eps]))
#             # print(n, refpeak, np.sum(distance < eps))
# 
#         # divide good_ref_peaks into several subsections for a more detailed investigation
#         x_step = nx / np.sqrt(n_sub).astype(int)
#         y_step = ny / np.sqrt(n_sub).astype(int)
#         x_centres = np.arange(0.5 * x_step, (np.sqrt(n_sub).astype(int) + 0.5) * x_step, x_step)
#         y_centres = np.arange(0.5 * y_step, (np.sqrt(n_sub).astype(int) + 0.5) * y_step, y_step)
#         peak_subsection_id = []
#         for refpeak in good_ref_peaks:
#             # first, figure out which subsection this particular peak falls into
#             xpos = refpeak[0]
#             ypos = refpeak[1]
#             nearest_x_ix = find_nearest(x_centres, xpos, return_index=True)
#             nearest_y_ix = find_nearest(y_centres, ypos, return_index=True)
#             # then save that information
#             peak_subsection_id.append((nearest_x_ix, nearest_y_ix))
# 
#         # give each subsection a label
#         subsection_id = []
#         for j in range(np.sqrt(n_sub).astype(int)):
#             for i in range(np.sqrt(n_sub).astype(int)):
#                 subsection_id.append((i, j))  # (x,y)
# 
#         # # divide chip into several subsections for a more detailed investigation
#         # section_masks = []
#         # section_indices = []
#         # x_step = nx / np.sqrt(n_sub).astype(int)
#         # y_step = ny / np.sqrt(n_sub).astype(int)
#         # for j in range(np.sqrt(n_sub).astype(int)):
#         #     for i in range(np.sqrt(n_sub).astype(int)):
#         #         q = np.zeros((ny, nx), dtype='bool')
#         #         q[j * y_step : (j+1) * y_step, i * x_step : (i+1) * x_step] = True
#         #         section_masks.append(q)
#         #         section_indices.append((i,j))   # (x,y)
# 
#         # go to homogeneous coordinates (ie add a z-component equal to 1, so that we can include translation into the matrix)
#         good_ref_peaks_xyz = np.hstack((np.array(good_ref_peaks), np.expand_dims(np.repeat(1, len(good_ref_peaks)), axis=1)))
#         good_obs_peaks_xyz = np.hstack((np.squeeze(np.array(good_obs_peaks)), np.expand_dims(np.repeat(1, len(good_obs_peaks)), axis=1)))
# 
#         # calculate transformed co-ordinates (ie the observed peaks transformed back to match the reference peaks)
#         xyz_prime = np.dot(good_obs_peaks_xyz, M)
#         delta_x = good_ref_peaks_xyz[:, 0] - xyz_prime[:, 0]
#         delta_y = good_ref_peaks_xyz[:, 1] - xyz_prime[:, 1]
# 
#         delta_x_list = []
#         delta_y_list = []
#         # loop over all subsections
#         for tup in subsection_id:
#             # find indices of peaks falling in each subsection
#             ix = [i for i, x in enumerate(peak_subsection_id) if x == tup]
#             if return_residuals:
#                 delta_x_list.append(delta_x[ix])
#                 delta_y_list.append(delta_y[ix])
#             else:
#                 # return difference between ref and obs
#                 delta_x_list.append(good_ref_peaks_xyz[ix,0] - good_obs_peaks_xyz[ix,0])
#                 delta_y_list.append(good_ref_peaks_xyz[ix,1] - good_obs_peaks_xyz[ix,1])
# 
#         # append to all-files list
#         all_delta_x_list.append(delta_x_list)
#         all_delta_y_list.append(delta_y_list)
# 
#     return all_delta_x_list, all_delta_y_list
    return



def vector_plot(dx, dy, plotval='med'):
    '''
    dx, dy are outputs from "check_transformation_scatter()"
    'plotval' : either 'med(ian)' or 'mean'
    '''
    
    n_sub = len(dx[0])
    
    if plotval.lower() in ['med', 'median']:
        print('OK, plotting the median of the residuals per subsection...')
        x = [np.median(dx[0][i]) for i in range(n_sub)]
        y = [np.median(dy[0][i]) for i in range(n_sub)]
    elif plotval.lower() == 'mean':
        print('OK, plotting the mean of the residuals per subsection...')
        x = [np.mean(dx[0][i]) for i in range(n_sub)]
        y = [np.mean(dy[0][i]) for i in range(n_sub)]
    else:
        print('FAIL!!!')
        return
    
    # give each subsection a label
    subsection_id = []
    for j in range(np.sqrt(n_sub).astype(int)):
        for i in range(np.sqrt(n_sub).astype(int)):
            subsection_id.append((i, j))  # (x-axis, y-axis)
    
    plt.figure()
    X = np.arange(np.sqrt(n_sub).astype(int)) + 0.5
    Y = np.arange(np.sqrt(n_sub).astype(int)) + 0.5
    U = np.reshape(x, (np.sqrt(n_sub).astype(int), np.sqrt(n_sub).astype(int)))
    V = np.reshape(y, (np.sqrt(n_sub).astype(int), np.sqrt(n_sub).astype(int)))
    plt.quiver(X, Y, U, V)
    
    return
    
    
    
