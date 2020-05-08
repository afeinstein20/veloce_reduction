'''
Created on 19 Dec. 2018

@author: christoph
'''

import glob
import time
import numpy as np
import matplotlib.pyplot as plt

from veloce_reduction.readcol import readcol
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
    
    



def lfc_peak_diffs(ref_obsname='25jun30256', ref_date='20190625', ref_year='2019', degpol=7, nx=4112, ny=4096):

    """check LFC shifts from DAOPHOT files"""

    from readcol import readcol
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    from veloce_reduction.veloce_reduction.lfc_peaks import divide_lfc_peaks_into_orders
    from veloce_reduction.veloce_reduction.helper_functions import find_nearest
    from veloce_reduction.veloce_reduction.helper_functions import sigma_clip, single_sigma_clip

    red_path = '/Volumes/BERGRAID/data/veloce/reduced/'
    root = '/Users/christoph/OneDrive - UNSW/'
    lfc_path = '/Volumes/BERGRAID/data/veloce/lfc_peaks/'
    outpath = root + 'dispsol_tests/diff_tests/'

    # read in LFC vac wls (f0 = 9.56 GHz   &   f_rep = 25 GHz)
    lfc_vac_wls = np.squeeze(np.array(readcol(root + 'dispsol/lfc_vac_wls_nm.txt', twod=False))) * 10.    # in Angstroms

    # read reference LFC peak positions
    try:
        _, yref, xref, _, _, _, _, _, _, _, _ = readcol(lfc_path + 'all/' + ref_year + '/' + ref_obsname + 'olc.nst', twod=False, skipline=2)
    except:
        _, yref, xref, _, _, _, _, _, _ = readcol(lfc_path + 'all/' + ref_year + '/' + ref_obsname + 'olc.nst', twod=False, skipline=2)
    xref = nx - xref
    yref = yref - 54.  # or 53??? but does not matter for getting the transformation matrix
    ref_peaks_xy_list = [(xpos, ypos) for xpos, ypos in zip(xref, yref)]
    ref_peaks = divide_lfc_peaks_into_orders(xref, yref)
    ref_vac_wl = pyfits.getdata(red_path + ref_date + '/' + ref_date + '_thxe_dispsol.fits', 1)

    # read in list of files to be investigated from file
    obsnames, dates = readcol(outpath + 'filelist.txt', twod=False)

    # loop over all observations to analyze
    for obsname, obs_date in zip(obsnames, dates):
        fail = False
        print('Cross-matching LFC peaks for ' + obsname)

        obs_date = str(obs_date)
        year = obs_date[:4]
        obsname = obsname.split('.')[0]

        # read obs. LFC peak positions
        if os.path.isfile(lfc_path + 'all/' + year + '/' + obsname + 'olc.nst'):
            try:
                _, y, x, _, _, _, _, _, _, _, _ = readcol(lfc_path + 'all/' + year + '/' + obsname + 'olc.nst', twod=False, skipline=2)
            except:
                _, y, x, _, _, _, _, _, _ = readcol(lfc_path + 'all/' + year + '/' + obsname + 'olc.nst', twod=False, skipline=2)
            x = nx - x
            y = y - 54.  # or 53??? but does not matter for getting the transformation matrix
            obs_peaks_xy_list = [(xpos, ypos) for xpos, ypos in zip(x, y)]
            obs_peaks = divide_lfc_peaks_into_orders(x, y)
            obs_vac_wl = pyfits.getdata(red_path + obs_date + '/' + obs_date + '_thxe_dispsol.fits', 1)
        else:
            print('ERROR: no LFC peaks measured for ' + obsname)
            fail = True

        # prepare arrays for 3D-plotting
        x_points = []
        y_points = []
        zx_points = []
        zy_points = []
        ord_points = []

        if not fail:
            # LOOP OVER ORDERS
            for ord in sorted(ref_peaks.keys()):
                # print('Processing ' + ord)
                o = int(ord[-2:])-1

                # divide DAOPHOT LFC peaks into orders
                ord_xref = np.array(ref_peaks[ord])[:,0]
                ord_yref = np.array(ref_peaks[ord])[:,1]
                ord_yref = ord_yref[ord_xref.argsort()]
                ord_xref = ord_xref[ord_xref.argsort()]
                ord_x = np.array(obs_peaks[ord])[:,0]
                ord_y = np.array(obs_peaks[ord])[:,1]
                ord_y = ord_y[ord_x.argsort()]
                ord_x = ord_x[ord_x.argsort()]

                # this just recovers the 7th order polynomial used in the 2-dim ThXe dispsol for the fibre closest to the LFC for this order (almost perfect)
                xx = np.arange(nx)
                ref_ord_wlfit = np.poly1d(np.polyfit(xx, ref_vac_wl[o, -2, :], degpol))
                obs_ord_wlfit = np.poly1d(np.polyfit(xx, obs_vac_wl[o, -2, :], degpol))
                ref_peak_thxe_wls = ref_ord_wlfit(ord_xref)
                obs_peak_thxe_wls = obs_ord_wlfit(ord_x)

                # find the nearest entries in the array of theoretical wavelengths for each peak based on the ThXe dispsol
                maxwl = np.max(ref_ord_wlfit(xx))
                minwl = np.min(ref_ord_wlfit(xx))
                ord_lfc_vac_wls = lfc_vac_wls[(lfc_vac_wls >= minwl) & (lfc_vac_wls <= maxwl)]
                thresh = (1./3.) * np.max(np.abs(np.diff(ord_lfc_vac_wls)))
                refpeaks_theo_wls = np.array([find_nearest(lfc_vac_wls, ref_ord_wlfit(peak_x)) for peak_x in ord_xref])
                obspeaks_theo_wls = np.array([find_nearest(lfc_vac_wls, obs_ord_wlfit(peak_x)) for peak_x in ord_x])
                refdiffs = np.abs(np.array(refpeaks_theo_wls) - ref_peak_thxe_wls)
                obsdiffs = np.abs(np.array(obspeaks_theo_wls) - obs_peak_thxe_wls)
                refpeaks_theo_wls = refpeaks_theo_wls[refdiffs < thresh]
                ord_xref = ord_xref[refdiffs < thresh]
                ord_yref = ord_yref[refdiffs < thresh]
                obspeaks_theo_wls = obspeaks_theo_wls[obsdiffs < thresh]
                ord_x = ord_x[obsdiffs < thresh]
                ord_y = ord_y[obsdiffs < thresh]

                # remove obvious outliers from reference observation (in both x-y-fit, and x-lambda-fit)
                # in the x-y-space
                pix_fit = np.poly1d(np.polyfit(ord_xref, ord_yref, degpol))
                pixres = ord_yref - pix_fit(ord_xref)
                clipped, goodboolix, goodix, badix = single_sigma_clip(pixres, 3, return_indices=True)
                new_outies = len(badix)
                while new_outies > 0:
                    ord_xref = ord_xref[goodboolix]
                    ord_yref = ord_yref[goodboolix]
                    refpeaks_theo_wls = refpeaks_theo_wls[goodboolix]
                    pix_fit = np.poly1d(np.polyfit(ord_xref, ord_yref, degpol))
                    pixres = ord_yref - pix_fit(ord_xref)
                    clipped, goodboolix, goodix, badix = single_sigma_clip(pixres, 3, return_indices=True)
                    new_outies = len(badix)
                # in the x-lambda-space
                lam_fit = np.poly1d(np.polyfit(ord_xref, refpeaks_theo_wls, degpol))
                wlres = refpeaks_theo_wls - lam_fit(ord_xref)
                clipped, goodboolix, goodix, badix = single_sigma_clip(wlres, 5, return_indices=True)
                new_outies = len(badix)
                while new_outies > 0:
                    ord_xref = ord_xref[goodboolix]
                    ord_yref = ord_yref[goodboolix]
                    refpeaks_theo_wls = refpeaks_theo_wls[goodboolix]
                    lam_fit = np.poly1d(np.polyfit(ord_xref, refpeaks_theo_wls, degpol))
                    wlres = refpeaks_theo_wls - lam_fit(ord_xref)
                    clipped, goodboolix, goodix, badix = single_sigma_clip(wlres, 5, return_indices=True)
                    new_outies = len(badix)

                # remove obvious outliers from observation
                # in the x-y-space
                pix_fit = np.poly1d(np.polyfit(ord_x, ord_y, degpol))
                pixres = ord_y - pix_fit(ord_x)
                clipped, goodboolix, goodix, badix = single_sigma_clip(pixres, 3, return_indices=True)
                new_outies = len(badix)
                while new_outies > 0:
                    ord_x = ord_x[goodboolix]
                    ord_y = ord_y[goodboolix]
                    obspeaks_theo_wls = obspeaks_theo_wls[goodboolix]
                    pix_fit = np.poly1d(np.polyfit(ord_x, ord_y, degpol))
                    pixres = ord_y - pix_fit(ord_x)
                    clipped, goodboolix, goodix, badix = single_sigma_clip(pixres, 3, return_indices=True)
                    new_outies = len(badix)
                # in the x-lambda-space
                lam_fit = np.poly1d(np.polyfit(ord_x, obspeaks_theo_wls, degpol))
                wlres = obspeaks_theo_wls - lam_fit(ord_x)
                clipped, goodboolix, goodix, badix = single_sigma_clip(wlres, 5, return_indices=True)
                new_outies = len(badix)
                while new_outies > 0:
                    ord_x = ord_x[goodboolix]
                    ord_y = ord_y[goodboolix]
                    obspeaks_theo_wls = obspeaks_theo_wls[goodboolix]
                    lam_fit = np.poly1d(np.polyfit(ord_x, obspeaks_theo_wls, degpol))
                    wlres = obspeaks_theo_wls - lam_fit(ord_x)
                    clipped, goodboolix, goodix, badix = single_sigma_clip(wlres, 5, return_indices=True)
                    new_outies = len(badix)

                # remove duplicate entries from the theo wls, which must correspond to false positive peaks (but we don't know which one, so let's remove all occurrences)
                refpeaks_theo_wls = list(refpeaks_theo_wls)
                ord_xref = list(ord_xref)
                ord_yref = list(ord_yref)
                obspeaks_theo_wls = list(obspeaks_theo_wls)
                ord_x = list(ord_x)
                ord_y = list(ord_y)
                if len(refpeaks_theo_wls) != len(set(refpeaks_theo_wls)):
                    dups = set([dumx for dumx in refpeaks_theo_wls if list(refpeaks_theo_wls).count(dumx) > 1])
                    dupix_ll = [np.squeeze(np.argwhere(refpeaks_theo_wls == dup)).tolist() for dup in dups]
                    dupix = [item for sublist in dupix_ll for item in sublist]
                    for ix in dupix:
                        del refpeaks_theo_wls[ix]
                        del ord_xref[ix]
                        del ord_yref[ix]
                if len(obspeaks_theo_wls) != len(set(obspeaks_theo_wls)):
                    dups = set([dumx for dumx in obspeaks_theo_wls if list(obspeaks_theo_wls).count(dumx) > 1])
                    dupix_ll = [np.squeeze(np.argwhere(obspeaks_theo_wls == dup)).tolist() for dup in dups]
                    dupix = [item for sublist in dupix_ll for item in sublist]
                    for ix in dupix:
                        del obspeaks_theo_wls[ix]
                        del ord_x[ix]
                        del ord_y[ix]

                # cross-match the two peak lists
                matching_ord_xref = [xdum for xdum,rp in zip(ord_xref,refpeaks_theo_wls) if rp in obspeaks_theo_wls]
                matching_ord_yref = [ydum for ydum,rp in zip(ord_yref,refpeaks_theo_wls) if rp in obspeaks_theo_wls]
                matching_ord_x = [xdum for xdum,op in zip(ord_x,obspeaks_theo_wls) if op in refpeaks_theo_wls]
                matching_ord_y = [ydum for ydum,op in zip(ord_y,obspeaks_theo_wls) if op in refpeaks_theo_wls]
                assert len(matching_ord_xref) == len(matching_ord_yref), 'ERROR: len(matching_ord_xref) != len(matching_ord_yref)'
                assert len(matching_ord_x) == len(matching_ord_y), 'ERROR: len(matching_ord_x) != len(matching_ord_y)'
                # assert len(matching_ord_xref) == len(matching_ord_x), 'ERROR: len(matching_ord_xref) != len(matching_ord_x)'
                if len(matching_ord_xref) != len(matching_ord_x):
                    print('ERROR: len(matching_ord_xref) != len(matching_ord_x)')
                    print('Skipping this file...')
                    fail=True
                else:
                    # now we can finally compare them
                    xdiff = np.array(matching_ord_x) - np.array(matching_ord_xref)
                    ydiff = np.array(matching_ord_y) - np.array(matching_ord_yref)

                    # fill output arrays for 3D-plotting
                    x_points.append(matching_ord_xref)
                    y_points.append(matching_ord_yref)
                    ord_points.append(list((np.ones(len(matching_ord_xref)) * (o+1)).astype(int)))
                    zx_points.append(list(xdiff))
                    zy_points.append(list(ydiff))

        if not fail:
            # now bring to 1-dim format for 3D-plotting
            x_points = np.sum(x_points)
            y_points = np.sum(y_points)
            zx_points = np.sum(zx_points)
            zy_points = np.sum(zy_points)
            ord_points = np.sum(ord_points)

            # # PLOT
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter3D(x_points, ord_points, zx_points, marker='.', s=0.5, color='b')
            # ax.set_xlabel('pixel')
            # ax.set_ylabel('order')
            # ax.set_zlabel('pixel shift')
            # plt.title(obsname)

            # save diffs to output file
            print('Saving results for a total of ' + str(len(ord_points))+ ' LFC peaks!')
            results = np.array([ord_points, x_points, y_points, zx_points, zy_points]).T
            outfn = outpath + obsname + '_lfc_peak_diffs.txt'
            np.savetxt(outfn, results, fmt='%2i   %4.8f   %4.8f   %4.8f   %4.8f')

    return