'''
Created on 13 Dec. 2018

@author: christoph
'''


from readcol import readcol
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import glob
import astropy.io.fits as pyfits
import os

from veloce_reduction.veloce_reduction.lfc_peaks import find_affine_transformation_matrix, divide_lfc_peaks_into_orders
from veloce_reduction.veloce_reduction.helper_functions import find_nearest


def telemetry():
    path = '/Users/christoph/OneDrive - UNSW/telemetry/'
    
    # # tau Ceti observation times
    # jd_sep = readcol('/Volumes/BERGRAID/data/veloce/reduced/tauceti/tauceti_with_LFC/sep2018/' + 'tauceti_all_jds.dat')
    # jd_nov = readcol('/Volumes/BERGRAID/data/veloce/reduced/tauceti/tauceti_with_LFC/nov2018/' + 'tauceti_all_jds.dat')
    # jd = list(jd_sep) + list(jd_nov)
    
    # # read detector temperature file(s)
    # with open(path + 'arccamera.txt') as det_temp_file:
    #     for line in det_temp_file:
    #         if line[0] == '#':
    #             column_names = line.lstrip('#').split()
    #         else:
    #             timestamp, det_temp, cryohead_temp, perc_heater = line.rstrip('\n')
            
    
    timestamp_cam, det_temp, cryohead_temp, perc_heater = readcol(path + 'arccamera.txt', skipline=1, twod=False)
    timestamp_cam_0, det_temp_0, cryohead_temp_0, perc_heater_0 = readcol(path + 'arccamera.txt.0', skipline=1, twod=False)
    timestamp_cam_1, det_temp_1, cryohead_temp_1, perc_heater_1 = readcol(path + 'arccamera.txt.1', skipline=1, twod=False)
    timestamp_cam_2, det_temp_2, cryohead_temp_2, perc_heater_2 = readcol(path + 'arccamera.txt.2', skipline=1, twod=False)
    timestamp_cam_3, det_temp_3, cryohead_temp_3, perc_heater_3 = readcol(path + 'arccamera.txt.3', skipline=1, twod=False)
    timestamp_cam_4, det_temp_4, cryohead_temp_4, perc_heater_4 = readcol(path + 'arccamera.txt.4', skipline=1, twod=False)
    
    cam_time = Time(timestamp_cam, format='unix')         
    cam_time_0 = Time(timestamp_cam_0, format='unix')     
    cam_time_1 = Time(timestamp_cam_1, format='unix')
    cam_time_2 = Time(timestamp_cam_2, format='unix')
    cam_time_3 = Time(timestamp_cam_3, format='unix')
    cam_time_4 = Time(timestamp_cam_4, format='unix')
    
    # combine all the subsets
    t = np.array(list(cam_time_4.jd) + list(cam_time_3.jd) + list(cam_time_2.jd) + list(cam_time_1.jd) + list(cam_time_0.jd) + list(cam_time.jd)) - 2.458e6
    tobj = Time(t + 2.458e6, format='jd')  
    dtemp = np.array(list(det_temp_4) + list(det_temp_3) + list(det_temp_2) + list(det_temp_1) + list(det_temp_0) + list(det_temp))
    cryo = np.array(list(cryohead_temp_4) + list(cryohead_temp_3) + list(cryohead_temp_2) + list(cryohead_temp_1) + list(cryohead_temp_0) + list(cryohead_temp))
    heater_load = np.array(list(perc_heater_4) + list(perc_heater_3) + list(perc_heater_2) + list(perc_heater_1) + list(perc_heater_0) + list(perc_heater))

    # # for showing tau ceti obstimes
    # ix = np.zeros(len(jd))
    # # I checked, and the maximum diff in time is less than ~30s :)
    # for i in range(len(jd)):
    #     ix[i] = find_nearest(t, jd[i] - 2.458e6, return_index=True)
    # # now len(ix) = 138, whereas len(x_shift = 133, b/c the LFC peaks were not successfully measured for the first 5 tau ceti observations;
    # # hence cut away first 5 entries
    # ix = ix[5:]
    # ix = ix.astype(int)
    
        
        

    # observing runs were:
    # 20180917 - 20180926
    tstart_sep18 = 2458378.0
    tend_sep18 = 2458388.0
    # 20181115 - 20181127
    tstart_nov18 = 2458437.0
    tend_nov18 = 2458450.0
    # 20190121 - 20190203
    tstart_janfeb1819 = 2458504.0
    tend_janfeb1819 = 2458518.0
    # 20190408 - 20190415
    tstart_apr19 = 2458581.0
    tend_apr19 = 2458589.0
    # 20190503 - 20190512
    tstart_may19_01 = 2458606.0
    tend_may19_01 = 2458616.0
    # 20190517 - 20190528
    tstart_may19_02 = 2458620.0
    tend_may19_02 = 2458632.0
    # 20190531 - 20190605
    tstart_jun19_01 = 2458634.0
    tend_jun19_01 = 2458640.0
    # 20190619 - 20190625
    tstart_jun19_02 = 2458653.0
    tend_jun19_02 = 2458660.0
    # 20190722 - 20190724
    tstart_jul19 = 2458686.0
    tend_jul19 = 2458689.0
    # 20190805 - 20190826
    tstart_aug19 = 2458700.0
    tend_aug19 = 2458722.0
    # 20190909 - 20190919
    tstart_sep19 = 2458735.0
    tend_sep19 = 2458746.0
    # 20190929 - 20191105
    tstart_sepoctnov19 = 2458755.0
    tend_sepoctnov19 = 2458793.0
    # 20191210 - 20200113
    tstart_decjan1920 = 2458827.0
    tend_decjan1920 = 2458862.0
    
    starts = np.array([tstart_sep18, tstart_nov18, tstart_janfeb1819, tstart_apr19, tstart_may19_01, tstart_may19_02, tstart_jun19_01, tstart_jun19_02, \
                       tstart_jul19, tstart_aug19, tstart_sepoctnov19, tstart_decjan1920]) - 2.458e6
    ends = np.array([tend_sep18, tend_nov18, tend_janfeb1819, tend_apr19, tend_may19_01, tend_may19_02, tend_jun19_01, tend_jun19_02, \
                     tend_jul19, tend_aug19, tend_sepoctnov19, tend_decjan1920]) - 2.458e6

    ##### make a nice stacked plot #####

    run = 'sepoctnov19'

    # runs = np.array(['all', 'sep', 'nov', 'jan', 'apr', 'may', 'jun', 'jul'])
    # runix = np.argwhere(runs == run)[0]
    # tlow, thigh = (starts[runix] - 3, ends[runix] + 3)

    if run == 'all':
        tlow, thigh = (np.min(starts) - 3, np.max(ends) + 3 )
    elif run == 'sep18':
        tlow, thigh = (tstart_sep18 - 2.458e6 - 3, tend_sep18 - 2.458e6 + 3)
    elif run == 'nov18':
        tlow, thigh = (tstart_nov18 - 2.458e6 - 3, tend_nov18 - 2.458e6 + 3)
    elif run == 'jan1819':
        tlow, thigh = (tstart_janfeb1819 - 2.458e6 - 3, tend_janfeb1819 - 2.458e6 + 3)
    elif run == 'apr19':
        tlow, thigh = (tstart_apr19 - 2.458e6 - 3, tend_apr19 - 2.458e6 + 3)
    elif run == 'may19':
        tlow, thigh = (tstart_may19_01 - 2.458e6 - 3, tend_may19_01 - 2.458e6 + 3)
    elif run == 'jun19':
        tlow, thigh = (tstart_may19_02 - 2.458e6 - 3, tend_jun19_02 - 2.458e6 + 3)
    elif run == 'jul19':
        tlow, thigh = (tstart_jul19 - 2.458e6 - 3, tend_jul19 - 2.458e6 + 3)
    elif run == 'aug19':
        tlow, thigh = (tstart_aug19 - 2.458e6 - 3, tend_aug19 - 2.458e6 + 3)
    elif run == 'sep19':
        tlow, thigh = (tstart_sep19 - 2.458e6 - 3, tend_sep19 - 2.458e6 + 3)
    elif run == 'sepoctnov19':
        tlow, thigh = (tstart_sepoctnov19 - 2.458e6 - 3, tend_sepoctnov19 - 2.458e6 + 3)
    elif run == 'decjan1920':
        tlow, thigh = (tstart_decjan1920 - 2.458e6 - 3, tend_decjan1920 - 2.458e6 + 3)



#     t_plot = tobj.datetime   # for plotting nice calendar dates
    t_plot = t.copy()   #for plotting JD
    fig, axarr = plt.subplots(3, sharex=True, figsize=(12,6.75))
    # plot 3 subplots
    axarr[0].plot(t_plot, dtemp, 'b')
    axarr[1].plot(t_plot, cryo, 'b')
    axarr[2].plot(t_plot, heater_load, 'b')
    # set (global) x-range
#     axarr[0].set_xlim(370,455)
#     axarr[0].set_xlim(500,692)  # ~17 Jan - 28 July 2019
#     axarr[0].set_xlim(375, 692)  # ~14 Sep 2018 - 28 July 2019
    axarr[0].set_xlim(tlow, thigh)
    # set y-ranges
    axarr[0].set_ylim(138,150)
    axarr[1].set_ylim(82,88)
    axarr[2].set_ylim(-5,30)
    # set titles
    axarr[0].set_title('detector temp')
    axarr[1].set_title('cryohead temp')
    axarr[2].set_title('heater load')
    # set x-axis label
    axarr[2].set_xlabel('JD - 2458000.0')
    # set y-axis labels
    axarr[0].set_ylabel('T [K]')
    axarr[1].set_ylabel('T [K]')
    axarr[2].set_ylabel('[%]')
    # indicate when Veloce was actually observing
    for x1,x2 in zip(starts,ends):
        axarr[0].axvspan(x1, x2, alpha=0.3, color='green')
        axarr[1].axvspan(x1, x2, alpha=0.3, color='green')
        axarr[2].axvspan(x1, x2, alpha=0.3, color='green')
    # # indicate tau Ceti obstimes with dashed vertical lines
    # for tobs in jd:
    #     axarr[0].axvline(tobs-2.458e6, color='gray', linestyle='--')
    #     axarr[1].axvline(tobs-2.458e6, color='gray', linestyle='--')
    #     axarr[2].axvline(tobs-2.458e6, color='gray', linestyle='--')
    # save to file
    # plt.savefig(...)
    
    
    
    
    timestamp_mech, temp_internal, temp_external, temp_room, temp_elec_cabinet, p_internal, p_room, p_regulator, p_set_point, p_rosso_cryo, h_internal, h_external, h_room, focus = readcol(path + 'velocemech.txt', skipline=1, twod=False)
    timestamp_mech_0, temp_internal_0, temp_external_0, temp_room_0, temp_elec_cabinet_0, p_internal_0, p_room_0, p_regulator_0, p_set_point_0, p_rosso_cryo_0, h_internal_0, h_external_0, h_room_0, focus_0 = readcol(path + 'velocemech.txt.0', skipline=1, twod=False)
    timestamp_mech_1, temp_internal_1, temp_external_1, temp_room_1, temp_elec_cabinet_1, p_internal_1, p_room_1, p_regulator_1, p_set_point_1, p_rosso_cryo_1, h_internal_1, h_external_1, h_room_1, focus_1 = readcol(path + 'velocemech.txt.1', skipline=1, twod=False)
    mech_time = Time(timestamp_mech, format='unix')       # from 26/10/2018 - 27/11/2018
    mech_time_0 = Time(timestamp_mech_0, format='unix')   # from 23/09/2018 - 26/10/2018
    mech_time_1 = Time(timestamp_mech_1, format='unix')   # from 13/08/2018 - 23/09/2018  
    
    timestamp_therm, temp_mc_setpoint, temp_mc_int, temp_extencl, temp_extencl_setpoint, state_mc = readcol(path + 'velocetherm.txt', skipline=1, twod=False)
    therm_time = Time(timestamp_therm, format='unix')
    
    timestamp_int_therm, temp_enc_setpoint, temp_enc_target, temp_cryo_setpoint, temp_sensor_1, temp_sensor_2, temp_sensor_3, temp_sensor_4, temp_sensor_5, temp_sensor_6, temp_sensor_7,\
        pwm_1, pwm_2, pwm_3, pwm_4, pwm_5 = readcol(path + 'veloceinttherm.txt', skipline=1, twod=False)
    timestamp_int_therm_0, temp_enc_setpoint_0, temp_enc_target_0, temp_cryo_setpoint_0, temp_sensor_1_0, temp_sensor_2_0, temp_sensor_3_0, temp_sensor_4_0, temp_sensor_5_0, temp_sensor_6_0, temp_sensor_7_0,\
        pwm_1_0, pwm_2_0, pwm_3_0, pwm_4_0, pwm_5_0 = readcol(path + 'veloceinttherm.txt.0', skipline=1, twod=False)
    timestamp_int_therm_1, temp_enc_setpoint_1, temp_enc_target_1, temp_cryo_setpoint_1, temp_sensor_1_1, temp_sensor_2_1, temp_sensor_3_1, temp_sensor_4_1, temp_sensor_5_1, temp_sensor_6_1, temp_sensor_7_1,\
        pwm_1_1, pwm_2_1, pwm_3_1, pwm_4_1, pwm_5_0 = readcol(path + 'veloceinttherm.txt.1', skipline=1, twod=False)
    int_therm_time = Time(timestamp_int_therm, format='unix')      # from 26/11/2018 - 27/11/2018
    int_therm_time_0 = Time(timestamp_int_therm_0, format='unix')  # from 25/11/2018 - 26/11/2018
    int_therm_time_1 = Time(timestamp_int_therm_1, format='unix')  # from 24/11/2018 - 25/11/2018
    
    return




def check_tauceti_shifts_with_telemetry():
    
    peak_path = '/Users/christoph/OneDrive - UNSW/lfc_peaks/tauceti/'
    red_path = '/Volumes/BERGRAID/data/veloce/reduced/tauceti/tauceti_with_LFC/'
    
    # load peak shifts (relto 21sep30019) from file
    peak_shifts = np.load(peak_path + 'LFC_peak_shifts.npy')
    x_shift = [shift[0] for shift in peak_shifts]
    y_shift = [shift[1] for shift in peak_shifts]
    
    # get sorted list of files
    files_sep = glob.glob(red_path + 'sep2018/' + '*10700*')
    files_nov = glob.glob(red_path + 'nov2018/' + '*10700*')
    allfiles = []
    for files in list([files_sep, files_nov]):
        all_shortnames = []
        for i,filename in enumerate(files):
            dum = filename.split('/')
            dum2 = dum[-1].split('.')
            dum3 = dum2[0]
            dum4 = dum3.split('_')
            shortname = dum4[1]
            all_shortnames.append(shortname)
        sortix = np.argsort(all_shortnames)
        files = np.array(files)
        files = files[sortix]
        allfiles.append(files)
    files = list(allfiles[0]) + list(allfiles[1])
    del all_shortnames
    del dum,dum2,dum3,dum4
    
    # get telemetry data from FITS headers
    temp_int = []
    temp_ext = []
    temp_rm = []
    temp_cab = []
    p_int = []
    p_rm = []
    p_rg = []
    p_sp = []
    hmd_ext = []
    hmd_int = []
    hmd_rm = []
    utmjd = []
    for i,file in enumerate(files[5:]):   #exclude the first 5 obs from 19 Sep 2018, as they currently do not have peak positions measured
        print('Reading file ' + str(i+1) + '/133')
        h = pyfits.getheader(file)
        utmjd.append(h['UTMJD'])
        temp_int.append(h['TEMPINT'])
        temp_ext.append(h['TEMPEXT'])
        temp_rm.append(h['TEMPRM'])
        temp_cab.append(h['TEMPCAB'])
        p_int.append(h['PRESSINT'])
        p_rm.append(h['PRESSRM'])
        p_rg.append(h['PRESSRG'])
        p_sp.append(h['PRESSSP'])
        hmd_int.append(h['HMDINT'])
        hmd_ext.append(h['HMDEXT'])
        hmd_rm.append(h['HMDRM'])
    
    return




def check_all_shifts_with_telemetry(nx=4112, save_M=False, save_shifts=False, verbose=False):
    
    peak_path = '/Volumes/BERGRAID/data/veloce/lfc_peaks/all/'
#     peak_path = '/Volumes/BERGRAID/data/veloce/lfc_peaks/tauceti/'
#     data_path = '/Volumes/BERGRAID/data/veloce/raw_goodonly/'
    data_path = '/Volumes/BERGRAID/data/veloce/raw/'
    
    peak_files_sep = glob.glob(peak_path + '*sep*olc.nst')
    peak_files_nov = glob.glob(peak_path + '*nov*olc.nst')
    
    # create sorted list of all LFC peakpos files
    allfiles = []
    for files in list([peak_files_sep, peak_files_nov]):
        all_shortnames = []
        for i,filename in enumerate(files):
            dum = filename.split('/')
            dum2 = dum[-1].split('.')
            dum3 = dum2[0][:10]
            shortname = dum3
            all_shortnames.append(shortname)
        sortix = np.argsort(all_shortnames)
        files = np.array(files)
        files = files[sortix]
        allfiles.append(files)
    files = list(allfiles[0]) + list(allfiles[1])
        
    # make list of the corresponding unique obsnum identifiers and dates
    obsnums = [file[-17:-7] for file in files] 
    day = [file[-17:-15] for file in files]
    num_month = list(np.repeat('09',len(peak_files_sep))) + list(np.repeat('11',len(peak_files_nov)))
#     nicedate = ['2018'+x+y for x,y in zip(num_month, day)]
    nicedate = ['18'+x+y for x,y in zip(num_month, day)]
    
    # prepare lists for telemetry data from FITS headers
    temp_int = []
    temp_ext = []
    temp_rm = []
    temp_cab = []
    p_int = []
    p_rm = []
    p_rg = []
    p_sp = []
    hmd_ext = []
    hmd_int = []
    hmd_rm = []
    utmjd = []
    heater_load = []
    cryotemp = []
    det_temp = []
    # prepare lists for peak shifts
    good_peakfiles = []
    M_list = []
    peak_shifts = []
    
    # loop over all peakpos files
    for i,file in enumerate(files):
        if verbose:
            print('Searching for file ' + str(i+1) + '/' + str(len(files)))
        # check if corresponding raw FITS file exists
        # fn = data_path + nicedate[i] + '/' + obsnums[i] + '.fits'
        fn = data_path + nicedate[i] + '/ccd_3/' + obsnums[i] + '.fits'
        if os.path.isfile(fn):
#         files_found = glob.glob(data_path + nicedate[i] + '/' + '*' + obsnums[i] + '*optimal*')
#             if len(files_found) == 1:
            # NORMAL CASE, ie FILE FOUND
            h = pyfits.getheader(fn)
            utmjd.append(h['UTMJD'])
            temp_int.append(h['TEMPINT'])
            temp_ext.append(h['TEMPEXT'])
            temp_rm.append(h['TEMPRM'])
            temp_cab.append(h['TEMPCAB'])
            p_int.append(h['PRESSINT'])
            p_rm.append(h['PRESSRM'])
            p_rg.append(h['PRESSRG'])
            p_sp.append(h['PRESSSP'])
            hmd_int.append(h['HMDINT'])
            hmd_ext.append(h['HMDEXT'])
            hmd_rm.append(h['HMDRM'])
            det_temp.append(h['DETTEMP'])
            cryotemp.append(h['CRYOTEMP'])
            heater_load.append(h['DETHTLD'])
            
            # keep track of which files could be cross-matched
            good_peakfiles.append(file)
            
        else:
            print('No raw spectrum found for ' + obsnums[i])
            
        # read in reference LFC peaks
        _, yref, xref, _, _, _, _, _, _, _, _ = readcol(peak_path + '21sep30019olc.nst', twod=False, skipline=2)    
        xref = nx - xref
        yref = yref - 54.    # or 53??? but does not matter for getting the transformation matrix
        # now loop over all the successfully cross-matched peakpos files and calculate the affine transformation matrix
        for file in good_peakfiles:
            try:
                _, y, x, _, _, _, _, _, _, _, _ = readcol(file, twod=False, skipline=2)
            except:
                _, y, x, _, _, _, _, _, _ = readcol(file, twod=False, skipline=2)
            del _
            x = nx - x
            y = y - 54.         # or 53??? but does not matter for getting the transformation matrix
        
        # compute affine transformation matrix
        if verbose:
            print('Calculating affine transformation matrix...')
        Minv = find_affine_transformation_matrix(xref, yref, x, y, eps=2.)
            
        M_list.append(Minv)
        peak_shifts.append((Minv[2,0], Minv[2,1]))   # append as tuple (x_shift,y_shift)
        
        if save_M:
            np.save('/Users/christoph/OneDrive - UNSW/lfc_peaks/M_list.npy', M_list)
        if save_shifts:
            np.save('/Users/christoph/OneDrive - UNSW/lfc_peaks/LFC_peak_shifts.npy', peak_shifts)    
    
    return





