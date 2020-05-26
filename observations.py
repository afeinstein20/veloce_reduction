import numpy as np
import time
import os
import glob
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

from veloce_reduction.readcol import readcol
from veloce_reduction.veloce_reduction.calibration import correct_orientation, crop_overscan_region
from veloce_reduction.veloce_reduction.helper_functions import laser_on, thxe_on, get_datestring






def create_PT0_dict(path='/Users/christoph/OneDrive - UNSW/observations/', savefile=True):

    # prepare dictionary
    PT0_dict = {}

#     # read TOI input file
#     targets, T0, P = readcol(path + 'toi_PT0_list.txt', twod=False, verbose=False)
#     # read input file for other targets (excluding B-stars)
#     targets, T0, P = readcol(path + 'other_PT0_list.txt', twod=False, verbose=False)

    # read input file for ALL targets (excluding B-stars)
    targets, T0, P = readcol(path + 'PT0_list.txt', twod=False, verbose=False)
    
    # fill dictionary with other targets
    for i in range(len(targets)):
        PT0_dict[targets[i]] = {'P':P[i], 'T0':T0[i]}

    if savefile:
        np.save(path + 'PT0_dict.npy', PT0_dict)

    return PT0_dict





def calculate_orbital_phase(name, jd=None, PT0_dict=None, t_ref = 2457000.):

    """
    Calculates the orbital phase of a given TOI planet. 
    Phase of 0 corresponds to time of transit, ie 0.25 and 0.75 correspond to quadratures.
    
    Input:
    'name'     : string containing TOI number (e.g. 'TOI136' or 'toi617' or 'HD212301')
    'jd'       : JD at which the phase is to evaluated (can be an array) - defaults to current JD if not provided
    'PT0_dict' : the dictionary containing information on each TOI's Period and T0
    't_ref'    : reference time used in TESS spreadsheet
    
    Output:
    'phase'  : orbital phase
    """

    if PT0_dict is None:
        PT0_dict = np.load('/Users/christoph/OneDrive - UNSW/observations/PT0_dict.npy').item()

    P = PT0_dict[name]['P']
    T0 = PT0_dict[name]['T0']

    if jd is None:
        # jd = jdnow()
        jd = time.time() / 86400. + 2440587.5

    phase = np.mod(jd - (t_ref + T0), P) / P

    return phase





def create_toi_velobs_dict(obspath='/Users/christoph/OneDrive - UNSW/observations/', savefile=True, src='raw', save_lists=True, laptop=False):

    """
    TODO:
    expand to other targets as well!
    add seeing, cal source, RV, etc.
    """

    # code defensively...
    if laptop:
        redpath = '/Users/christoph/data/reduced/'
        rawpath = '/Users/christoph/data/raw_godoonly/'
        logpath = '/Users/christoph/data/veloce_logs/'
    else:
        redpath = '/Volumes/BERGRAID/data/veloce/reduced/'
        rawpath = '/Volumes/BERGRAID/data/veloce/raw_goodonly/'
        logpath = '/Volumes/BERGRAID/data/veloce/veloce_logs/'

    while src.lower() not in ["red", "reduced", "raw"]:
        print("ERROR: invalid source input !!!")
        src = raw_input("Do you want to create the observation dictionary from raw or reduced files? (valid options are ['raw' / 'red(uced)'] )?") 
    if src.lower() in ['red', 'reduced']:
        src_path = redpath
        rawred = 'reduced'
    elif src.lower() == 'raw':
        src_path = rawpath
        rawred = 'raw'
    else:
        # that should never happen!
        print('ERROR: you broke the world!')
        return -1
        
    assert os.path.isdir(src_path), "ERROR: directory containing the " + rawred + " data does not exist!!!"


    # read input file
    targets, T0, P = readcol(obspath + 'PT0_list.txt', twod=False, verbose=False)

    # initialise dictionary
    vo = {}
    
    if src.lower() in ['red', 'reduced']:
        all_obs_list = get_reduced_obslist(laptop=laptop)
        if save_lists:
            datestring = get_datestring()
            np.save(obspath + 'all_' + rawred + '_obs_list_' + datestring + '.npy', all_obs_list)
    elif src.lower() == 'raw':
        all_obs_list, all_target_list = get_raw_obslist(return_targets=True, rawpath=rawpath)
        if save_lists:
            datestring = get_datestring()
            np.save(obspath + 'all_' + rawred + '_obs_list_' + datestring + '.npy', all_obs_list)
            np.save(obspath + 'all_' + rawred + '_target_list_' + datestring + '.npy', all_target_list)

    # loop over all targets
    for targ, t0, per in zip(targets, T0, P):
        
        print(targ)

        # loop over all "names"
        typ = targ[:3]
        
        # for TOIs
        if typ == 'TOI':
#             name = targ[-3:]
            name = targ.split('I')[-1]
            synonyms = ['TOI'+name, 'TOI'+name+'.01', 'TIC'+name+'.01', name+'.01', name+'.01A', name+'.01BC', name+'.01_Bouma', 'BKT'+name+'.01', 'BKTRM'+name+'.01']
            if src.lower() in ['red', 'reduced']:
#                 fn_list = [fn for fn in all_obs_list if ((fn.split('/')[-1]).split('_')[0]).split('+')[0] in synonyms]     # before I added the date to the reduced-spectrum filenames
                fn_list = [fn for fn in all_obs_list if ((fn.split('/')[-1]).split('_')[1]).split('+')[0] in synonyms]
            elif src.lower() == 'raw':
                fn_list = [fn for fn,target in zip(all_obs_list, all_target_list) if target.split('+')[0] in synonyms]
        # for other targets
        else:
            targ2 = targ[:]
            if targ[:2] == 'HD':
                targ2 = targ[2:]
            elif targ == 'ksiHya_new':
                targ2 = 'ksiHya'
            elif targ == 'KELT-15b':
                targ2 = 'KELT 15b'
            elif targ == 'AST301':
                targ2 = 'TIC141808675'
            elif targ == 'AST303':
                targ2 = 'TIC142142718'
            if src.lower() in ['red', 'reduced']:
#                 fn_list = [fn for fn in all_obs_list if ((fn.split('/')[-1]).split('_')[0]).split('+')[0] in [targ, targ2]]     # before I added the date to the reduced-spectrum filenames
                fn_list = [fn for fn in all_obs_list if ((fn.split('/')[-1]).split('_')[1]).split('+')[0] in [targ, targ2]]
            elif src.lower() == 'raw':
                fn_list = [fn for fn,target in zip(all_obs_list, all_target_list) if target.split('+')[0] in [targ, targ2]]
        
        # sort the entire list            
        fn_list.sort()

        # only create an entry if we have observations for this target
        if len(fn_list) > 0:
            # initialise sub-dictionary for this object
            vo[targ] = {}
    
            # prepare dictionary entry for this target
            vo[targ]['filenames'] = fn_list
            vo[targ]['dates'] = []
            vo[targ]['nobs'] = len(fn_list)
            vo[targ]['P'] = per
            vo[targ]['T0'] = t0
            vo[targ]['JD'] = []
            vo[targ]['texp'] = []
            vo[targ]['obsnames'] = []
            vo[targ]['phase'] = []
            vo[targ]['epoch_phase'] = []
            vo[targ]['epoch_snr'] = []
            vo[targ]['epoch_seeing'] = []
            vo[targ]['epoch_dates'] = []
            vo[targ]['epoch_JD'] = []
            vo[targ]['snr'] = []     # from logsheet
            vo[targ]['sqrtnphot'] = []     # from logsheet
            vo[targ]['seeing'] = []  # from logsheet
            # vo[targ]['cal'] = []     # would be nice as an upgrade in the future (calibration source: LFC, ThXe, interp?)
            # vo[targ]['rv'] = []     # would be nice as an upgrade in the future
            days = []
            seqs = []
            
            # fill dictionary
            # loop over all observations for this target
            for n,file in enumerate(fn_list):
#                 print(n, file)
                h = pyfits.getheader(file)
                vo[targ]['JD'].append(h['UTMJD'] + 2.4e6 + 0.5 + (0.5 * h['ELAPSED'] / 86400.))  # use plain JD here, in order to avoid confusion
                vo[targ]['texp'].append(h['ELAPSED'])
                if src.lower() in ['red', 'reduced']:
#                     obsname = (file.split('/')[-1]).split('_')[1]   # before I added the date to the reduced-spectrum filenames
#                     obsname = (file.split('/')[-1]).split('_')[2]   # this does not work if there is a PI name appended (eg "20191012_429.01_Dragomir_12oct30116_optimal3a_extracted.fits")
                    obsname = (file.split('/')[-1]).split('_')[-3]
                elif src.lower() == 'raw':
                    obsname = ((file.split('/')[-1]).split('_')[0]).split('.')[0]
                vo[targ]['obsnames'].append(obsname)
                date = file.split('/')[-2] 
                vo[targ]['dates'].append(date)
                day = obsname[:5]
                days.append(day)
                seq = obsname[5:]
                seqs.append(seq)
                logfilename = glob.glob(logpath + '*' + date[2:] + '*.log')[0]
#                 print(date)
                snr = np.nan   # fall-back option if snr not found in logsheets
                sqrtnphot = np.nan   # fall-back option if snr not found in logsheets
                seeing = np.nan   # fall-back option if seeing not found in logsheets
                with open(logfilename) as logfile:
                    for line in logfile:
                        if line[:4] == seq[1:]: 
#                             print(line)
                            # SNR
                            snr = line.split()[5]
                            try:
                                snr = int(snr)
                            except:
                                snr = np.nan
                            # SNR
                            sqrtnphot = line.split()[6]
                            try:
                                sqrtnphot = int(sqrtnphot)
                            except:
                                sqrtnphot = np.nan
                            # seeing
                            seeing = line.split()[7]
                            try:
                                if date <= 20180926:
                                    seeing = np.round(float(seeing), 1)
                                else:
                                    seeing = np.round(float(seeing[:-1]),1)
                            except:
                                seeing = np.nan
                if int(date) >= 20200129:
                    seeing_fac = 1.
                else:
                    seeing_fac = np.sqrt(2)
                vo[targ]['snr'].append(snr)
                vo[targ]['sqrtnphot'].append(sqrtnphot)
                vo[targ]['seeing'].append(np.round(seeing * seeing_fac,1))
                
            vo[targ]['phase'].append(calculate_orbital_phase(targ, vo[targ]['JD']))
            vo[targ]['phase'] = vo[targ]['phase'][0] 
            # check which exposures are adjacent to determine the number of epochs
#             vo[targ]['nepochs'] = vo[targ]['nobs']
            vo[targ]['nepochs'] = len(set(vo[targ]['dates']))
            
            if vo[targ]['nobs'] >= 2:
#                 for d in set(days):
                for d in sorted(set(vo[targ]['dates'])):
#                     print(d)
                    n_epoch_per_night = 1
#                     ix = [i for i, day in enumerate(days) if day == d]
                    ix = [i for i, date in enumerate(vo[targ]['dates']) if date == d]
                    rundiff = np.diff(np.array(seqs)[ix].astype(int))
                    # if the obsnums are not consecutive, check how long in between - count it as a new epoch if more than an hour in between exposures
                    if (rundiff > 1).any():
                        gaps = np.where(rundiff > 1)[0]
#                         obsnums = np.array(seqs)[ix].astype(int)
                        jds = np.array(vo[targ]['JD'])[ix]
                        exptimes = np.array(vo[targ]['texp'])[ix]
                        for gap in sorted(gaps):
                            delta_t = 86400. * (jds[gap+1] - jds[gap]) - exptimes[gap+1]/2. - exptimes[gap]/2. 
                            if delta_t > 3600. :
                                vo[targ]['nepochs'] += 1
                                n_epoch_per_night += 1
                    
                    if n_epoch_per_night == 1:
                        # for the individual single-shot exposures within this epoch get the phases, SNRs, and JDs
                        ss_phases = vo[targ]['phase'][ix]
                        snrs = np.array(vo[targ]['snr'])[ix]
                        seeings = np.array(vo[targ]['seeing'])[ix]
                        jds = np.array(vo[targ]['JD'])[ix]
                        # get the combined SNR for the epoch
                        vo[targ]['epoch_snr'].append(np.sqrt(np.sum(snrs**2)))
                        # now get the weighted mean phase and mean JD for the epoch - weight by SNR if possible; if not, use exposure times
                        if ~np.isnan(snrs).any():
                            vo[targ]['epoch_phase'].append(np.average(ss_phases, weights=snrs))
                            vo[targ]['epoch_JD'].append(np.average(jds, weights=snrs))
                            vo[targ]['epoch_seeing'].append(np.average(seeings, weights=snrs))
                        else:
                            print('WARNING: logsheet incomplete for', d)
                            exptimes = np.array(vo[targ]['texp'])[ix]
                            vo[targ]['epoch_phase'].append(np.average(ss_phases, weights=exptimes))
                            vo[targ]['epoch_JD'].append(np.average(jds, weights=exptimes))
                            vo[targ]['epoch_seeing'].append(np.average(seeings, weights=exptimes))
                        # record the date for that epoch
                        vo[targ]['epoch_dates'].append(d)
                    else:
                        ix_sublist = []
                        substart_indices = np.r_[0, np.array(gaps)+1, len(ix)]
                        ix_sublist = [ix[substart_indices[j]:substart_indices[j+1]] for j in range(len(substart_indices)-1)]   # that hurts my brain, but it works ;)
                        for subix in ix_sublist:
                            # for the individual single-shot exposures within this epoch get the phases, SNRs, and JDs
                            ss_phases = vo[targ]['phase'][subix]
                            snrs = np.array(vo[targ]['snr'])[subix]
                            seeings = np.array(vo[targ]['seeing'])[subix]
                            jds = np.array(vo[targ]['JD'])[subix]
                            # get the combined SNR for the epoch
                            vo[targ]['epoch_snr'].append(np.sqrt(np.sum(snrs**2))) 
                            # now get the weighted mean phase and mean JD for the epoch - weight by SNR if possible; if not, use exposure time
                            if ~np.isnan(snrs).any():
                                vo[targ]['epoch_phase'].append(np.average(ss_phases, weights=snrs))
                                vo[targ]['epoch_JD'].append(np.average(jds, weights=snrs))
                                vo[targ]['epoch_seeing'].append(np.average(seeings, weights=snrs))
                            else:
                                print('WARNING: logsheet incomplete for', d)
                                exptimes = np.array(vo[targ]['texp'])[subix]
                                vo[targ]['epoch_phase'].append(np.average(ss_phases, weights=exptimes))
                                vo[targ]['epoch_JD'].append(np.average(jds, weights=exptimes))
                                vo[targ]['epoch_seeing'].append(np.average(seeings, weights=exptimes))
                            # record the date for that epoch
                            vo[targ]['epoch_dates'].append(d)

    if savefile:
        np.save(obspath + 'velobs_' + rawred + '.npy', vo)
        np.save(obspath + 'velobs_' + rawred + '_' + datestring + '.npy', vo)

    return vo





def create_bstar_velobs_dict(path='/Users/christoph/OneDrive - UNSW/observations/', savefile=True, src='raw', laptop=False):

    # code defensively...
    if laptop:
        redpath = '/Users/christoph/data/reduced/'
        rawpath = '/Users/christoph/data/raw_godoonly/'
    else:
        redpath = '/Volumes/BERGRAID/data/veloce/reduced/'
        rawpath = '/Volumes/BERGRAID/data/veloce/raw_goodonly/'
    
    while src.lower() not in ["red", "reduced", "raw"]:
        print("ERROR: invalid source input !!!")
        src = raw_input("Do you want to create the observation dictionary from raw or reduced files? (valid options are ['raw' / 'red(uced)'] )?") 
    if src.lower() in ['red', 'reduced']:
        src_path = redpath
        rawred = 'reduced'
    elif src.lower() == 'raw':
        src_path = rawpath
        rawred = 'raw'
    else:
        # that should never happen!
        print('ERROR: you broke the world!')
        return -1
        
    assert os.path.isdir(src_path), "ERROR: directory containing the " + rawred + " data does not exist!!!"


    # read input file
#     targets, T0, P = readcol(path + 'PT0_list.txt', twod=False, verbose=False)
    targets = readcol('/Users/christoph/OneDrive - UNSW/observing/AAT/bstars.txt', twod=False)
    targets = targets[0].astype(str)     # readcol peculiarity...

    # initialise dictionary
    vo = {}
    
    if src.lower() in ['red', 'reduced']:
        all_obs_list = get_reduced_obslist(laptop=laptop)
    elif src.lower() == 'raw':
        all_obs_list, all_target_list = get_raw_obslist(return_targets=True, laptop=laptop)

    # loop over all targets
    for targ in targets:
        
        print(targ)
        
        # initialise sub-dictionary for this object
        vo[targ] = {}

        # for B stars
        synonyms = [targ, 'HD'+targ]
        if targ == '10144':
            synonyms.append('Achernar')
            synonyms.append('Achenar')
        if targ == '209952':
            synonyms.append('alphaGru')
            synonyms.append('AlphaGru')
        
        if src.lower() in ['red', 'reduced']:
            fn_list = [fn for fn in all_obs_list if ((fn.split('/')[-1]).split('_')[0]).split('+')[0] in synonyms]
        elif src.lower() == 'raw':
            fn_list = [fn for fn,target in zip(all_obs_list, all_target_list) if target.split('+')[0] in synonyms]      
    
        fn_list.sort()

        # prepare dictionary entry for this target
        vo[targ]['filenames'] = fn_list
        vo[targ]['nobs'] = len(fn_list)
#         vo[targ]['P'] = per
#         vo[targ]['T0'] = t0
        vo[targ]['JD'] = []
        vo[targ]['texp'] = []
        vo[targ]['obsnames'] = []
#         vo[targ]['phase'] = []
        # vo[targ]['snr'] = []     # would be nice as an upgrade in the future
        # vo[targ]['cal'] = []     # would be nice as an upgrade in the future (calibration source: LFC, ThXe, interp?)
        # vo[targ]['rv'] = []     # would be nice as an upgrade in the future
        days = []
        seq = []
        # fill dictionary
        # loop over all observations for this target
        for file in fn_list:
            h = pyfits.getheader(file)
            vo[targ]['JD'].append(h['UTMJD'] + 2.4e6 + 0.5 + (0.5 * h['ELAPSED'] / 86400.))  # use plain JD here, in order to avoid confusion
            vo[targ]['texp'].append(h['ELAPSED'])
            if src.lower() in ['red', 'reduced']:
                obsname = (file.split('/')[-1]).split('_')[1]
            elif src.lower() == 'raw':
                obsname = ((file.split('/')[-1]).split('_')[0]).split('.')[0]
            vo[targ]['obsnames'].append(obsname)
            days.append(obsname[:5])
            seq.append(obsname[5:])
#         vo[targ]['phase'].append(calculate_orbital_phase(targ, vo[targ]['JD']))
        # check which exposures are adjacent to determine the number of epochs
        vo[targ]['nepochs'] = vo[targ]['nobs']
        if vo[targ]['nobs'] >= 2:
            for d in set(days):
                ix = [i for i, day in enumerate(days) if day == d]
                rundiff = np.diff(np.array(seq)[ix].astype(int))
                vo[targ]['nepochs'] -= np.sum(rundiff == 1)

    if savefile:
        np.save(path + 'velobs_bstars_' + rawred + '.npy', vo)

    return vo





def plot_toi_phase(toi, vo=None, saveplot=False, outpath=None):
    if vo is None:
        vo = np.load('/Users/christoph/OneDrive - UNSW/observations/velobs_raw.npy').item()
    # representative plot of a normalized circular orbit with the orbital phases of the obstimes indicated
    plt.figure()
    x = np.linspace(0, 1, 1000)
    plt.plot(x, np.sin(2. * np.pi * x), 'k')
    phi = np.squeeze(vo[toi]['phase'])
    plt.plot(phi, np.sin(2. * np.pi * phi), 'ro')
    plt.xlabel('orbital phase')
    plt.ylabel('dRV / K')
    plt.title(toi + '  -  orbital phase coverage')
    plt.text(0.95, 0.85, '   #obs: ' + str(vo[toi]['nobs']), size='x-large', horizontalalignment='right')
    plt.text(0.95, 0.70, '#epochs: ' + str(vo[toi]['nepochs']), size='x-large', horizontalalignment='right')
    if saveplot:
        try:
            plt.savefig(outpath + toi + '_orbital_phase_coverage.eps')
        except:
            print('ERROR: output directory not provided...')
    plt.close()
    return





def plot_toi_phase_lin(toi, vo=None, saveplot=False, outpath=None, no_xlabel=False, no_title=False):
    if vo is None:
        vo = np.load('/Users/christoph/OneDrive - UNSW/observations/velobs_raw.npy').item()
    # representative plot of a normalized circular orbit with the orbital phases of the obstimes indicated
#     plt.figure()
    if not no_xlabel:
        plt.figure(figsize=(10,1.2))
    else:
        plt.figure(figsize=(10,0.9))
    x = np.linspace(0, 1, 1000)
    plt.plot(x, np.zeros(len(x)), 'k')
    phi = np.squeeze(vo[toi]['phase'])
    ep_phi = vo[toi]['epoch_phase']
    plt.plot(phi, np.zeros(len(phi)), 'ro', markersize=3)
    plt.plot(ep_phi, np.zeros(len(ep_phi)), 'bx', markersize=7)
    plt.ylabel(toi, rotation='horizontal', ha='right', va='center', fontsize='large', fontweight='bold')
    plt.yticks([])
    plt.xlim(-0.05,1.25)
    plt.text(1.225, 0.025, '   #obs: ' + str(vo[toi]['nobs']), size='large', horizontalalignment='right', va='center')
    plt.text(1.225, -0.025, '#epochs: ' + str(vo[toi]['nepochs']), size='large', horizontalalignment='right', va='center')
    if not no_xlabel:
        plt.xticks([0,0.25,0.5,0.75,1])
        plt.xlabel('orbital phase')
    else:
        plt.xticks([0,0.25,0.5,0.75,1], [])
    if not no_title:
        plt.title(toi + '  -  orbital phase coverage')
    plt.tight_layout()
    
    if saveplot:
        try:
            plt.savefig(outpath + toi + '_orbital_phase_coverage_lin.eps')
        except:
            print('ERROR: output directory not provided...')
    plt.close()
    return





def plot_all_toi_phases(src='raw', path='/Users/christoph/OneDrive - UNSW/observations/', saveplots=True, lin=False, per_page=16):
    if src.lower() == 'raw':
        vo = np.load(path + 'velobs_raw.npy').item()
    elif src.lower() in ['red', 'reduced']:
        vo = np.load(path + 'velobs_reduced.npy').item()
    else:
        return -1
    
    toi_list = np.array([targ for targ in vo.keys() if targ[:3] == 'TOI'])
    toi_nums = np.array([int(targ[3:]) for targ in vo.keys() if targ[:3] == 'TOI'])
    sortix = np.argsort(toi_nums)
    
    for n,targ in enumerate(toi_list[sortix]):
        if lin:
            nox = True
            if ((n+1) % per_page == 0) or (n == len(toi_list)-1): 
                nox = False
            plot_toi_phase_lin(targ, vo=vo, saveplot=saveplots, outpath = path + 'plots/', no_xlabel=nox, no_title=True)
        else:
            plot_toi_phase(targ, vo=vo, saveplot=saveplots, outpath = path + 'plots/')
    return





def get_reduced_obslist(laptop=False):

    if laptop:
        redpath = '/Users/christoph/data/reduced/'
    else:
        redpath = '/Volumes/BERGRAID/data/veloce/reduced/'

    assert os.path.isdir(redpath), "ERROR: directory containing the reduced data does not exist!!!"

    # count all the nightly directories only (ie not the other ones like "tauceti")
    datedir_list = glob.glob(redpath + '20*')
    datedir_list.sort()
    print('Searching for reduced spectra in ' + str(len(datedir_list)) + ' nights of observations...')

    # all_target_list = []
    all_obs_list = []

    for datedir in datedir_list:
        datedir += '/'
        obs_list = glob.glob(datedir + '*optimal*')
        all_obs_list.append(obs_list)
        # target_set = set([(fn.split('/')[-1]).split('_')[0] for fn in obs_list])
        # all_target_list.append(list(target_set))

    all_obs_list = [item for sublist in all_obs_list for item in sublist]
#     all_target_list = [((fn.split('/')[-1]).split('_')[0]).split('+')[0] for fn in all_obs_list]
#     all_targets = set([item for sublist in all_target_list for item in sublist])
#     unique_targets = set(all_target_list)

    return all_obs_list





def get_raw_obslist(return_targets=False, rawpath='/Volumes/BERGRAID/data/veloce/raw_goodonly/', verbose=True):

    assert os.path.isdir(rawpath), "ERROR: directory containing the RAW data does not exist!!!"

    # count all the nightly directories only (ie not the other ones like "tauceti")
    datedir_list = glob.glob(rawpath + '20*')
    datedir_list.sort()
    print('Searching for reduced spectra in ' + str(len(datedir_list)) + ' nights of observations...')

    dum_obs_list = []
    
    for datedir in datedir_list:
        datedir += '/'
        obs_list = glob.glob(datedir + '[0-3]*.fits')
        dum_obs_list.append(obs_list)
        # target_set = set([(fn.split('/')[-1]).split('_')[0] for fn in obs_list])
#         all_target_list.append(list(target_set))

    dum_obs_list = [item for sublist in dum_obs_list for item in sublist]
    shortfn_list = [(fn.split('/')[-1]).split('.')[0] for fn in dum_obs_list]
    all_obs_list = [obs for obs,shortfn in zip(dum_obs_list, shortfn_list) if len(shortfn)==10]
    
    if return_targets:
        all_target_list = []
        for i,file in enumerate(all_obs_list):
            if verbose:
                print('Reading FITS header ' + str(i+1) + '/' + str(len(all_obs_list)) + '...')
            object = pyfits.getval(file, 'OBJECT')
            all_target_list.append(object)
 
#     unique_targets = set(all_target_list)
    
    if return_targets:
        return all_obs_list, all_target_list
    else:
        return all_obs_list





def make_text_file_for_latex(vo):
    outfile = open('/Users/christoph/OneDrive - UNSW/observations/plots/dumdum.txt', 'w')
    n=1
    for i,toi in enumerate(sorted(vo.keys())):
        if toi[0:3].lower() == 'toi':
            n += 1
            if np.mod(n,2) == 0:
                outfile.write(r'\begin{figure}[H]' + '\n')
            outfile.write(r'\includegraphics[width=0.99\linewidth]{' + toi + '_orbital_phase_coverage.eps}' + '\n')
            if np.mod(n,2) != 0:
                outfile.write(r'\end{figure}' + '\n')
                outfile.write(r'\newpage' + '\n')
    outfile.close()
    return





def make_text_file_for_latex_lin(vo, per_page=16):
    outfile = open('/Users/christoph/OneDrive - UNSW/observations/plots/dumdum_lin.txt', 'w')
    
    toi_list = np.array([targ for targ in vo.keys() if targ[:3] == 'TOI'])
    toi_nums = np.array([int(targ[3:]) for targ in vo.keys() if targ[:3] == 'TOI'])
    sortix = np.argsort(toi_nums)
    
    for n,toi in enumerate(toi_list[sortix]):
        if n % per_page == 0:
            outfile.write(r'\begin{figure}[H]' + '\n')
        outfile.write(r'\includegraphics[width=0.99\linewidth]{' + toi + '_orbital_phase_coverage_lin.eps}' + '\n')
        if (n+1) % per_page == 0:
            outfile.write(r'\end{figure}' + '\n')
            outfile.write(r'\newpage' + '\n')
    outfile.write(r'\end{figure}' + '\n')
    outfile.close()
    return






class star(object):
    def __init__(self, name):
        self.name = name   # only name is required input
        self.path = '/Users/christoph/OneDrive - UNSW/observations/'
#         PT0_dict = np.load(self.path + 'toi_PT0_dict.npy').item()
        vo = np.load(self.path + 'velobs_raw.npy').item()
        self.P = vo[name]['P']
        self.T0 = vo[name]['T0']
        self.nobs = vo[name]['nobs']
        self.nepochs = vo[name]['nepochs']
        del vo
        
#         self.ra = vo[toi]['ra']
#         self.dec =vo[toi]['dec']

    def phase_plot(self, saveplot=False):
        """
        representative plot of a normalized circular orbit with the orbital phases of the obstimes indicated
        """
        plt.figure()
        x = np.linspace(0, 1, 1000)
        plt.plot(x, np.sin(2. * np.pi * x), 'k')
        phi = np.squeeze(vo[toi]['phase'])
        plt.plot(phi, np.sin(2. * np.pi * phi), 'ro')
        plt.xlabel('orbital phase')
        plt.ylabel('dRV / K')
        plt.title(toi + '  -  orbital phase coverage')
        plt.text(0.95, 0.85, '   #obs: ' + str(vo[toi]['nobs']), size='x-large', horizontalalignment='right')
        plt.text(0.95, 0.70, '#epochs: ' + str(vo[toi]['nepochs']), size='x-large', horizontalalignment='right')
        if saveplot:
            try:
                plt.savefig(self.path + 'plot/' + self.name + '_orbital_phase_coverage.eps')
            except:
                print('ERROR: output directory not provided...')
        plt.close()
        return
        
        


def update_redobs(starname, starpath=None, pattern=None, overwrite=False):
    """
    USAGE:
    # update_redobs('TOI375')
    # update_redobs('TOI274')
    # update_redobs('GJ674', pattern='674')
    # update_redobs('HD212301')
    """
    redpath = '/Volumes/BERGRAID/data/veloce/reduced/'
    working_redpath = '/Volumes/WORKING/data/veloce/reduced/'
    rawpath = '/Volumes/BERGRAID/data/veloce/raw_goodonly/'
    chipmask_path = '/Users/christoph/OneDrive - UNSW/chipmasks/archive/'
    
    # create target directory if not provided
    if starpath is None:
        starpath = working_redpath + starname + '/'
    lfc_path = starpath + 'with_lfc/'
    thxe_path = starpath + 'with_thxe/'
    neither_path = starpath + 'neither/'
    
    # define search pattern if not provided
    if pattern is None:
        if starname[:2] == 'HD':
            pattern = starname[2:]
        elif starname[:3] == 'TOI':
            pattern = starname[3:]
        else:
            print('WARNING: automated pattern creation failed - please provide pattern')
            return
    
    # check if path exists, if not then create it
    if not os.path.exists(starpath):
        os.system("mkdir " + starpath)
    if not os.path.exists(lfc_path):    
        os.system("mkdir " + lfc_path)
    if not os.path.exists(thxe_path):    
        os.system("mkdir " + thxe_path)
    if not os.path.exists(neither_path):    
        os.system("mkdir " + neither_path)
        
    # copy all reduced files of the target to the target folder
#     os.system("find " + redpath + "20* -name '" + "*" + pattern + "*optimal*' -exec cp '{}' " + starpath + " \;")
    if starname[:2] == 'HD':
        synonyms = ['HD'+pattern, pattern]
    elif starname[:3] == 'TOI':
        synonyms = ['TOI'+pattern, 'TOI'+pattern+'.01', 'TIC'+pattern+'.01', pattern+'.01']
    else:
        synonyms = [pattern]
    for syn in synonyms:
        os.system("find " + redpath + "20* -name '" + "*" + syn + "*optimal*' -exec cp '{}' " + starpath + " \;")
    
    # some more housekeeping...
    all_files = glob.glob(starpath + "*" + pattern + '*.fits')

    for i,file in enumerate(all_files):
        short_filename = file.split('/')[-1]
        obsname = file.split('_')[-3]
        utdate = pyfits.getval(file, 'UTDATE')
        date = utdate[:4] + utdate[5:7] + utdate[8:]
        print('Processing file ' + str(i+1) + '/' + str(len(all_files)) + '   (' + obsname + ')')
        chipmask = np.load(chipmask_path + 'chipmask_' + date + '.npy').item()
        img = crop_overscan_region(correct_orientation(pyfits.getdata(rawpath + date + '/' + obsname + '.fits')))
        lc = laser_on(img, chipmask)
        thxe = thxe_on(img, chipmask)
        if lc:
            if os.path.isfile(lfc_path + short_filename):
                if overwrite:
                    os.rename(file, lfc_path + short_filename)
                else:
                    os.remove(file)
            else:
                os.rename(file, lfc_path + short_filename)
        else:
            if thxe:
                if os.path.isfile(thxe_path + short_filename):
                    if overwrite:
                        os.rename(file, thxe_path + short_filename)
                    else:
                        os.remove(file)
                else:
                    os.rename(file, thxe_path + short_filename)
            else:
                if os.path.isfile(neither_path + short_filename):
                    if overwrite:
                        os.rename(file, neither_path + short_filename)
                    else:
                        os.remove(file)
                else:
                    os.rename(file, neither_path + short_filename)
    
    # make sure there are no remaining (uncategorised) files
    rem_files = glob.glob(starpath + pattern + '*.fits')
    if len(rem_files) > 0:
        print('WARNING: some files could not be taken care of!!!')
        print(rem_files)
        
    return













