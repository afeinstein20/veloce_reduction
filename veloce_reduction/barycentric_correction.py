import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import astropy.io.fits as pyfits
import barycorrpy
import numpy as np
from readcol import readcol
import glob

from veloce_reduction.veloce_reduction.get_info_from_headers import get_obstype_lists
from veloce_reduction.veloce_reduction.helper_functions import short_filenames

# starnames = ['HD10700', 'HD190248', 'Gl87', 'GJ1132', 'GJ674', 'HD194640', 'HD212301']



def create_gaiadr2_id_dict(path='/Users/christoph/OneDrive - UNSW/observations/', savefile=True):

    # prepare dictionary
    gaia_dict = {}

    # read input file for ALL targets (no including some B-stars)
    targets, T0, P, g2id = readcol(path + 'PT0_gaiadr2_list.txt', twod=False, verbose=False)
    
    # fill dictionary with target
    for i in range(len(targets)):
        gaia_dict[targets[i]] = {'gaia_dr2_id':g2id[i][1:]}

    if savefile:
        np.save(path + 'gaiadr2_id_dict.npy', gaia_dict)

    return gaia_dict



def get_barycentric_correction(fn, rvabs=None, obs_path='/Users/christoph/OneDrive - UNSW/observations/'):
    """
    wrapper routine for using barycorrpy with Gaia DR2 coordinates
    """
    
    # use 2015.5 as an epoch (Gaia DR2)
    epoch = 2457206.375
    
    # get UT obs start time
    utmjd = pyfits.getval(fn, 'UTMJD') + 2.4e6 + 0.5   # the fits header has 2,400,000.5 subtracted!!!!!
    # add half the exposure time in days
    texp = pyfits.getval(fn, 'ELAPSED')
    utmjd = utmjd + (texp/2.)/86400.
    
    # read in Gaia DR2 IDs
    gaia_dict = np.load(obs_path + 'gaiadr2_id_dict.npy').item()    

    # check what kind of target it is
    targ_raw = pyfits.getval(fn, 'OBJECT')
    targ = targ_raw.split('+')[0]
    typ = targ[-3:]   

    try:
        # for TOIs
        if (typ == '.01') or (targ[:3] in ['TOI', 'TIC']):
            if len(targ) <= 10:
                if targ[:3] in ['TOI', 'TIC']:
                    gaia_dr2_id = gaia_dict['TOI'+targ[3:6]]['gaia_dr2_id']
                else:
                    gaia_dr2_id = gaia_dict['TOI'+targ[:3]]['gaia_dr2_id']
            else:
                gaia_dr2_id = gaia_dict[targ]['gaia_dr2_id']
        # for other targets
        else:
            if targ.lower() in ['gj674', 'gl87', 'gl480.1', 'proxima', 'kelt-15b', 'wasp-54b', 'gj514', 'gj526', 'gj699', 'gj3192', 'gj3193']:
                gaia_dr2_id = gaia_dict[targ]['gaia_dr2_id']
            elif targ.lower() == 'gj87':
                gaia_dr2_id = gaia_dict['Gl87']['gaia_dr2_id']
            elif targ.lower() in ['zetapic', 'zeta pic']:
                gaia_dr2_id = gaia_dict['zetaPic']['gaia_dr2_id']
            elif targ.lower() in ['ekeri', 'ek eri']:
                gaia_dr2_id = gaia_dict['EKEri']['gaia_dr2_id']
            elif targ.lower() in ['ksihya', 'ksi hya', 'ksihya_new']:
                gaia_dr2_id = gaia_dict['ksiHya']['gaia_dr2_id']
            elif targ.lower()[:2] in ['hd', 'hr', 'as']:
                gaia_dr2_id = gaia_dict[targ]['gaia_dr2_id']
            else:
                gaia_dr2_id = gaia_dict['HD'+targ]['gaia_dr2_id']
    except:
        print('WARNING: could not find Gaia DR2 ID for target: ', targ)
        gaia_dr2_id = None
        return np.nan


#     coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
#     width = u.Quantity(w, u.deg)
#     height = u.Quantity(h, u.deg)

#     gaia_data = Gaia.query_object_async(coordinate=coord, width=width, height=height)
#     q = Gaia.launch_job_async('SELECT * FROM gaiadr2.gaia_source WHERE source_id = ' + str(gaia_dr2_id))
    q = Gaia.launch_job('SELECT * FROM gaiadr2.gaia_source WHERE source_id = ' + str(gaia_dr2_id))
    gaia_data = q.results

    # some targets don't have a RV from Gaia
    if rvabs is None:
        rvabs = gaia_data['radial_velocity']
        if np.isnan(rvabs.data.data)[0]:
            rvabs = 0.

    bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=gaia_data['ra'], dec=gaia_data['dec'], pmra=gaia_data['pmra'], pmdec=gaia_data['pmdec'],
                               px=gaia_data['parallax'], rv=rvabs*1e3, epoch=epoch, obsname='AAO', ephemeris='de430')
    # bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=gaia_data['pmra'], pmdec=gaia_data['pmdec'],
    #                            px=gaia_data['parallax'], rv=gaia_data['radial_velocity']*1e3, obsname='AAO', ephemeris='de430')
    # bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=pmra, pmdec=pmdec,
    #                            px=px, rv=rv, obsname='AAO', ephemeris='de430')

    try:
        final_bc = bc[0][0][0]
    except:
        final_bc = bc[0][0]
        
    return final_bc



def get_bc_from_gaia(gaia_dr2_id, jd, rvabs=0):
    """
    wrapper routine for using barycorrpy with Gaia DR2 coordinates
    """
    
    # use 2015.5 as an epoch (Gaia DR2)
    epoch = 2457206.375

#     gaia_data = Gaia.query_object_async(coordinate=coord, width=width, height=height)
#     q = Gaia.launch_job_async('SELECT * FROM gaiadr2.gaia_source WHERE source_id = ' + str(gaia_dr2_id))
    q = Gaia.launch_job('SELECT * FROM gaiadr2.gaia_source WHERE source_id = ' + str(gaia_dr2_id))
    gaia_data = q.results

    bc = barycorrpy.get_BC_vel(JDUTC=jd, ra=gaia_data['ra'], dec=gaia_data['dec'], pmra=gaia_data['pmra'], pmdec=gaia_data['pmdec'],
                               px=gaia_data['parallax'], rv=rvabs*1e3, epoch=epoch, obsname='AAO', ephemeris='de430')
    # bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=gaia_data['pmra'], pmdec=gaia_data['pmdec'],
    #                            px=gaia_data['parallax'], rv=gaia_data['radial_velocity']*1e3, obsname='AAO', ephemeris='de430')
    # bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=pmra, pmdec=pmdec,
    #                            px=px, rv=rv, obsname='AAO', ephemeris='de430')

    return bc[0][0]



def append_bc_to_reduced_files(date, root='/Volumes/BERGRAID/data/veloce/'):
    
    print('Appending barycentric corrections to the reduced spectra of ' + str(date) + '...')
    
    acq_list, bias_list, dark_list, flat_list, skyflat_list, domeflat_list, arc_list, thxe_list, laser_list, laser_and_thxe_list, stellar_list, unknown_list = get_obstype_lists(root+'raw_goodonly/'+date+'/')
    stellar_list.sort()
    obsnames = short_filenames(stellar_list)
    object_list = [pyfits.getval(file, 'OBJECT').split('+')[0] for file in stellar_list]
    
    print('Calculating barycentric correction for stellar observation:')
    
    for i,(file,obsname) in enumerate(zip(stellar_list, obsnames)):
        
        print(str(i+1) + '/' + str(len(stellar_list)) + '   (' + object_list[i] + ')')
        
        bc = get_barycentric_correction(file)
        bc = np.round(bc,2)
        if np.isnan(bc):
            bc = ''
        # write the barycentric correction into the FITS header of both the quick-extracted and the optimal-extracted reduced spectrum files
        sublist = glob.glob(root + 'reduced/' + date + '/*' + obsname + '*extracted*')
        for outfn in sublist:
            pyfits.setval(outfn, 'BARYCORR', value=bc, comment='barycentric velocity correction [m/s]')   
    
    print('DONE!')
    
    return    
    
        
       



def old_get_barycentric_correction(fn, starname='tauceti', h=0.01, w=0.01):

    # wrapper routine for using barycorrpy with Gaia DR2 coordinates

    # use 2015.5 as an epoch (Gaia DR2)
    epoch = 2457206.375
    
    # get UT obs start time
    utmjd = pyfits.getval(fn, 'UTMJD') + 2.4e6 + 0.5   # the fits header has 2,400,000.5 subtracted!!!!!
    # add half the exposure time in days
    texp = pyfits.getval(fn, 'ELAPSED')
    utmjd = utmjd + (texp/2.)/86400.
    
    # ra = pyfits.getval(fn, 'MEANRA')
    # dec = pyfits.getval(fn, 'MEANDEC')
    if starname.lower() == 'tauceti':
        gaia_dr2_id = 2452378776434276992
        ra = 26.00930287666994
        dec = -15.933798650941204
        rv = -16.68e3
        h=0.01
        w=0.01
    elif starname.lower() == 'toi129':
        ra = 0.187097
        dec = -54.830506
        rv = 21.04070239e3
        h=0.005
        w=0.005
    elif starname.lower() == 'gj674':
        gaia_dr2_id = 5951824121022278144
        rv = -2.73
    else:
        fu=1
        assert fu != 1, 'ERROR: need to implement that first...'

    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(w, u.deg)
    height = u.Quantity(h, u.deg)

#     gaia_data = Gaia.query_object_async(coordinate=coord, width=width, height=height)
    q = Gaia.launch_job_async('SELECT * FROM gaiadr2.gaia_source WHERE source_id = ' + str(gaia_dr2_id))
    gaia_data = q.results

    bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=gaia_data['ra'], dec=gaia_data['dec'], pmra=gaia_data['pmra'], pmdec=gaia_data['pmdec'],
                               px=gaia_data['parallax'], rv=rv, epoch=epoch, obsname='AAO', ephemeris='de430')
    # bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=gaia_data['pmra'], pmdec=gaia_data['pmdec'],
    #                            px=gaia_data['parallax'], rv=gaia_data['radial_velocity']*1e3, obsname='AAO', ephemeris='de430')
    # bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=pmra, pmdec=pmdec,
    #                            px=px, rv=rv, obsname='AAO', ephemeris='de430')

    return bc[0][0]



def define_target_dict(starnames):
    target_dict = {}
    #add stars

    # for TAUCETI
    ra = 26.00930287666994
    dec = -15.933798650941204
    pmra = -1729.7257241911389
    pmdec = 855.492578244384
    px = 277.516215785613
    rv = -16.68e3

    # for TOI129
    ra = 0.187097
    dec = -54.830506
    pmra = -202.8150572
    pmdec = -71.51751871
    px = 16.15604552
    rv = 21.04070239e3

    # for TOI394
    ra = 48.49947914772984
    dec = -8.573851369791486
    pmra = -9.3210485208989
    pmdec = -76.26200762315906
    px = 6.994665452843984
    rv = 18.040873585123297e3

    return target_dict


