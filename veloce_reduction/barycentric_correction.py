import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import astropy.io.fits as pyfits
import barycorrpy


# starnames = ['HD10700', 'HD190248', 'Gl87', 'GJ1132', 'GJ674', 'HD194640', 'HD212301']


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



def get_barycentric_correction(fn, starname='tauceti', h=0.01, w=0.01):

    # wrapper routine for using barycorrpy with Gaia DR2 coordinates

    # use 2015.5 as an epoch (Gaia DR2)
    epoch = 2457206.375
    
    # get UT obs start time
    utmjd = pyfits.getval(fn, 'UTMJD') + 2.4e6 + 0.5   # the fits header has 2,400,000.5 subtracted!!!!!
    # add half the exposure time in days
    texp = pyfits.getval(fn, 'TOTALEXP')
    utmjd = utmjd + (texp/2.)/86400.
    
    # ra = pyfits.getval(fn, 'MEANRA')
    # dec = pyfits.getval(fn, 'MEANDEC')
    if starname.lower() == 'tauceti':
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
    else:
        fu=1
        assert fu != 1, 'ERROR: need to implement that first...'

    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(w, u.deg)
    height = u.Quantity(h, u.deg)

    gaia_data = Gaia.query_object_async(coordinate=coord, width=width, height=height)

    bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=gaia_data['pmra'], pmdec=gaia_data['pmdec'],
                               px=gaia_data['parallax'], rv=rv, epoch=epoch, obsname='AAO', ephemeris='de430')
    # bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=gaia_data['pmra'], pmdec=gaia_data['pmdec'],
    #                            px=gaia_data['parallax'], rv=gaia_data['radial_velocity']*1e3, obsname='AAO', ephemeris='de430')
    # bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, pmra=pmra, pmdec=pmdec,
    #                            px=px, rv=rv, obsname='AAO', ephemeris='de430')

    return bc[0][0]

