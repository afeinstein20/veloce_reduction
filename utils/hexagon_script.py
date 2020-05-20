import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import astropy.io.fits as pyfits


def hexagon(dim, width, scale=1.5, offset=0.5, flat_top=True, interp_edge=True):
    """Adapted from code by Michael Ireland
    
    This function creates a hexagon.
    
    Parameters
    ----------
    dim: int
        Size of the 2D array
    width: int
        flat-to-flat width of the hexagon
        
    Returns
    -------
    pupil: float array (sz,sz)
        2D array hexagonal pupil mask
    """
    x = np.arange(dim)-dim/2.0
    xy = np.meshgrid(x,x)
    xx = xy[1]
    yy = xy[0]
    hex = np.zeros((dim,dim))
    
    if interp_edge:
        #!!! Not fully implemented yet. Need to compute the orthogonal distance 
        #from each line and accurately find fractional area of each pixel.
        hex = np.minimum(np.maximum(width/2 - yy + offset,0),1) * \
            np.minimum(np.maximum(width/2 + yy + offset,0),1) * \
            np.minimum(np.maximum((width-np.sqrt(3)*xx - yy + offset)*scale,0),1) * \
            np.minimum(np.maximum((width-np.sqrt(3)*xx + yy + offset)*scale,0),1) * \
            np.minimum(np.maximum((width+np.sqrt(3)*xx - yy + offset)*scale,0),1) * \
            np.minimum(np.maximum((width+np.sqrt(3)*xx + yy + offset)*scale,0),1)
    else:
        w = np.where( (yy < width/2) * (yy > (-width/2)) * \
              (yy < (width-np.sqrt(3)*xx)) * (yy > (-width+np.sqrt(3)*xx)) * \
              (yy < (width+np.sqrt(3)*xx)) * (yy > (-width-np.sqrt(3)*xx)))
        hex[w]=1.0
        
    if flat_top:
        hex = nd.rotate(hex, 90)
        
    return hex



def hexplot_brendan(param, obs, mode, label="Normalised Median Fibre Flux"):
    """
    Adapted from code by Michael Ireland
    """
    sz = 700
    scale = 0.95
    s32 = np.sqrt(3)/2
    lenslet_width = 0.4
    arcsec_pix = 0.0045
    yoffset = (lenslet_width/arcsec_pix*s32*np.array([-2,-2,-2,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,2,2,2])).astype(int)
    xoffset = (lenslet_width/arcsec_pix*0.5*np.array([-2,0,2,-3,-1,1,3,-4,-2,0,2,4,-3,-1,1,3,-2,0,2])).astype(int)
    im = np.zeros( (sz,sz,3) )

    for i in range(len(xoffset)):

        hexData = nd.shift(hexagon(sz, lenslet_width/arcsec_pix*scale),(yoffset[i], xoffset[i]))

        hexData = np.expand_dims(hexData, 2)

        hexData[:,:,0] *= param[i]/np.max(param)

        dim = np.zeros((sz,sz,1))
        hexData = np.concatenate((hexData,dim), axis=2)
        hexData = np.concatenate((hexData,dim), axis=2)

        hexData[:,:,1] = hexData[:,:,0]
        hexData[:,:,2] = hexData[:,:,0]

        im += hexData

    cmap = plt.get_cmap('jet')

    im = 1-im
    

    image = plt.imshow(im, cmap="gray_r", alpha=0.75)
    cbar = plt.colorbar(image)
    cbar.set_label(label, fontsize=14)
    plt.xticks([], [])
    plt.yticks([], [])
    
    if mode,lower() == "old":
        plt.title(obs[60:70], fontsize=14)
    if mode.lower() == "new":
        plt.title(obs[63:73], fontsize=14)
    plt.savefig("fluxFibresHex.pdf", bbox_inches='tight')
    
    plt.show()



def hexplot(param, obsname, mode='new', flat_top=True):
    """
    Adapted from code by Michael Ireland
    """
    # Mike's original values
#     sz = 700
#     lenslet_width = 0.4
#     arcsec_pix = 0.0045
    # CMB tests
    sz = 701
    lenslet_width = 0.52
    arcsec_pix = 0.0055
    
    # this is so that we have a little boundary between hexagons
    scale = 0.95
    s32 = np.sqrt(3)/2
   
#     yoffset = (lenslet_width/arcsec_pix*s32*np.array([-2,-2,-2,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,2,2,2])).astype(int)
#     xoffset = (lenslet_width/arcsec_pix*0.5*np.array([-2,0,2,-3,-1,1,3,-4,-2,0,2,4,-3,-1,1,3,-2,0,2])).astype(int)
    if flat_top:
        xoffset = (lenslet_width/arcsec_pix*s32*np.array([  1,  2,  1,  0,  0, -1, -1, -2, -2,  0, -2, -1, -1,  0,  1,  0,  2,  2,  1])).astype(int)
        yoffset = (lenslet_width/arcsec_pix*0.5*np.array([ -1, -2, -3, -2, -4, -3, -1, -2,  0,  0,  2,  3,  1,  4,  3,  2,  2,  0,  1])).astype(int)
    else:
        yoffset = (lenslet_width/arcsec_pix*s32*np.array([1, 2, 2, 1, 2, 1, 0, 1, 0, 0, 0, -1, 0, -1, -2, -1, -2, -2, -1])).astype(int)
        xoffset = (lenslet_width/arcsec_pix*0.5*np.array([-1, -2, 0, 1, 2, -3, -2, 3, -4, 0, 4, -3, 2, 3, -2, -1, 0, 2, 1])).astype(int)
#     im = np.zeros( (sz,sz,3) )
    im = np.zeros((sz,sz))

    for i in range(len(xoffset)):

        hexData = nd.shift(hexagon(sz, lenslet_width/arcsec_pix*scale, flat_top=flat_top),(yoffset[i], xoffset[i]))

#         hexData = np.expand_dims(hexData, 2)

        hexData *= param[i]/np.max(param)

#         hexData[:,:,0] *= param[i]/np.max(param)
#
#         dim = np.zeros((sz,sz,1))
#         hexData = np.concatenate((hexData,dim), axis=2)
#         hexData = np.concatenate((hexData,dim), axis=2)
# 
#         hexData[:,:,1] = hexData[:,:,0]
#         hexData[:,:,2] = hexData[:,:,0]

        im += hexData

#     cmap = plt.get_cmap('jet')
    cmap = plt.get_cmap('gray_r')
#     cmap = plt.get_cmap('binary')
    
#     im = 1-im
    
    # create a pretty plot
    plt.figure()
    plt.imshow(im, cmap=cmap, alpha=0.75, origin='lower', extent=np.array([-0.5,0.5,-0.5,0.5])*sz*arcsec_pix)
    plt.colorbar(label='Normalised Median Fibre Flux')
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.text(0,1.75,'E', ha='center', va='center', fontsize=14)
    plt.text(0,-1.75,'W', ha='center', va='center', fontsize=14)
    plt.text(-1.75,0,'S', ha='center', va='center', fontsize=14)
    plt.text(1.75,0,'N', ha='center', va='center', fontsize=14)
    dum = obsname.split('/')
    plt.title(dum[-1].split('_')[1], fontsize=14)
    
#     ###########################
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.imshow(im, cmap=cmap, alpha=0.75, origin='lower', extent=np.array([-0.5,0.5,-0.5,0.5])*sz*arcsec_pix)
#     ax.set_xlabel('arcsec')
#     ax.set_ylabel('arcsec')
#     fig.colorbar(im, label='Normalised Median Fibre Flux')
#     ax.set_title(obsname.split('_')[-3], fontsize=14)
#     ###########################

    
   
#     plt.savefig("fluxFibresHex.pdf", bbox_inches='tight')
#     print('haehaehae2')
    
#     plt.show()

    return



def hexplotobs(obs, mode="new"):
    """
    # usage:
    hexplotobs(filepath + "HD10700+ThXe+LFC_21sep30108_optimal3a_extracted.fits")
    """
    fluxVals = pyfits.getdata(obs) 
    if mode.lower() == "new":
#         fluxFibres = np.nanmedian(fluxVals[:,2:21:],(0, 2))
        fluxFibres = np.nanmedian(fluxVals[:,3:22:], axis=(0,2))
    if mode.lower() == "old":  
        fluxFibres = np.nanmedian(fluxVals,(0, 2))
    hexplot(fluxFibres, obs, mode)
    return

