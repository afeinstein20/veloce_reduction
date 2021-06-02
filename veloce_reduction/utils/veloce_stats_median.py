import pandas as pd
import numpy as np
import pyfits
import glob
from astroquery.vizier import Vizier
import time

location = "/data/malice/brendano/"

def stats(directories):

    not_source_arr = ["test", "ThAr", "Flat", "Dark Frame", "Bias Frame", "Nothing", "ThXe", "ThXe test", "CuAr", "FLAT", "ARC - ThAr", "FOCUS", "Flat Field - Quartz", "EpsSGRTelluric", "Archenar", "testdark", "alphaGru", "Archenar Acq"]

    objectArr = []
    magArr = []
    expArr = []
    fluxMedArr = []
    flux95Arr = []
    flux99Arr = []
    MMedArr = []
    M95Arr = []
    M99Arr = []


    filterArr = ["Umag", "Vmag", "Bmag", "Imag", "Rmag", "mpg"]

    for i in directories:
        for extracted_file in glob.glob(i):
            if "14aug" in extracted_file:
                original_file = "/data/malice/brendano/veloce/180814/ccd_3/"+extracted_file[26:len(extracted_file)-15]+".fits"
            elif "15aug" in extracted_file:
                original_file = "/data/malice/brendano/veloce/180815/ccd_3/"+extracted_file[26:len(extracted_file)-15]+".fits"
    
            object_name = pyfits.getheader(original_file)["OBJECT"]
            print(object_name)
            if object_name not in not_source_arr:
                if "Acq" in object_name:
                    query_name = object_name[:len(object_name)-3]
                else:
                    query_name = object_name
                result_table = Vizier.query_object(query_name)

                for i in result_table[2].keys():
                    if i in filterArr:
                        mag = result_table[2][0][i]
                        objectArr.append(pyfits.getheader(original_file)["OBJECT"])

                        magArr.append(mag)
                        expArr.append(pyfits.getheader(original_file)["EXPOSED"])
                        
                        fluxData = pyfits.getdata(extracted_file)
                        for flux in fluxData:
                            fluxMedArr.append(np.median(flux))   
                            flux95Arr.append(np.percentile(flux, 95))   
                            flux99Arr.append(np.percentile(flux, 99))

                        MMedArr.append(np.median(fluxMedArr))
                        M95Arr.append(np.median(flux95Arr))
                        M99Arr.append(np.median(flux99Arr))

    rvArr = np.zeros(len(objectArr))
    rvErrArr = np.zeros(len(objectArr))

    columns=['Object', 'Magnitude', 'Median Flux', '95% Flux', '99% Flux', 'Exposure Time', 'Radial Velocity', 'RV Uncertainty']
    df = pd.DataFrame({'Object': objectArr, 'Magnitude': magArr, 'Median Flux': MMedArr, '95% Flux': M95Arr, '99% Flux': M99Arr, 'Exposure Time': expArr, 'Radial Velocity': rvArr, 'RV Uncertainty': rvErrArr}, columns=columns)
                      
    print(df)

    df.to_csv("malice_"+"stats.csv")

start = time.clock()

stats([location+"*extracted*"])

end = time.clock()

print("Time Elapsed: ",end-start)
