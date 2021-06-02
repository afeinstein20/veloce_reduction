import pyfits
import matplotlib.pyplot as plt
import numpy as np

optimal_extracted_filename = "/Users/Brendan/Downloads/Achenar+ThXe_17sep30121_optimal3a_extracted.fits.txt"

data = pyfits.getdata(optimal_extracted_filename, 0)
err = pyfits.getdata(optimal_extracted_filename, 1)

def remove_nan_orders(data, err):
    outdata = []
    outerr = []
    for order in np.arange(len(data)):
        if np.isnan(data[order]).any():
            continue
        else:
            outdata.append(data[order])
            outerr.append(err[order])
    return outdata, outerr

def combine_fibres(data, err, m=1):
    """
    Perform a weighted average over all fibre orders excluding outliers
    """
    combined_data = []

    for order in np.arange(len(data)):
        cleaned_data = []
        cleaned_err = []
        mean = np.mean(data[order])
        std = np.std(data[order])

        for fibre in np.arange(len(data[order])):
            outlier = False

            for pixel in data[order][fibre]:
                if abs(pixel - mean) > m*std: #Checking for outlier pixels
                    outlier=True

            if outlier==False:
                cleaned_data.append(data[order][fibre])
                cleaned_err.append(err[order][fibre])

        #Weighted average
        combined_data.append(np.average(np.array(cleaned_data), weights=np.array(cleaned_err), axis=0))
    return np.array(combined_data)

def plot_orders(combined_data, data, orders=20):
	for i in np.arange(20):
		medData = np.median(data, axis=1)
		plt.figure(figsize=(7,4))
		plt.xlabel("Pixel Number", fontsize=14)
		plt.ylabel("Normalised Intensity", fontsize=14)
		plt.plot(medData[i]/np.max(medData[i]), label="Median")
		plt.plot(combined_data[i]/np.max(combined_data[i]), label="Outlier Removed Weighted Averaged")
		plt.legend(fontsize=12)
		plt.show()

data, err = remove_nan_orders(data, err)
combined_data = combine_fibres(data, err)
plot_orders(combined_data, data)
