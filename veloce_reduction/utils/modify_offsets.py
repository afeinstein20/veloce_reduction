from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import pdb
SPEED_CODES = {0:'Slow', 1:'Std', 2:'Eng', 3:'Turbo'}
GAINS = [1,2,5,10]

def extract_from_gain_table(fn, gains=GAINS, new_offsets=np.ones( (4,4,4), dtype=int) * 2048, outfile=None):
    offsets = new_offsets.copy()
    with open(fn) as file:
        lines = file.readlines()
        #offset_lines = [l for l in lines if l.find('OffsetAt') == 0]
        for line in lines:
            if line.find('OffsetAt') < 0:
                if outfile is not None:
                    outfile.write(line)
                continue
            else:
                amp = int(line[line.find('Amp')+3])
                gain = gains.index(int(line[line.find('.G')+2:line.find('.Amp')]))
                offset = int(line[line.find('=')+2:-1])
                if line.find('Slow')>0:
                    speed=0
                elif line.find('Std')>0:
                    speed=1
                elif line.find('Eng')>0:
                    speed=2
                elif line.find('Turbo')>0:
                    speed=3
                offsets[speed, gain, amp] = offset
                if outfile is not None:
                    outfile.write("OffsetAt." + SPEED_CODES[speed] + '.G{:d}.Amp{:d} = {:d}\n'.format(gains[gain], amp, new_offsets[speed, gain, amp]))
    return offsets
    
if __name__=="__main__":
    dir = '/Users/mireland/tel/veloce/'
    bias_levels = np.loadtxt(dir + 'offsets5.csv', delimiter=',', dtype=int)
    offsets = extract_from_gain_table(dir + 'velocesvr/config/rosso-gain-table.txt')
    for b in bias_levels:
        offsets[b[0], GAINS.index(b[1]), :] += ((b[2:] - 500)/9.53).astype(int)
    dummy_file = open(dir + 'velocesvr/config/dummy.txt','w')
    offsets = extract_from_gain_table(dir + 'velocesvr/config/rosso-gain-table.txt', outfile=dummy_file, new_offsets=offsets)
    dummy_file.close()
    print(offsets)
    
    