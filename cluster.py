import os
from os import listdir
ImgResols = [8]
clusterNs = [5]#, 10, 15]

for ir in ImgResols:
    os.system(f'python process_logo_data_logo2k.py --ImgResol {ir} --clusterN {clusterNs[0]}') #{clusterNs[1]} {clusterNs[2]}')
