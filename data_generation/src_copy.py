# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:10:44 2021

@author: nhaq
"""

import numpy as np
import path
from pathlib import Path
import os
import glob

src_dir = 'D:/pulsatility_mapping/PulseMapper3TE/Nick/Images MC/'
target_dir = 'D:/pulsatility_mapping/PulseMapper3TE/data/Images MC/'



echo_index =[1,2,3]

procs=[f'ica{i}' for i in range(9)]

sblist = os.listdir(target_dir)


for sbdir in sblist:
    print(f"participant: {sbdir}")
    for procdir in procs:
        src_fullpath = f"{src_dir}/{sbdir}/{procdir}/"
        target_fullpath = f"{target_dir}/{sbdir}/{procdir}/"
        
        for ec in echo_index:
            src_files = glob.glob(f"{src_fullpath}/{ec}_*")
            for filepath in src_files:
                data = np.load(filepath)
                target_file = f"{target_fullpath}/{os.path.basename(filepath)}"
                np.save( target_file,   np.fliplr(data))
    
