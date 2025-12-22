import os
from readFormulaFromCif import *
from cavd.local_environment import CifParser_new
# from monty import os
import pandas as pd
from ccnb.bvse_cavd import MigrationNetwork
from ccnb.mergecluster import Void, Channel
from ccnb.neb_packages import neb_packages
from ccnb.bvse import bv_calculation
from ccnb import (load_struc, load_bvse_from_npy,
                  load_voids_channels_from_file, get_bvse)
from ccnb.cavd_channel import cal_channel_cavd
from ccnb.MigrationNetworkLatticeSites import MigrationNetworkLatticeSites
import csv

if __name__ == "__main__":
    filedir = 'data/cavd_cif'  # Directory containing CIF files
    rightResult = []  # Store successful calculation results temporarily
    rightResults = []  # Store all successful calculation results
    errorFile = []  # Store filenames that caused errors
    for root, directories, files in os.walk(filedir):
        count = 0  # Counter for error files
        for filename in files:
            filename_CIF = os.path.join(root, filename)  # Full path of CIF file
            try:
                # 0. Extract ID from filename
                id = filename.split('.cif')[0]
                # 1. Calculate chemical formula
                formula = readFormulaFromCif(filename_CIF)
                if formula == 'NaCl':
                    print(f"Filename: {filename_CIF}, Formula: {formula}")
                # 2. Calculate RT (Relative Thermal conductivity?)
                RT = cal_channel_cavd(filename_CIF, 'Li', ntol=0.02, rad_flag=True, lower=0.4, upper=10.0,
                                      rad_dict=None)
                print(RT)
                # 3. Calculate BVSE (Bond Valence Site Energy)
                # barrier = get_bvse(filename_CIF, moveion='Li', valenceofmoveion=1, resolution=0.2)
                # print(barrier)

                # Collect results
                rightResult.append(id)
                rightResult.append(formula)
                rightResult.extend(RT)
                # rightResult.extend(barrier)

                # Save to list and CSV file
                rightResults.append(rightResult)
                f1 = open('result/Lai_CIF/right_results.csv', 'a', encoding='utf-8', newline='')
                rightwriter = csv.writer(f1)
                rightwriter.writerow(rightResult)
                f1.close()
                rightResult = []
                print(f'{filename} saved successfully')

            except Exception as e:
                # Handle errors
                errorFile.append(filename)
                f2 = open('result/Lai_CIF/error_file.csv', 'a', encoding='utf-8', newline='')
                errorwriter = csv.writer(f2)
                errorwriter.writerow(errorFile)
                f2.close()
                print("error file: ", filename)
                errorFile = []
                print(e)
                count += 1
                continue
    print("filename_CIF")
    print("error file count: ", count)
    print(rightResults)
    print(errorFile)

    # Convert to DataFrame and save to CSV
    #if no BVSE
    columns = ['id', 'formular', 'RT_1', 'RT_2', 'RT_3']
    #if BVSE
    # columns = ['id', 'formular', 'RT_1', 'RT_2', 'RT_3', 'BVSE_1d', 'BVSE_2d', 'BVSE_3d']
    rightResults = pd.DataFrame(columns=columns, data=rightResults)
    errorFile = pd.DataFrame(columns=['filename'], data=errorFile)
    rightResults.to_csv('result/first/RT_targets.csv', index=False)
    errorFile.to_csv('result/first/error_file.csv', index=False)
