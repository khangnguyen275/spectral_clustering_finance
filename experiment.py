import pandas as pd
import numpy as np
from cluster import *
from preprocessing import *
from rmatrix import *
import argparse

path = '/Users/khang/Desktop/math285j_project/CRSP Data Set'
eligible_dates_list_path = path + '/eligible_dates.txt'

def main(preprocessing):
    print("Experiment script running.")
    
    if preprocessing:
        print("Preprocessing step enabled, extracting eligible dates from CRSP data set.")
        start_time = time.time()
        available_dates = get_all_date_paths(path)
        eligible_dates, ineligible_dates = get_eligible_date_paths(available_dates, eligible_dates_list_path)
        print(f"Eligible dates: {len(eligible_dates)}")
        print(f"Ineligible dates: {len(ineligible_dates)}")
        end_time = time.time()
        print(f"Preprocessing took: {end_time - start_time:.2f} seconds. Extracted {len(eligible_dates)} eligible dates.")
    else: 
        print("Preprocessing step skipped, getting eligible dates from existing file.")
        eligible_dates_txt_output = path + '/eligible_dates.txt'
        eligible_dates = get_eligible_date_paths_from_file(eligible_dates_txt_output)
        print(f"Loaded {len(eligible_dates)} eligible dates from file.")
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with optional preprocessing.")
    parser.add_argument('--preprocessing', action='store_true', help='Enable preprocessing step')
    args = parser.parse_args()
    main(preprocessing=args.preprocessing)