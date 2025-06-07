"""
This script preprocesses the CRSP data set by validating file paths, filtering out dates with a high percentage of zero values in the 'pvCLCL' column, and saving the eligible and ineligible dates to text files.
Functions:
----------
get_ordered_csv_file_names(folder_path):
    Returns a sorted list of all '.csv.gz' file names in the specified folder.
get_all_date_paths(path):
    Iterates through year-based subfolders (2000-2021) in the given path, collecting full paths to all '.csv.gz' files. Returns a list of these file paths.
check_valid_file_paths(file_path_array):
    Checks if all paths in the provided array are valid files. Prints an error message for the first invalid path found, or confirms if all are valid.
get_eligible_date_paths(available_dates, output_path):
    For each file in available_dates, reads the CSV and calculates the percentage of zero values in the 'pvCLCL' column. Files with more than 10% zeros are considered ineligible. Saves eligible and ineligible file paths to separate text files and returns both lists.
Main Execution:
---------------
- Sets the data path and collects all date file paths.
- Validates the existence of all file paths.
- Filters eligible and ineligible dates based on the 'pvCLCL' zero percentage.
- Saves the results and prints summary statistics and execution duration.
"""

import sys
import os
import pandas as pd
import time

def get_ordered_csv_file_names(folder_path):
    all_file_names = os.listdir(folder_path)
    csv_file_names = []
    for file_name in all_file_names:
        if file_name.endswith('.csv.gz'):
            csv_file_names.append(file_name)
    csv_file_names.sort()
    return csv_file_names

def get_all_date_paths(path):
    eligible_dates = []
    for year in range(2000, 2022):
        # construct the path to each year's folder
        year_folder_name = str(year)
        year_folder_path = path + '/' + year_folder_name

        if os.path.isdir(year_folder_path):   # check if the folder exists
            # get all file names ordered by dates
            csv_file_names = get_ordered_csv_file_names(year_folder_path)
            eligible_dates += [year_folder_path + '/' + file_name for file_name in csv_file_names]
        else:
            print(f"Folder not found: {year_folder_name}")

    return eligible_dates

def check_valid_file_paths(file_path_array):
    all_paths_are_files = True
    for p in file_path_array:
        if not os.path.isfile(p):
            print(f"Error: {p} is not a valid file path.")
            all_paths_are_files = False
            break # Stop checking once the first invalid path is found

    if all_paths_are_files:
        print("All paths are valid file paths.")

def get_eligible_date_paths(available_dates, output_path):
    eligible_dates = []
    ineligible_dates = []
    for date in available_dates:
        df = pd.read_csv(date, compression='gzip')
        # Get the number of rows in the DataFrame
        total_rows = df.shape[0]
        # Count the number of zeros in the 'pvCLCL' column
        zero_count = (df['pvCLCL'] == 0).sum()
        # Calculate the percentage of zeros
        zero_percentage = (zero_count / total_rows)
        if zero_percentage > 0.1:
            ineligible_dates.append(date)
        else:
            eligible_dates.append(date)
    # Save eligible dates to a text file
    with open(output_path, 'w') as f:
        for date in eligible_dates:
            f.write(date + '\n')
    print(f"Eligible dates saved to {output_path}")
    # Save ineligible dates to a text file
    ineligible_dates_path = output_path.replace('eligible_dates.txt', 'ineligible_dates.txt')
    with open(ineligible_dates_path, 'w') as f:
        for date in ineligible_dates:
            f.write(date + '\n')
    print(f"Ineligible dates saved to {ineligible_dates_path}")
    # Return the lists of eligible and ineligible dates
    return eligible_dates, ineligible_dates

if __name__ == "__main__":
    start_time = time.time()
    possible_paths = [
        '/Users/khang/Desktop/math285j_project/CRSP Data Set',
        '/Users/lunjizhu/Desktop/MATH 285J Project Workspace/spectral_clustering_finance/data',
        'F:/spectral_clustering_finance/data/drive-download-20250531T145738Z-1-001/CRSP Data Set',
        '/Users/yifangu/Desktop/MATH 285J/285J Project/spectral_clustering_finance/data/CRSP Data Set'
    ]

    path = None
    for p in possible_paths:
        if os.path.isdir(p):
            path = p
            print(f"Using data path: {path}")
            break

    if path is None:
        print("Error: No valid data path found.")
        sys.exit(1)
        
    available_dates = get_all_date_paths(path)
    check_valid_file_paths(available_dates)
    eligible_dates_list_path = path + '/eligible_dates.txt'
    eligible_dates, ineligible_dates = get_eligible_date_paths(available_dates, eligible_dates_list_path)
    print(f"Eligible dates: {len(eligible_dates)}")
    print(f"Ineligible dates: {len(ineligible_dates)}")
    end_time = time.time()
    print(f"Duration: {end_time - start_time:.2f} seconds")