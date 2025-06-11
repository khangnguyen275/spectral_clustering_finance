import numpy as np

def get_eligible_date_paths_from_file(file_path):
    """
    Reads a file containing paths (one per line) and returns a list of cleaned paths.
    Args:
        file_path (str): The path to the file containing eligible date paths.
    Returns:
        list: A list of strings, each representing a cleaned path read from the file.
              Returns an empty list if the file is not found or an error occurs.
    Side Effects:
        Prints the number of successfully read paths or error messages to the console.
    """
    
    eligible_dates_from_file = []
    # Open the file in read mode ('r')
    try:
        with open(file_path, 'r') as f:
            # Read each line from the file
            for line in f:
                # Remove any leading or trailing whitespace (including the newline character)
                cleaned_line = line.strip()
                # Add the cleaned line (path) to the list
                eligible_dates_from_file.append(cleaned_line)

        print(f"Successfully read {len(eligible_dates_from_file)} paths from {file_path}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    return eligible_dates_from_file

def winsorize(data, lower_pct=0.01, upper_pct=0.99):
    """
    Winsorize a list or array of numbers by replacing values below the lower 
    percentile and above the upper percentile with the respective percentile values.

    Parameters:
        data (list or np.ndarray): The input numeric data.
        lower_pct (float): Lower percentile threshold (e.g., 0.05 for 5%).
        upper_pct (float): Upper percentile threshold (e.g., 0.95 for 95%).

    Returns:
        np.ndarray: Winsorized version of the input data.
    """
    data = np.asarray(data)
    lower_bound = np.percentile(data, lower_pct * 100)
    upper_bound = np.percentile(data, upper_pct * 100)
    
    return np.clip(data, lower_bound, upper_bound)