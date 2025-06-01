def get_eligible_date_paths_from_file(file_path):
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