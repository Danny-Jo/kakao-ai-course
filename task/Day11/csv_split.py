import pandas as pd

# Load the large CSV file
file_path = 'creditcard.csv'
df = pd.read_csv(file_path)

# Determine the number of rows per split file
total_rows = len(df)
rows_per_file = total_rows // 3

# Create file names for the smaller CSV files
output_files = ['creditcard_part1.csv', 'creditcard_part2.csv', 'creditcard_part3.csv']

# Split the dataframe and save each part to a new CSV file
for i, output_file in enumerate(output_files):
    start_idx = i * rows_per_file
    # Make sure the last file contains all remaining rows
    if i == len(output_files) - 1:
        end_idx = total_rows
    else:
        end_idx = (i + 1) * rows_per_file
    
    df_part = df.iloc[start_idx:end_idx]
    df_part.to_csv(output_file, index=False)

print(f"Successfully split {file_path} into {len(output_files)} files.")
