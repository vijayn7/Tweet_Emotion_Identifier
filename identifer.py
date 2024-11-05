import pandas as pd

def print_first_lines(file_path, num_lines=2):
    try:
        df = pd.read_parquet(file_path)
        print(df.head(num_lines))
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'training_data.parquet' with the actual path to your Parquet file
print_first_lines('training_data.parquet', 10)