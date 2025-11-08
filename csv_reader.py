import glob 
import os
import pandas as pd

def read_csv_file(file_path='datasets/15k_ga_adaptive.csv'):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded '{file_path}'.")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst few rows:")
        df_reordered = df.loc[:, ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'matrix_size_x', 'matrix_size_y', 'performance']]
        print(df_reordered)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

if __name__ == "__main__":
    df = read_csv_file()


# ordre 6-5-9-10-1-2-8-3-4-11


