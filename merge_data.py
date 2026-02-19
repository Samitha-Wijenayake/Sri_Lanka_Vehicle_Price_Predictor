import glob
import pandas as pd
import os

def merge_csvs():
    # Find all partial csv files
    # Find all partial csv files
    files = glob.glob("vehicle_data_*_*.csv")
    if os.path.exists("vehicle_data.csv"):
        files.append("vehicle_data.csv")
    print(f"Found {len(files)} files to merge: {files}")
    
    dfs = []
    for f in files:
        try:
            # Check if file is not empty
            if os.path.getsize(f) > 0:
                df = pd.read_csv(f)
                if not df.empty:
                    dfs.append(df)
                    print(f"Loaded {len(df)} rows from {f}")
            else:
                print(f"Skipping empty file: {f}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        # Drop duplicates if any
        initial_len = len(combined)
        # Drop duplicates based on content if Link is missing
        # combined.drop_duplicates(subset=['Link'], inplace=True) 
        combined.drop_duplicates(subset=['Title', 'Price', 'Mileage', 'Location'], inplace=True)
        print(f"Dropped {initial_len - len(combined)} duplicates.")
        
        combined.to_csv("vehicle_data_withoutprocessed.csv", index=False)
        print(f"Saved {len(combined)} rows to vehicle_data_withoutprocessed.csv")
    else:
        print("No data found to merge.")

if __name__ == "__main__":
    merge_csvs()
