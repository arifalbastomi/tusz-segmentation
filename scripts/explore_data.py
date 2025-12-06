import os
import mne
import pandas as pd

def find_first_edf(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".edf"):
                return os.path.join(root, file)
    return None

def main():
    # 1. Find an EDF file
    train_dir = "train"
    print(f"Searching for EDF files in {train_dir}...")
    edf_path = find_first_edf(train_dir)
    
    if not edf_path:
        print("No EDF files found!")
        return

    print(f"Found EDF file: {edf_path}")

    # 2. Load EDF info
    try:
        raw = mne.io.read_raw_edf(edf_path, verbose=False)
        print("\n--- EDF Info ---")
        print(raw.info)
        print(f"Duration: {raw.times[-1]} seconds")
        print(f"Sample Rate: {raw.info['sfreq']} Hz")
        print(f"Channels: {raw.ch_names[:5]} ... (total {len(raw.ch_names)})")
    except Exception as e:
        print(f"Error reading EDF: {e}")

    # 3. Load Metadata
    meta_path = "seizures_types_v02.xlsx"
    if os.path.exists(meta_path):
        print(f"\nLoading metadata from {meta_path}...")
        try:
            # Load the collection of sheets to see what's inside or just load the first one
            xls = pd.ExcelFile(meta_path)
            print(f"Sheet names: {xls.sheet_names}")
            
            # Assuming data is in the first sheet or a sheet named 'Sheet1' or similar. 
            # Let's inspect the first sheet.
            df = pd.read_excel(meta_path, sheet_name=0)
            print(f"Metadata shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            
            # Simple check if the filename (without path) is in the metadata
            # The metadata usually contains generic session info. 
            # We might need to match by file stem.
            filename = os.path.basename(edf_path)
            # Remove extension for matching if needed, or keep it.
            # TUSZ metadata often references the 'TCP' or 'AR' montage reference or session ID.
            
            # Let's just print the head to see structure
            print("\nFirst 5 rows of metadata:")
            print(df.head())
            
        except Exception as e:
            print(f"Error reading metadata: {e}")
    else:
        print(f"Metadata file {meta_path} not found.")

if __name__ == "__main__":
    main()
