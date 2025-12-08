import os
import mne
import pandas as pd
import numpy as np
from pathlib import Path

class TUSZDataset:
    def __init__(self, root_dir, seizure_types_path=None):
        """
        Args:
            root_dir (str): Path to the root directory containing patient folders (e.g., 'train').
            seizure_types_path (str, optional): Path to the xlsx file with seizure metadata.
        """
        self.root_dir = Path(root_dir)
        self.seizure_types_path = seizure_types_path
        self.sessions = self._find_sessions()

    def _find_sessions(self):
        """Recursively finds all EDF files and their corresponding CSV annotations."""
        sessions = []
        for file_path in self.root_dir.rglob("*.edf"):
            # Check for corresponding CSV file
            csv_path = file_path.with_suffix('.csv')
            if csv_path.exists():
                sessions.append({
                    'edf': str(file_path),
                    'csv': str(csv_path)
                })
        return sessions

    def load_session(self, idx):
        """
        Loads the EDF and CSV for a given index.
        
        Returns:
            raw (mne.io.Raw): Minimal loaded raw object.
            df (pd.DataFrame): Annotation dataframe.
        """
        if idx >= len(self.sessions):
            raise IndexError("Session index out of range")
        
        session = self.sessions[idx]
        
        # Load EDF
        # verbose=False to reduce clutter
        # preload=True is usually better for segmentation if memory allows, 
        # but for large dataset we might want False. Let's start with False.
        try:
            raw = mne.io.read_raw_edf(session['edf'], verbose=False, preload=False)
        except Exception as e:
            print(f"Error loading EDF {session['edf']}: {e}")
            return None, None

        # Load CSV
        try:
            # Skip commented lines if any, usually TUSZ csv has comments at top starts with #
            df = pd.read_csv(session['csv'], comment='#')
            # The columns are usually: channel, start_time, stop_time, label, confidence
            # Let's standardize column names just in case by stripping whitespace
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f"Error loading CSV {session['csv']}: {e}")
            return raw, None

        return raw, df

    def extract_segments(self, raw, annotations):
        """
        Segments the raw data based on annotations.
        
        Args:
            raw (mne.io.Raw): The EEG data.
            annotations (pd.DataFrame): The annotation dataframe.
            
        Returns:
            segments (list of dict): List of segments. Each dict contains:
                - 'data': numpy array of the segment
                - 'label': label of the segment
                - 'start_time': start time
                - 'stop_time': stop time
                - 'channels': channel names
        """
        if raw is None or annotations is None:
            return

        # We need to make sure we don't request times beyond the file duration
        max_time = raw.times[-1]

        for _, row in annotations.iterrows():
            start = row['start_time']
            stop = row['stop_time']
            label = row['label']
            
            # Basic validation
            if start >= max_time:
                continue
            if stop > max_time:
                stop = max_time
            if start >= stop:
                continue

            # Convert times to sample indices
            sfreq = raw.info['sfreq']
            start_idx = int(start * sfreq)
            stop_idx = int(stop * sfreq)

            # Load the data for this specific time range
            try:
                # get_data returns (n_channels, n_times)
                # start and stop are sample indices
                data, times = raw.get_data(start=start_idx, stop=stop_idx, return_times=True)
                
                yield {
                    'data': data,
                    'label': label,
                    'start_time': start,
                    'stop_time': stop,
                    'channels': raw.ch_names
                }
            except Exception as e:
                print(f"Error extracting segment {start}-{stop}: {e}")
                continue

if __name__ == "__main__":
    # fast test
    dataset = TUSZDataset(root_dir="../train")
    print(f"Found {len(dataset.sessions)} sessions.")
    if len(dataset.sessions) > 0:
        raw, df = dataset.load_session(0)
        print(f"Loaded session 0: {dataset.sessions[0]['edf']}")
        if df is not None:
             print("Annotations found:", len(df))
             segments = dataset.extract_segments(raw, df)
             print(f"Extracted {len(segments)} segments.")
             if len(segments) > 0:
                 print(f"First segment shape: {segments[0]['data'].shape}, label: {segments[0]['label']}")
