import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from data_loader import TUSZDataset

def process_dataset(root_dir, output_dir, limit=None):
    """
    Processes the dataset and saves segments to disk.
    
    Args:
        root_dir (str): Path to TUSZ train directory.
        output_dir (str): Path to save processed data.
        limit (int, optional): Max number of sessions to process.
    """
    dataset = TUSZDataset(root_dir=root_dir)
    print(f"Found {len(dataset.sessions)} sessions.")
    
    if limit:
        print(f"Limiting to first {limit} sessions.")
        sessions_to_process = range(min(limit, len(dataset.sessions)))
    else:
        sessions_to_process = range(len(dataset.sessions))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load previously processed sessions
    log_file = output_path / "processed_sessions.txt"
    if log_file.exists():
        with open(log_file, 'r') as f:
            processed_set = set(f.read().splitlines())
    else:
        processed_set = set()

    success_count = 0
    fail_count = 0
    total_segments = 0

    for i in tqdm(sessions_to_process, desc="Processing Sessions"):
        # Check if already processed
        session_path = Path(dataset.sessions[i]['edf'])
        session_name = session_path.stem
        
        if session_name in processed_set:
            continue

        raw, annotations = dataset.load_session(i)
        
        if raw is None or annotations is None:
            fail_count += 1
            continue
            
        try:
            segments = dataset.extract_segments(raw, annotations)
            
            # Save each segment
            for j, seg in enumerate(segments):
                label = seg['label'].lower().strip() # Normalize label
                
                # Filter out background
                if label == 'bckg':
                    continue

                # Create label directory
                label_dir = output_path / label
                label_dir.mkdir(exist_ok=True)
                
                # Save segment
                filename = f"{session_name}_seg{j}.npy"
                file_path = label_dir / filename
                
                np.save(file_path, seg['data'])
                total_segments += 1
            
            # Mark as done
            with open(log_file, 'a') as f:
                f.write(session_name + "\n")
            processed_set.add(session_name)
            success_count += 1

        except Exception as e:
            print(f"Error processing session {session_name}: {e}")
            fail_count += 1
        
        # Explicit cleanup
        if 'raw' in locals(): del raw
        if 'segments' in locals(): del segments
        # import gc; gc.collect() 

    print("\n--- Processing Complete ---")
    print(f"Sessions Processed: {success_count}")
    print(f"Sessions Failed: {fail_count}")
    print(f"Total Segments Saved: {total_segments}")
    print(f"Output Directory: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TUSZ dataset into .npy files")
    parser.add_argument("--root", type=str, default="../train", help="Path to TUSZ train directory")
    parser.add_argument("--output", type=str, default="../processed_data/train", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of sessions to process")
    
    args = parser.parse_args()
    
    process_dataset(root_dir=args.root, output_dir=args.output, limit=args.limit)
