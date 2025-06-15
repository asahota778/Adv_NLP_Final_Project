# Reference Code
# Note - This code was used only to extract the data from Hugging Face validation set. The data was stored in a csv.


from pathlib import Path
import pandas as pd
import gzip
import os 

os.chdir(r"D:\BSE Semester 3\Advanced Methods in NLP\Adv NLP Final Project\big_patent\data\2.1.2\val")

def load_patent_corpus(root: str | Path,
                       fields=("publication_number", "abstract")) -> pd.DataFrame:
    """
    Build a DataFrame with the requested `fields` plus the folder name as `label`
    from a corpus organised as  root/<label>/*.gz  where each .gz is ND-JSON.
    """
    root = Path(root).expanduser().resolve()
    chunks = []                           # we'll append DataFrames here
    
    print(f"Looking for data in: {root}")  # Debug line
    
    if not root.exists():
        print(f"Error: Directory {root} does not exist!")
        return pd.DataFrame()
    
    # Check if there are any subdirectories
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    print(f"Found subdirectories: {[d.name for d in subdirs]}")  # Debug line
    
    for label_dir in sorted(subdirs):
        label = label_dir.name
        print(f"Processing label directory: {label}")  # Debug line
        
        gz_files = list(label_dir.glob("*.gz"))
        print(f"Found {len(gz_files)} .gz files in {label}")  # Debug line
        
        for gz_file in gz_files:
            print(f"Processing file: {gz_file}")  # Debug line
            try:
                # read in streaming mode → low memory even for big files
                for chunk in pd.read_json(gz_file,
                                          compression="gzip",
                                          lines=True,
                                          chunksize=50_000,   # tweak to taste
                                          encoding="utf-8"):
                    chunk = chunk.loc[:, list(fields)]        # keep only needed cols
                    chunk["label"] = label                    # add the folder label
                    chunks.append(chunk)
            except Exception as e:
                print(f"Error processing {gz_file}: {e}")
    
    if not chunks:
        print("No data was loaded. Check your directory structure and file paths.")
        return pd.DataFrame()
    
    return pd.concat(chunks, ignore_index=True)

# Example usage:
data_root_path = r"D:\BSE Semester 3\Advanced Methods in NLP\Adv NLP Final Project\big_patent\data\2.1.2\val"

# Call the function to load the data
patent_val_df = load_patent_corpus(data_root_path, fields=["publication_number", "abstract"])

# Print the first few rows to verify
if not patent_val_df.empty:
    print(patent_val_df.head())
    print(f"\nTotal rows loaded: {len(patent_val_df)}")
else:
    print("No data was loaded.")




########################### Load All Fields ############################

from pathlib import Path
import pandas as pd
import gzip
import os 

os.chdir(r"D:\BSE Semester 3\Advanced Methods in NLP\Adv NLP Final Project\big_patent\data\2.1.2\val")

def load_patent_corpus(root: str | Path,
                       fields=None) -> pd.DataFrame:  # Changed default to None
    """
    Build a DataFrame with the requested `fields` plus the folder name as `label`
    from a corpus organised as  root/<label>/*.gz  where each .gz is ND-JSON.
    """
    root = Path(root).expanduser().resolve()
    chunks = []                           # we'll append DataFrames here
    
    print(f"Looking for data in: {root}")  # Debug line
    
    if not root.exists():
        print(f"Error: Directory {root} does not exist!")
        return pd.DataFrame()
    
    # Check if there are any subdirectories
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    print(f"Found subdirectories: {[d.name for d in subdirs]}")  # Debug line
    
    for label_dir in sorted(subdirs):
        label = label_dir.name
        print(f"Processing label directory: {label}")  # Debug line
        
        gz_files = list(label_dir.glob("*.gz"))
        print(f"Found {len(gz_files)} .gz files in {label}")  # Debug line
        
        for gz_file in gz_files:
            print(f"Processing file: {gz_file}")  # Debug line
            try:
                # read in streaming mode → low memory even for big files
                for chunk in pd.read_json(gz_file,
                                          compression="gzip",
                                          lines=True,
                                          chunksize=50_000,   # tweak to taste
                                          encoding="utf-8"):
                    if fields:  # Only filter if specific fields requested
                        chunk = chunk.loc[:, list(fields)]        # keep only needed cols
                    chunk["label"] = label                    # add the folder label
                    chunks.append(chunk)
            except Exception as e:
                print(f"Error processing {gz_file}: {e}")
    
    if not chunks:
        print("No data was loaded. Check your directory structure and file paths.")
        return pd.DataFrame()
    
    return pd.concat(chunks, ignore_index=True)

# Example usage:
data_root_path = r"D:\BSE Semester 3\Advanced Methods in NLP\Adv NLP Final Project\big_patent\data\2.1.2\val"

# Load specific fields
patent_val_df = load_patent_corpus(data_root_path, fields=["publication_number", "abstract"])

# Load ALL fields
patent_val_df_all = load_patent_corpus(data_root_path)  # No fields parameter = load all

# Print results
if not patent_val_df_all.empty:
    print(f"All columns: {list(patent_val_df_all.columns)}")
    print(patent_val_df_all.head())
    print(f"\nTotal rows loaded: {len(patent_val_df_all)}")
else:
    print("No data was loaded.")