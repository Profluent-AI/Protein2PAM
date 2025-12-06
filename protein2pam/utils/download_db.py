"""
This module provides functionality to download and extract the protein2pam database.
"""

import os
import urllib.request
import tarfile
from tqdm import tqdm

def download_and_extract_database():
    """Download and extract the protein2pam database."""
    database_url = "https://storage.googleapis.com/protein2pam-public/test.tar.gz"
    package_dir = os.path.dirname(os.path.dirname(__file__))  # Adjust as needed
    data_dir = os.path.join(package_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    tarball_path = os.path.join(data_dir, os.path.basename(database_url))

    try:
        # Step 1: Download the tarball if it doesn't exist
        if not os.path.exists(tarball_path):
            print(f"Downloading database file to: {tarball_path}")
            urllib.request.urlretrieve(database_url, tarball_path)
            print("Database download complete.")
        else:
            print("Database file already exists. Skipping download.")

        # Step 2: Extract files directly into `data/`
        print(f"Extracting contents of {tarball_path}")
        with tarfile.open(tarball_path, "r:gz") as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
                for member in members:
                    # Remove the "test/" prefix (or top-level directory prefix)
                    member_path = os.path.relpath(member.name, start=member.name.split('/')[0])
                    extracted_path = os.path.join(data_dir, member_path)
                    
                    if member.isdir():
                        os.makedirs(extracted_path, exist_ok=True)
                    else:
                        tar.extract(member, path=data_dir)
                        os.rename(os.path.join(data_dir, member.name), extracted_path)
                    
                    # Update progress bar
                    pbar.update(1)

        print("Database extraction complete.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_and_extract_database()
