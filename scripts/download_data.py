import urllib.request
from pathlib import Path
import ssl

def main():
    # Bypasses strict SSL verification which sometimes blocks Windows machines
    ssl._create_default_https_context = ssl._create_unverified_context
    
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63990/matrix/GSE63990_series_matrix.txt.gz"
    dest = Path("data/raw/GSE63990_series_matrix.txt.gz")
    
    if not dest.exists():
        print(f"Downloading GSE63990 Microarray Matrix from NCBI GEO...")
        print("This is a large genomic file. Please wait a minute...")
        urllib.request.urlretrieve(url, dest)
        print("Download complete! Saved to data/raw.")
    else:
        print("File already exists.")

if __name__ == "__main__":
    main()