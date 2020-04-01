import os
import tarfile
import requests

def check_dataset(dataset_url, data_path):
    if not os.path.exists(data_path):
        print(f"Dataset not found at "+data_path+".\n"
              f"Downloading from {dataset_url}")
        # Download dataset to data directory
        os.makedirs('../data', exist_ok=True)
        data_gz_path = data_path.split('/*')[0] + '.tar.gz'
        with requests.get(dataset_url, stream=True) as r:
            r.raise_for_status()
            with open(data_gz_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("Decompressing...")
        f_in = tarfile.open(data_gz_path, 'r:gz')
        f_in.extractall('../data')

        print("Deleting compressed file...")
        os.remove(data_gz_path)

        print('done!')
    else:
        print(f"Found dataset at {data_path}")

census_data_url = 'https://s3.us-east-2.amazonaws.com/rapidsai-data/viz-data/census_data_minimized.parquet.tar.gz'
census_data_path = "../data/census_data_minimized.parquet"
check_dataset(census_data_url, census_data_path)

acs_data_url = 'https://s3.us-east-2.amazonaws.com/rapidsai-data/viz-data/acs2018_county_population.parquet.tar.gz'
acs2018_data_path = "../data/acs2018_county_population.parquet"
check_dataset(acs_data_url, acs2018_data_path)