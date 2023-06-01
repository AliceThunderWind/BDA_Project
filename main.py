import numpy as np
import pandas as pd
from utils import *
from hdf5_getters import *
import pyarrow as pa
import time
import pyarrow.parquet as pq
from geopy.geocoders import Nominatim
import requests
import multiprocessing
from tqdm import tqdm


# *****************************************************************************
#                  GLOBAL VARIABLES
# *****************************************************************************

files = get_all_files('data', ext='.h5')
csv_path = 'data/data.csv'
csv_file = pd.read_csv(csv_path)
copy_file = csv_file.copy()
api_key = "059e637024c2da6d558a09dfa118a79a"
global i = 0


# *****************************************************************************
#                  FUNCTIONS
# *****************************************************************************

# Get the country of an artist based on his Long and Lat coordinates
def get_country(latitude, longitude, df):
    if latitude is None or longitude is None or np.isnan(latitude) or np.isnan(longitude):
        df['artist_location'][i] = np.nan
        i += 1
        return np.nan
    geolocator = Nominatim(user_agent="geo_app")
    location = geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True)
    if location is not None and 'address' in location.raw:
        df['artist_location'][i] = location.raw['address'].get('country', '')
        i += 1
        return location.raw['address'].get('country', '')
    return None

# Get the music genre of an artiste based on his name and by using Lastfm's API
def get_artist_genre(artist_name, df):
    base_url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "artist.getinfo",
        "artist": artist_name,
        "api_key": api_key,
        "format": "json"
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            artist = data["artist"]
            if "tags" in artist and "tag" in artist["tags"]:
                df['artist_genre'][i] = [tag["name"] for tag in artist["tags"]["tag"]]
                i += 1
                return
        except:
            df['artist_genre'][i] = np.nan
            i += 1
            return
    else:
        print('Error 6: Failed request')
        return 

# Multiprocessing function
def process_chunk(chunk):
    results = []
    for row in chunk.itertuples(index=False):
        time.sleep(3)
        result = get_country(row.artist_latitude, row.artist_longitude)
        if result is np.nan or None:
            results.append(np.nan)
            i+=1
        else:
            results.append(result)
            i+=1
    return results

# Data augmentation functions
def data_augmentation_location(df):
    return df.apply(lambda row: get_country(row['artist_latitude'], row['artist_longitude']), axis=1)

def data_augmentation_genre(df):
    return df.apply(lambda row: get_artist_genre(row['artist_name']), axis=1)


# *****************************************************************************
#                  DATA AUGMENTATION - ANALYTICS
# *****************************************************************************

# # Create a list of variable names
# header = [
#     'artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude',
#     'song_id', 'title', 'song_hotttnesss', 'similar_artists', 'artist_terms', 'artist_terms_freq',
#     'artist_terms_weight', 'duration', 'time_signature', 'time_signature_confidence',
#     'beats_start', 'beats_confidence', 'key', 'key_confidence', 'loudness', 'energy',
#     'mode', 'mode_confidence', 'tempo', 'year'
# ]

# # Open the CSV file in write mode
# with open(csv_path, 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=header)
#     writer.writeheader()  # Write the header row
#     # Iterate over the .h5 files
#     for file in files:
#         # Open the .h5 file using a context manager
#         with hdf5_getters.open_h5_file_read(file) as data:
#             # Select only the relevant features
#             row = {
#                 'artist_id': hdf5_getters.get_artist_id(data),
#                 'artist_name': hdf5_getters.get_artist_name(data),
#                 'artist_location': hdf5_getters.get_artist_location(data),
#                 'artist_latitude': hdf5_getters.get_artist_latitude(data),
#                 'artist_longitude': hdf5_getters.get_artist_longitude(data),
#                 'song_id': hdf5_getters.get_song_id(data),
#                 'title': hdf5_getters.get_title(data),
#                 'song_hotttnesss': hdf5_getters.get_song_hotttnesss(data),
#                 'similar_artists': hdf5_getters.get_similar_artists(data),
#                 'artist_terms': hdf5_getters.get_artist_terms(data),
#                 'artist_terms_freq': hdf5_getters.get_artist_terms_freq(data),
#                 'artist_terms_weight': hdf5_getters.get_artist_terms_weight(data),
#                 'duration': hdf5_getters.get_duration(data),
#                 'time_signature': hdf5_getters.get_time_signature(data),
#                 'time_signature_confidence': hdf5_getters.get_time_signature_confidence(data),
#                 'beats_start': hdf5_getters.get_beats_start(data),
#                 'beats_confidence': hdf5_getters.get_beats_confidence(data),
#                 'key': hdf5_getters.get_key(data),
#                 'key_confidence': hdf5_getters.get_key_confidence(data),
#                 'loudness': hdf5_getters.get_loudness(data),
#                 'energy': hdf5_getters.get_energy(data),
#                 'mode': hdf5_getters.get_mode(data),
#                 'mode_confidence': hdf5_getters.get_mode_confidence(data),
#                 'tempo': hdf5_getters.get_tempo(data),
#                 'year': hdf5_getters.get_year(data)
#             }          
#             for key, value in row.items():
#                 if isinstance(value, bytes):
#                     row[key] = value.decode()
#                 elif isinstance(value, list):
#                     row[key] = [item.decode() for item in value]
#                 elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.bytes_):
#                     row[key] = [val.decode() for val in value]
#                 else:
#                     row[key] = value
#             writer.writerow(row)

# # Convert the DataFrame to a pyarrow Table
# table = pa.Table.from_pandas(csv_file)

# # Write the pyarrow Table to a Parquet file
# pq.write_table(table, "data/data.parquet")

# # Begin multiprocessing
# if __name__ == '__main__':
#     # Split the data into chunks for parallel processing
#     num_processes = multiprocessing.cpu_count()
#     chunks = np.array_split(copy_file, num_processes)

#     # Create a pool of worker processes
#     pool = multiprocessing.Pool(processes=num_processes)

#     # Apply the process_chunk function to each chunk of data
#     results = pool.map(process_chunk, chunks)

#     # Combine the results from all worker processes
#     all_results = [result for chunk_results in results for result in chunk_results]

#     # Close the pool of worker processes
#     pool.close()
#     pool.join()
