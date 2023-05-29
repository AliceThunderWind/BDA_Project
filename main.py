import csv
import numpy as np
import pandas as pd
import hdf5_getters
from utils import *
from hdf5_getters import *
import pyarrow as pa
import pyarrow.parquet as pq

files = get_all_files('data', ext='.h5')

# Specify the path to your CSV file
csv_path = 'data/data.csv'

# Create a list of variable names
header = [
    'artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude',
    'song_id', 'title', 'song_hotttnesss', 'similar_artists', 'artist_terms', 'artist_terms_freq',
    'artist_terms_weight', 'duration', 'time_signature', 'time_signature_confidence',
    'beats_start', 'beats_confidence', 'key', 'key_confidence', 'loudness', 'energy',
    'mode', 'mode_confidence', 'tempo', 'year'
]

# Open the CSV file in write mode
with open(csv_path, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()  # Write the header row

    # Iterate over the .h5 files
    for file in files:
        # Open the .h5 file using a context manager
        with hdf5_getters.open_h5_file_read(file) as data:
            # Select only the relevant features
            row = {
                'artist_id': hdf5_getters.get_artist_id(data),
                'artist_name': hdf5_getters.get_artist_name(data),
                'artist_location': hdf5_getters.get_artist_location(data),
                'artist_latitude': hdf5_getters.get_artist_latitude(data),
                'artist_longitude': hdf5_getters.get_artist_longitude(data),
                'song_id': hdf5_getters.get_song_id(data),
                'title': hdf5_getters.get_title(data),
                'song_hotttnesss': hdf5_getters.get_song_hotttnesss(data),
                'similar_artists': hdf5_getters.get_similar_artists(data),
                'artist_terms': hdf5_getters.get_artist_terms(data),
                'artist_terms_freq': hdf5_getters.get_artist_terms_freq(data),
                'artist_terms_weight': hdf5_getters.get_artist_terms_weight(data),
                'duration': hdf5_getters.get_duration(data),
                'time_signature': hdf5_getters.get_time_signature(data),
                'time_signature_confidence': hdf5_getters.get_time_signature_confidence(data),
                'beats_start': hdf5_getters.get_beats_start(data),
                'beats_confidence': hdf5_getters.get_beats_confidence(data),
                'key': hdf5_getters.get_key(data),
                'key_confidence': hdf5_getters.get_key_confidence(data),
                'loudness': hdf5_getters.get_loudness(data),
                'energy': hdf5_getters.get_energy(data),
                'mode': hdf5_getters.get_mode(data),
                'mode_confidence': hdf5_getters.get_mode_confidence(data),
                'tempo': hdf5_getters.get_tempo(data),
                'year': hdf5_getters.get_year(data)
            }
            
            for key, value in row.items():
                if isinstance(value, bytes):
                    row[key] = value.decode()
                elif isinstance(value, list):
                    row[key] = [item.decode() for item in value]
                elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.bytes_):
                    row[key] = [val.decode() for val in value]
                else:
                    row[key] = value

            writer.writerow(row)

csv_file = pd.read_csv(csv_path)

# Convert the DataFrame to a pyarrow Table
table = pa.Table.from_pandas(csv_file)

# Write the pyarrow Table to a Parquet file
pq.write_table(table, "data/data.parquet")









