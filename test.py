import hdf5_getters
from utils import *
from hdf5_getters import *
    
files = get_all_files('data',ext='.h5')

# Open the .h5 file in read-only mode
for file in files[:1]:
    data = hdf5_getters.open_h5_file_read(file)
    print(data)
    artist_id = hdf5_getters.get_artist_id(data)
    artist_lat = hdf5_getters.get_artist_latitude(data)
    print(artist_lat)