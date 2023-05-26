import hdf5_getters
h5 = hdf5_getters.open_h5_file_read("C:\\Users\\dalia\\Documents\\DMH\\04 HES-SO Master\\S2\\MA_BDA\\Projet\\BDA_Project\\MillionSongSubset\\B\\B\\B\\TRBBBLA128F424E963.h5")
duration = hdf5_getters.get_artist_id(h5)
print(duration)
h5.close()