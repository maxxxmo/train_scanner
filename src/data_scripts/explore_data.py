import pandas as pd
path_h5 = "./data/data/datasets/frsign/FRSign_modified/FRSign/frsign_v1.0.h5"
store = pd.HDFStore(path_h5, mode='r')

# all tables in the HDF5 file
print("Keys available :", store.keys())

# charge the dataframe from the HDF5 file
df = store['dataframe'] 

# Look into the structure of the DataFrame
print("\n--- Structure of DataFrame ---")
print(df.info())

# Show an overview of the data
print("\n--- Data overview ---")
print(df.head())

# now we do the same with images
# charge the images from the HDF5 file
print("\n--- Exploring images ---")
images = store['images']
print("fullpath of first image:", images['fullpath'].iloc[0])
print(images.columns)
print(images.head())
print("Shape of images dataset:", images.shape)

