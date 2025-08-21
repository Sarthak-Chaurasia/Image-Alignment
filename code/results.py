import numpy as np
from PIL import Image
import sys
import os
import pickle

image_path = sys.argv[1]
image = Image.open(image_path).convert('L')
image = np.array(image)

recons_path = "./reconstructed_image.jpg"
recons_image = Image.open(recons_path).convert('L')
recons_image = np.array(recons_image)

mse = np.mean((image - recons_image) ** 2)

file_size = os.path.getsize(image_path)

with open('compressed_rgb_image.bin', 'rb') as file:
    compressed_data = pickle.load(file)

bpp = sum(compressed_data['len_encoded_strings']) / (3*image.shape[0] * image.shape[1])

print(str(mse) + "," + str(bpp))