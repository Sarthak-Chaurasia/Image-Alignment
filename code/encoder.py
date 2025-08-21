import numpy as np
from scipy.fftpack import dct
import huff
import pickle
from PIL import Image
import sys

np.random.seed(0)

if len(sys.argv) < 3:
    print("Usage: python encoder.py <image_path> <quality-factor>")
    sys.exit(1)
np.set_printoptions(suppress=True)
image_path = sys.argv[1]

Q = int(sys.argv[2])

M = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def compute_2d_dct(image, block_size=8):
    h, w = image.shape
    dct_coefficients = np.zeros_like(image, dtype=float)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_coefficients[i:i+block_size, j:j+block_size] = dct2(block)
    
    return dct_coefficients

def quantize(dct_coeffs, Q, block_size=8):
    h, w = dct_coeffs.shape
    quantized_coeffs = np.zeros_like(dct_coeffs)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_coeffs[i:i+block_size, j:j+block_size]
            mat = np.round(block / (50 / Q * M))
            mat = mat.astype(int)
            quantized_coeffs[i:i+block_size, j:j+block_size] = mat
    
    return quantized_coeffs

def differential_encoding(quantized_coeffs, block_size=8):
    h, w = quantized_coeffs.shape
    diff_encoded_coeffs = np.zeros_like(quantized_coeffs)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = quantized_coeffs[i:i+block_size, j:j+block_size]
            if i == 0 and j == 0:
                diff_encoded_coeffs[i:i+block_size, j:j+block_size] = block
            else:
                if j == 0:
                    prev_block = quantized_coeffs[i-block_size:i, j:j+block_size]
                else:
                    prev_block = quantized_coeffs[i:i+block_size, j-block_size:j]
                diff_encoded_coeffs[i:i+block_size, j:j+block_size] = block - prev_block
    
    return diff_encoded_coeffs

def zigzag_order(matrix):
    h, w = matrix.shape
    result = []
    for i in range(h + w - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                if j < h and i - j < w:
                    # print(matrix.shape)
                    # print(i, j, i - j)
                    result.append(matrix[j, i - j])
        else:
            for j in range(i + 1):
                if j < w and i - j < h:
                    # print(matrix.shape)
                    # print(i, j, i - j)
                    result.append(matrix[i - j, j])
    return result

def huffman_encode(string):
    freq = {}
    for c in string:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1

    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    nodes = freq

    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = huff.NodeTree(key1, key2)
        nodes.append((node, c1 + c2))

        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    
    return huff.huffman_code_tree(nodes[0][0])

images = Image.open(image_path).convert('RGB')
images = np.array(images)
images = np.moveaxis(images, -1, 0)  # Move the color channel to the first dimension
images =  -127 + images
huffmanCodes = []
encoded_strings = []

for i in range(3):
    image = images[i]
    image = np.array(image)    
    dct_coeffs = compute_2d_dct(image)
    diff_encoded_coeffs = differential_encoding(dct_coeffs)
    quantized_coeffs = quantize(diff_encoded_coeffs, Q)
    zigzag_coeffs = zigzag_order(quantized_coeffs)
    huffmanCode = huffman_encode(zigzag_coeffs)
    encoded_string = ''.join([huffmanCode[char] for char in zigzag_coeffs])

    huffmanCodes.append(huffmanCode)
    encoded_strings.append(encoded_string)

data_to_save = {
    'file_size': images[0].shape,
    'block_size': 8,
    'Q': Q,
    'quantization_table': M,
    'huffman_tables': huffmanCodes,
    'len_encoded_strings': [len(encoded_strings[k]) for k in range(3)],
    'encoded_strings': [int(encoded_strings[k], 2) for k in range(3)]
}

output_file = './compressed_rgb_image.bin'
with open(output_file, 'wb') as f:
    pickle.dump(data_to_save, f)