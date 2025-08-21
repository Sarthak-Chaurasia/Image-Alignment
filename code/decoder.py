import numpy as np
from scipy.fftpack import dct
import huff
import pickle
from PIL import Image

output_file = './compressed_rgb_image.bin'

def decode_huffman(encoded_string, huffman_table):
    reverse_huffman_table = {v: k for k, v in huffman_table.items()}
    current_code = ""
    decoded_output = []

    for bit in encoded_string:
        current_code += bit
        if current_code in reverse_huffman_table:
            decoded_output.append(reverse_huffman_table[current_code])
            current_code = ""

    return decoded_output

def inverse_zigzag_order(result, h, w):
    matrix = np.zeros((h, w), dtype=int)
    index = 0
    for i in range(h + w - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                if j < h and i - j < w:
                    matrix[j, i - j] = result[index]
                    index += 1
        else:
            for j in range(i + 1):
                if j < w and i - j < h:
                    matrix[i - j, j] = result[index]
                    index += 1
    return matrix

def dequantize(quantized_coeffs, Q, block_size=8):
    h, w = quantized_coeffs.shape
    dequantized_coeffs = np.zeros_like(quantized_coeffs, dtype=float)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = quantized_coeffs[i:i+block_size, j:j+block_size]
            dequantized_coeffs[i:i+block_size, j:j+block_size] = block * (50 * M / Q)
    
    return dequantized_coeffs

def inverse_differential_encoding(diff_encoded_coeffs, block_size=8):
    h, w = diff_encoded_coeffs.shape
    quantized_coeffs = np.zeros_like(diff_encoded_coeffs)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = diff_encoded_coeffs[i:i+block_size, j:j+block_size]
            if i == 0 and j == 0:
                quantized_coeffs[i:i+block_size, j:j+block_size] = block
            else:
                if j == 0:
                    prev_block = quantized_coeffs[i-block_size:i, j:j+block_size]
                else:
                    prev_block = quantized_coeffs[i:i+block_size, j-block_size:j]
                quantized_coeffs[i:i+block_size, j:j+block_size] = block + prev_block

    return quantized_coeffs

def idct2(block):
    return dct(dct(block.T, type=3, norm='ortho').T, type=3, norm='ortho')

def compute_2d_idct(dct_coefficients, block_size=8):
    h, w = dct_coefficients.shape
    image = np.zeros_like(dct_coefficients, dtype=float)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_coefficients[i:i+block_size, j:j+block_size]
            image[i:i+block_size, j:j+block_size] = idct2(block)
    
    return image

with open(output_file, 'rb') as f:
    data = pickle.load(f)

M = data['quantization_table']
encoded_strings = [ bin(data['encoded_strings'][k])[2:].zfill(data['len_encoded_strings'][k]) for k in range(3) ]
huffman_tables = data['huffman_tables']
decoded_strings = [ decode_huffman(encoded_strings[k], huffman_tables[k]) for k in range(3) ]
h = data['file_size'][0]
w = data['file_size'][1]
quantized_coeffs = [ inverse_zigzag_order(decoded_strings[k], h, w) for k in range(3) ]
dequantized_coeffs = [ dequantize(quantized_coeffs[k], data['Q']) for k in range(3) ]
diff_encoded_coeffs = [ inverse_differential_encoding(dequantized_coeffs[k]) for k in range(3) ]
reconstructed_image = np.array([ compute_2d_idct(diff_encoded_coeffs[k]) for k in range(3) ])

reconstructed_image += 127
reconstructed_image = np.round(np.clip(reconstructed_image, 0, 255))
np.set_printoptions(suppress=True)
reconstructed_image_path = './reconstructed_image.jpg'
reconstructed_image = reconstructed_image.transpose(1, 2, 0).astype(np.uint8)
reconstructed_image = Image.fromarray(reconstructed_image)
reconstructed_image.save(reconstructed_image_path)