#!/bin/bash

# Create the 'plots', 'results', and 'reconstructed_images' directories if they don't exist
mkdir -p plots
mkdir -p results
mkdir -p reconstructed_images
mkdir -p reconstructed_color_images
rm -rf reconstructed_images/*  # Clear the 'reconstructed_images' directory
rm -rf results/*  # Clear the 'results' directory
rm -rf plots/*  # Clear the 'plots' directory 
rm -rf reconstructed_color_images/*  # Clear the 'reconstructed_color_images' directory

# Loop over all 20 images
for i in {1..20}
do
    # Create a subfolder for the current image in 'reconstructed_images'
    mkdir -p reconstructed_images/obj${i}__0

    # Process each image for 10 different quantization levels
    for j in {1..10}
    do
        python3 encoder.py ./coil-20/obj${i}__0.png $((10 * j))  # Encoding
        python3 decoder.py  # Decoding

        # Save the reconstructed image for each quality factor in the respective subfolder
        cp reconstructed_image.jpg reconstructed_images/obj${i}__0/reconstructed_image_qf${j}.jpg
        
        # Collect results in the 'results' folder
        python3 results.py ./coil-20/obj${i}__0.png >> results/results_batch_${i}.txt  # Collect results in the 'results' folder
    done
done

# Loop over all 20 color images
for i in {1..20}
do
    # Create a subfolder for the current image in 'reconstructed_color_images'
    mkdir -p reconstructed_color_images/obj${i}__0

    # Process each image for 10 different quantization levels
    for j in {1..10}
    do
        python3 encoder.py ./color_images/obj${i}__0.png $((10 * j))  # Encoding
        python3 decoder.py  # Decoding

        # Save the reconstructed color image for each quality factor in the respective subfolder
        cp reconstructed_image.jpg reconstructed_color_images/obj${i}__0/reconstructed_image_qf${j}.jpg
        
        # Collect results in the 'results' folder
        python3 results.py ./color_images/obj${i}__0.png >> results/results_color_batch_${i}.txt  # Collect results in the 'results' folder
    done
done

# Run the plotting script to generate individual plots for all images
python3 plot.py
