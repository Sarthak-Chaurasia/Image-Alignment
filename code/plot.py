import matplotlib.pyplot as plt
import os

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Loop over all 20 grayscale images
for image_num in range(1, 21):
    x = []
    y = []
    
    # Read the result file for each grayscale image
    results_path = f'./results/results_batch_{image_num}.txt'
    if os.path.exists(results_path):
        with open(results_path, 'r') as file:
            for line in file:
                data = line.split(',')
                y.append(float(data[0]))  # RMSE
                x.append(float(data[1]))  # BPP

        # Plotting
        plt.figure()
        plt.plot(x, y, marker='o', label=f'Grayscale Image {image_num}')
        plt.xlabel('BPP')
        plt.ylabel('RMSE')
        plt.title(f'Compression Performance (Grayscale Image {image_num})')
        plt.legend()
        plt.grid(True)

        # Save the grayscale plot
        plt.savefig(f'plots/greyscale_image_{image_num}_plot.png')
        plt.close()

# Loop over all 20 color images
for image_num in range(1, 21):
    x = []
    y = []
    
    # Read the result file for each color image
    results_path = f'./results/results_color_batch_{image_num}.txt'
    if os.path.exists(results_path):
        with open(results_path, 'r') as file:
            for line in file:
                data = line.split(',')
                y.append(float(data[0]))  # RMSE
                x.append(float(data[1]))  # BPP

        # Plotting
        plt.figure()
        plt.plot(x, y, marker='o', label=f'Color Image {image_num}')
        plt.xlabel('BPP')
        plt.ylabel('RMSE')
        plt.title(f'Compression Performance (Color Image {image_num})')
        plt.legend()
        plt.grid(True)

        # Save the color plot
        plt.savefig(f'plots/color_image_{image_num}_plot.png')
        plt.close()
