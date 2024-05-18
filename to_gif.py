import os
from PIL import Image

def create_gif_from_images(image_dir, output_path, label_number, fps=10):
    images = []
    # Loop through potential image files based on the naming convention
    for epoch in range(1, 51):
        image_path = os.path.join(image_dir, f'./integrated_grads_epoch_{epoch}_img_{label_number}.png')
        if os.path.exists(image_path):
            img = Image.open(image_path)
            images.append(img)
    
    # Save images as a GIF if any valid images are found
    if images:
        images[0].save(
            output_path, save_all=True, append_images=images[1:], 
            duration=int(1000 / fps), loop=0
        )
    else:
        print(f"No images found for label {label_number}.")

if __name__ == "__main__":
    image_directory = "./integrated/"
    # Define label mappings
    label_mappings = {4: '0', 1: '7', 2: '2', 3: '1', 5: '4'}
    
    # Loop through the label mappings and create a GIF for each
    for label_num, label_text in label_mappings.items():
        output_gif_path = f"./gifs/label_{label_text}_validation_integrated_heatmaps.gif"
        create_gif_from_images(image_directory, output_gif_path, label_num, fps=10)
