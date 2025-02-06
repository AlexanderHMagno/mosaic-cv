import cv2
import numpy as np
import random
import os
from PIL import Image



def preprocess_image(image, grid_size):
    """
    Preprocess the uploaded image:
    1. Convert to RGB format if needed.
    2. Resize to a fixed resolution for consistent processing.
    """
    image = np.array(image)
    
    # Ensure the image is in RGB format
    if image.shape[-1] == 4:  # Handle images with alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:  # Handle grayscale images
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    h, w = image.shape[:2];

    # Round down to nearest multiple of grid_size
    h = h - (h % grid_size)
    
    # Resize to a fixed size keep aspect ratio  
    resized_image = cv2.resize(image, (h, h), interpolation=cv2.INTER_AREA)
    
    return resized_image

def divide_and_analyze_grid(image, grid_size):
    """
    Divide the image into a grid and classify each grid cell by average intensity or dominant color.
    Args:
        image: Preprocessed image (numpy array).
        grid_size: Size of the grid cells (e.g., 8x8 or 16x16).
    Returns:
        Annotated image with grid division and analysis results.
    """
    height, width, _ = image.shape
    cell_height = height // grid_size
    cell_width = cell_height

    print(f"from divide_and_analyze_grid cell_height: {cell_height}, cell_width: {cell_width}, image shape: {image.shape}")
    # Create a copy of the image for annotation
    annotated_image = image.copy()
    
    # Analyze each grid cell
    for row in range(grid_size):
        for col in range(grid_size):
            # Extract the grid cell
            col_start = col * cell_width
            col_end = col_start + cell_width
            row_start = row * cell_height
            row_end = row_start + cell_height
            grid_cell = image[row_start:row_end, col_start:col_end]

            # Calculate the average color
            avg_color = np.mean(grid_cell, axis=(0, 1)).astype(int)

            # Draw the grid and annotate the average color
            # cv2.rectangle(
            #     annotated_image, (x_start, y_start), (x_end, y_end), (0, 0, 0), 1
            # )

            # i need integer values
            avg_color = (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))

            # How to replace this area of the image with a robot image? 
            # How to make the robot merged with the average color?
            
            cv2.rectangle(
                annotated_image, (col_start, row_start), (col_end, row_end), avg_color, -1
            )

    # I want to cv2.resize the image to the original size
    annotated_image = cv2.resize(annotated_image, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    # return robot_images
    return Image.fromarray(annotated_image)


def blend_tile_with_pixel(tile, color):
    """Blend a 32x32 tile with a given color."""
    color_layer = np.full_like(tile, color, dtype=np.uint8)  # Create a solid color layer
    blended_tile = cv2.addWeighted(tile, 0.3, color_layer, 0.7, 0)  
    return blended_tile

def add_tile_mapping(image, grid_size, tile_type):

    print(f"tile_type: {tile_type}")

    
    robot_images = random.sample(os.listdir('public/' + tile_type), 10)
    random_robot_image = random.choice(robot_images)
    robotina = cv2.imread(os.path.join('public/' + tile_type, random_robot_image))  # Loads in BGR format

    # Convert BGR to RGB
    robotina = cv2.cvtColor(robotina, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    h, w = grid_size, grid_size
    # This comes from the array of robot images
    tile_h , tile_w = robotina.shape[:2]
    
    # print(f"h: {h}, w: {w}")
    mosaic = np.zeros((tile_h * grid_size, tile_w * grid_size, 3), dtype=np.uint8)

    print(f"mosaic shape: {mosaic.shape}")
    print(f"robotina shape: {robotina.shape}")
    print(f"image shape: {image.shape}")
    
    # Iterate over each pixel in the base image
    for i in range(h):
        for j in range(w):
            pixel_color = image[i, j]  # Get the pixel color
            
            random_robot_image = random.choice(robot_images)
            robotina = cv2.imread(os.path.join('public/' + tile_type, random_robot_image)) 
            # Blend the tile with the pixel color
            blended_tile = blend_tile_with_pixel(robotina, pixel_color)

            # Place it in the correct position
            mosaic[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w] = blended_tile

    return mosaic


    # # Analyze each grid cell
    # for row in range(grid_size):
    #     for col in range(grid_size):
    #         # select 1 random robot_images  
    #         random_robot_image = random.choice(robot_images)
    #         # Read image using OpenCV
    #         robotina = cv2.imread(os.path.join('public/robots', random_robot_image))  # Loads in BGR format

    #         # Convert BGR to RGB
    #         robotina = cv2.cvtColor(robotina, cv2.COLOR_BGR2RGB)

    #         # Resize image
    #         robotina = cv2.resize(robotina, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            
    #         # Create a rectangle with the average color
    #         color_rect = np.full((cell_height, cell_width, 3), avg_color, dtype=np.uint8)
            
    #         # Blend the colored rectangle with the robot image
    #         cv2.addWeighted(color_rect, 0.6, robotina, 0.4, 0, robotina)
    
    # return image

def resize_image(image, grid_result):
    image = np.array(image)
    return cv2.resize(grid_result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)


def process_image(image, grid_size, tile_type, keep_image_size):
    """
    Full pipeline: Preprocess the image, divide it into grids, and analyze.
    """

    # # Preprocess the image
    processed_image = preprocess_image(image, grid_size)
    # # # Divide into grids and analyze
    grid_result = divide_and_analyze_grid(processed_image, grid_size)
    # # Dont blend multiply the size of the image to coincide with the grid size
    grid_result = add_tile_mapping(grid_result, grid_size, tile_type.lower())

    # fetch fake user images
    # fetch_fake_user_images(10)
    #resize the grid_result to the original size
    if keep_image_size == "Original Size":
        grid_result = resize_image(image, grid_result)

    # Add a button to download the image
    # download_button = gr.Button("Download Image")
    # download_button.click(lambda: grid_result.save("output.png"))

    return grid_result