import gradio as gr
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from faker import Faker
import os
import uuid
import random

fake = Faker()

def fetch_fake_user_images(num_images=10):
    """Fetch fake profile images from various sources."""
    avatars = []
    for _ in range(num_images):
        avatar_url = f"https://robohash.org/{fake.uuid4()}?size=32x32"
        response = requests.get(avatar_url)
        img = Image.open(BytesIO(response.content)) # Resize for mosaic
        avatars.append(np.array(img))         # Save the image
        img_path = os.path.join('robots', f"{uuid.uuid4()}.png")
        img.save(img_path, "PNG")

    return avatars


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

    _, w = image.shape[:2];

    # Round down to nearest multiple of grid_size
    w = w - (w % grid_size)

    # Resize to a fixed size keep aspect ratio  
    resized_image = cv2.resize(image, (w, w), interpolation=cv2.INTER_AREA)
    
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

    print(f"height: {height}, width: {width}")

    cell_height = height // grid_size
    cell_width = cell_height

    
    print(f"cell_height: {cell_height}, cell_width: {cell_width} grid_size: {grid_size}")
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

    
    # return robot_images
    return Image.fromarray(annotated_image)

def add_tile_mapping(image):

    robot_images = random.sample(os.listdir('robots'), 10)
    # Analyze each grid cell
    for row in range(grid_size):
        for col in range(grid_size):
            # select 1 random robot_images  
            random_robot_image = random.choice(robot_images)
            # Read image using OpenCV
            robotina = cv2.imread(os.path.join('robots', random_robot_image))  # Loads in BGR format

            # Convert BGR to RGB
            robotina = cv2.cvtColor(robotina, cv2.COLOR_BGR2RGB)

            # Resize image
            robotina = cv2.resize(robotina, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            
            # Create a rectangle with the average color
            color_rect = np.full((cell_height, cell_width, 3), avg_color, dtype=np.uint8)
            
            # Blend the colored rectangle with the robot image
            cv2.addWeighted(color_rect, 0.6, robotina, 0.4, 0, robotina)
    
    return image

def image_mosaic(image):

    image = np.array(image)

    

def process_image(image, grid_size):
    """
    Full pipeline: Preprocess the image, divide it into grids, and analyze.
    """
    # # Preprocess the image
    processed_image = preprocess_image(image, grid_size)

    # # Divide into grids and analyze
    grid_result = divide_and_analyze_grid(processed_image, grid_size)

    # image_mosaic(grid_result)
    # image_mosaic(grid_result)

    # Dont blend multiply the size of the image to coincide with the grid size


    # fetch fake user images
    # fake_user_images = fetch_fake_user_images(10)

    return grid_result

# Create Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Slider(2, 256, step=2, value=8, label="Grid Size (e.g., 8x8)")
    ],
    outputs=gr.Image(type="pil", label="Processed Image with Grid Analysis"),
    title="Image Grid Analysis",
    description="Upload an image and divide it into grids to analyze color or intensity data in each cell."
)

# Run the app
if __name__ == "__main__":
    demo.launch()
