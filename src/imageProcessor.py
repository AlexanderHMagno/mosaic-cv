import cv2
import numpy as np
import random
import os
from PIL import Image
from imageComparison import ImageComparison

class ImageProcessor:
    def __init__(self):
        pass

    def preprocess_image(self, image, grid_size):
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

        h, w = image.shape[:2]

        
        # Round down to nearest multiple of grid_size
        h = h - (h % grid_size)
        w = w - (w % grid_size)
        print(f"h: {h}, w: {w}")
        # Resize to a fixed size keep aspect ratio  
        resized_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        
        return resized_image

    def divide_and_analyze_grid(self, image, grid_size):
        """
        Divide the image into a grid and classify each grid cell by average intensity or dominant color.
        Args:
            image: Preprocessed image (numpy array).
            grid_size: Size of the Tile cells (e.g., 8x8 or 16x16).
            keep_image_size: If True, keep the original image size.
        Returns:
            Annotated image with grid division and analysis results.
        """

        height, width, _ = image.shape
        new_height = height // grid_size
        new_width = width // grid_size
        cell_height = grid_size
        cell_width = grid_size

        print(cell_height, cell_width, grid_size, height, width)
        # Create a copy of the image for annotation
        annotated_image = image.copy()
        
        # Analyze each grid cell
        for row in range(new_height):
            for col in range(new_width):
                # Extract the grid cell
                col_start = col * cell_width
                col_end = col_start + cell_width
                row_start = row * cell_height
                row_end = row_start + cell_height
                grid_cell = image[row_start:row_end, col_start:col_end]

                # Calculate the average color
                avg_color = np.mean(grid_cell, axis=(0, 1)).astype(int)

                # i need integer values
                avg_color = (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
                
                cv2.rectangle(
                    annotated_image, (col_start, row_start), (col_end, row_end), avg_color, -1
                )

       
        return Image.fromarray(annotated_image)

    def blend_tile_with_pixel(self, tile, color):
        """Blend a 32x32 tile with a given color."""
        color_layer = np.full_like(tile, color, dtype=np.uint8)  # Create a solid color layer
        blended_tile = cv2.addWeighted(tile, 0.3, color_layer, 0.7, 0)  
        return blended_tile 

    def add_tile_mapping(self, image, grid_size, tile_type):
        """
        Add tile mapping to the image.
        """
        robot_images = random.sample(os.listdir('public/' + tile_type), 10)
        image = np.array(image)
        image_height, image_width, _ = image.shape
        h, w = image_height // grid_size, image_width // grid_size

        # This comes from the array of robot images
        tile_h , tile_w = grid_size, grid_size
        
        mosaic = image.copy()
        
        print(f"h Final: {h}, w Final: {w}")
        # Iterate over each pixel in the base image
        for i in range(h):
            for j in range(w):
                pixel_color = image[i * grid_size, j * grid_size]  # Get the pixel color
                
                random_robot_image = random.choice(robot_images)
                robotina = cv2.imread(os.path.join('public/' + tile_type, random_robot_image)) 
                robotina = cv2.resize(robotina, (grid_size, grid_size))
                # Blend the tile with the pixel color
                blended_tile = self.blend_tile_with_pixel(robotina, pixel_color)

                # Place it in the correct position
                mosaic[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w] = blended_tile

        return mosaic

    def resize_image(self, image, grid_result):
        image = np.array(image)
        return cv2.resize(grid_result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

    def process_image(self, image, grid_size, tile_type):
        """
        Full pipeline: Preprocess the image, divide it into grids, and analyze.
        """
        # Preprocess the image
        image = self.preprocess_image(image, grid_size)
        # Divide into grids and analyze
        grid_result = self.divide_and_analyze_grid(image, grid_size)

        # Add tile mapping
        grid_result = self.add_tile_mapping(grid_result, grid_size, tile_type.lower())

        # # Compare the original image with the mosaic
        image_comparison = ImageComparison(image, grid_result)
        metrics = image_comparison.generate_metrics_explanation()
        
        
        return grid_result , metrics