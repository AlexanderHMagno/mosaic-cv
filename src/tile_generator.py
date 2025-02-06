from faker import Faker
import requests
from PIL import Image
from io import BytesIO
import os
import uuid
import numpy as np
fake = Faker()


def fetch_fake_user_images(num_images=10):
    """
    Fetch and save random cat avatar images using the RoboHash API.
    
    This function generates random cat avatars, saves them to the public/cats directory,
    and returns them as a list of numpy arrays.
    
    Args:
        num_images (int, optional): Number of avatar images to generate. Defaults to 10.
        
    Returns:
        list: A list of numpy arrays containing the avatar images.
        
    Note:
        - Images are fetched from robohash.org using random UUIDs
        - Images are saved as 32x32 PNG files in the public/cats directory
        - Each image is assigned a unique UUID as filename
    """
    avatars = []
    for _ in range(num_images):
        # Generate unique URL for each avatar
        avatar_url = f"https://robohash.org/{fake.uuid4()}?size=32x32&set=set4"
        
        # Fetch the image from the API
        response = requests.get(avatar_url)
        img = Image.open(BytesIO(response.content))
        
        # Convert to numpy array and store in list
        avatars.append(np.array(img))
        
        # Save image to local filesystem
        img_path = os.path.join('public', 'cats', f"{uuid.uuid4()}.png")
        img.save(img_path, "PNG")

    return avatars

