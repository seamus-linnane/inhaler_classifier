import os
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import transforms
import random

# Paths
processed_data_path = "./data/processed/"

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])

# Define the standard file extension
standard_extension = ".jpeg"

# Function to check if an image is already standardised
def is_image_standardised(file_path):
    try:
        with Image.open(file_path) as img:
            return img.size == (224, 256) and file_path.endswith(standard_extension)
    except Exception as e:
        print(f"Error checking file {file_path}: {e}")
        return False

# Function to validate if an image is corrupted or invalid
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Ensure the file is a valid image
        return True
    except UnidentifiedImageError:
        print(f"Corrupted image detected: {file_path}")
        return False
    except Exception as e:
        print(f"Error validating image {file_path}: {e}")
        return False

# Function to standardise images in a folder
def standardise_images(folder_path):
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Validate the image
                if not is_valid_image(file_path):
                    print(f"Deleting corrupted image: {file_path}")
                    os.remove(file_path)  # Delete invalid files
                    continue

                try:
                    # Open the image
                    image = Image.open(file_path).convert("RGB")

                    # Apply transformations
                    standardised_image = transform(image)

                    # Save the standardised image with the standard extension
                    base_name = os.path.splitext(file_name)[0]
                    new_file_name = f"{base_name}{standard_extension}"
                    new_file_path = os.path.join(root, new_file_name)
                    standardised_image.save(new_file_path)

                    # Remove the original file if the extension changed
                    if file_path != new_file_path:
                        os.remove(file_path)

                    print(f"Standardised: {new_file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

# Check if images are already standardised
def check_if_standardised(folder_path, sample_size=5):
    """Randomly check a sample of images in the folder to see if they are standardised."""
    all_files = [
        os.path.join(root, file_name)
        for root, _, files in os.walk(folder_path)
        for file_name in files
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    # If there are no images, skip the check
    if not all_files:
        print("No images found in the processed folder.")
        return False

    # Randomly sample files to check
    sample_files = random.sample(all_files, min(sample_size, len(all_files)))
    for file_path in sample_files:
        if not is_image_standardised(file_path):
            print(f"Found a non-standardised image: {file_path}")
            return False

    print("All sampled images are standardised.")
    return True

train_path = os.path.join(processed_data_path, "train")
if not os.path.exists(train_path):
    print(f"Error: '{train_path}' does not exist. Please run 'prep_data.py' first.")
    exit()
else:
    # Check if the images are already standardised
    if check_if_standardised(processed_data_path):
        print("Images are already standardised. Skipping standardisation.")
    else:
        # Start the process from the processed folder
        standardise_images(processed_data_path)
        print("All images have been standardised.")