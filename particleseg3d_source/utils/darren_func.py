import os
import zarr
import tifffile
import numpy as np

def safe_makedirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {path}: {e}")
        cleaned_path = path.strip('"')
        print(f"Retrying with cleaned path: {cleaned_path}")
        try:
            os.makedirs(cleaned_path, exist_ok=True)
            path = cleaned_path
        except OSError as e2:
            print(f"Failed again with cleaned path {cleaned_path}: {e2}")
            raise e2
    return path

def clean_path(path):
    return path.strip('"')

def is_valid_zarr_directory(zarr_dir):
    if not os.path.exists(zarr_dir):
        print(f"Error: Zarr directory {zarr_dir} does not exist.")
        return False
    if not os.path.isdir(zarr_dir):
        print(f"Error: {zarr_dir} is not a directory.")
        return False
    for image_name in os.listdir(zarr_dir):
        image_path = os.path.join(zarr_dir, image_name)
        zarr_file = os.path.join(image_path, f"{image_name}.zarr")
        if os.path.isdir(image_path) and os.path.exists(zarr_file):
            continue
        else:
            print(f"Warning: {image_path} is missing a corresponding .zarr file.")
            return False
    return True

def convert_zarr_to_tiff(zarr_dir, tiff_dir):
    print(f"Original Zarr Directory: {zarr_dir}")
    print(f"Original TIFF Directory: {tiff_dir}")
    if not is_valid_zarr_directory(zarr_dir):
        print("Invalid Zarr directory. Aborting conversion.")
        return
    tiff_dir = safe_makedirs(tiff_dir)
    for image_name in os.listdir(zarr_dir):
        image_path = os.path.join(zarr_dir, image_name)
        if os.path.isdir(image_path):
            zarr_input = os.path.join(image_path, f"{image_name}.zarr")
            image_zarr = zarr.open(zarr_input, mode='r')
            for i in range(image_zarr.shape[0]):
                image_slice = image_zarr[i]
                image_slice = (image_slice / np.max(image_slice) * 255).astype(np.uint8)
                tiff_output_dir = os.path.join(tiff_dir, image_name)
                tiff_output_dir = safe_makedirs(tiff_output_dir)
                tifffile.imwrite(os.path.join(tiff_output_dir, f"{image_name}_{i:04d}.tiff"), image_slice)

def setup_paths(dir_location, output_to_cloud, run_tag, is_original_data, weights_tag):
    if dir_location.lower() == 'internal':
        base_path = r'C:\Senior_Design'
    elif dir_location.lower() == 'external':
        base_path = r'D:\Senior_Design'
    elif dir_location.lower() == 'cloud':
        base_path = r'C:\Users\dchen\OneDrive - University of Connecticut\Courses\Year 4\Fall 2024\BME 4900 and 4910W (Kumavor)\Python\Files'
    elif dir_location.lower() == 'refine':
        base_path = r'D:\Darren\Files'
    else:
        raise ValueError('Invalid directory location type')
    
    base_input_path = os.path.join(base_path, 'database')
    base_output_path = os.path.join(base_path, 'outputs')
    if output_to_cloud:
        base_output_path = os.path.join(r'C:\Users\dchen\OneDrive - University of Connecticut\Courses\Year 4\Fall 2024\BME 4900 and 4910W (Kumavor)\Python\Files', 'outputs')
    base_weights_path = os.path.join(base_path, 'weights')

    output_zarr_path = os.path.join(base_output_path, 'zarr', run_tag)
    output_tiff_path = os.path.join(base_output_path, 'tiff', run_tag)
    
    if is_original_data:
        input_path = os.path.join(base_input_path, 'orignal_dataset', 'grayscale', 'dataset')
    else:
        input_path = os.path.join(base_input_path, 'tablet_dataset', 'grayscale', 'dataset')

    weights_path = os.path.join(base_weights_path, weights_tag)

    print('Paths set')
    return input_path, output_zarr_path, output_tiff_path, weights_path