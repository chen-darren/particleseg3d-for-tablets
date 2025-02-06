import os
import zarr
import tifffile
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

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

def is_valid_zarr_directory(zarr_dir, image_name=None):
    if not os.path.exists(zarr_dir):
        print(f"Error: Zarr directory {zarr_dir} does not exist.")
        return False
    if not os.path.isdir(zarr_dir):
        print(f"Error: {zarr_dir} is not a directory.")
        return False
    
    # If image_name is None, process all directories in zarr_dir
    image_list = image_name if image_name is not None else os.listdir(zarr_dir)
    for image in image_list:
        image_path = os.path.join(zarr_dir, image)
        zarr_file = os.path.join(image_path, f"{image}.zarr")
        if os.path.isdir(image_path) and os.path.exists(zarr_file):
            continue
        else:
            print(f"Warning: {image_path} is missing a corresponding .zarr file.")
            return False
    return True

def convert_zarr_to_tiff(zarr_dir, tiff_dir, image_name=None):
    print(f"Original Zarr Directory: {zarr_dir}")
    print(f"Original TIFF Directory: {tiff_dir}")
    
    if not is_valid_zarr_directory(zarr_dir, image_name):
        print("Invalid Zarr directory. Aborting conversion.")
        return
    
    tiff_dir = safe_makedirs(tiff_dir)
    
    # If image_name is None, process all directories in zarr_dir
    image_list = image_name if image_name is not None else os.listdir(zarr_dir)

    for image in tqdm(image_list, desc="Processing images"):
        image_path = os.path.join(zarr_dir, image)
        
        if os.path.isdir(image_path):
            zarr_input = os.path.join(image_path, f"{image}.zarr")
            
            if not os.path.exists(zarr_input):
                print(f"Skipping {image}: Zarr file not found.")
                continue
            
            image_zarr = zarr.open(zarr_input, mode='r')
            
            for i in tqdm(range(image_zarr.shape[0]), desc=f"Converting {image}", leave=False):
                image_slice = image_zarr[i]

                # Ensure valid values in image_slice
                if np.isnan(image_slice).any() or np.isinf(image_slice).any():
                    # print(f"Warning: NaN or Inf detected in {image} slice {i}, replacing with 0.")
                    image_slice = np.nan_to_num(image_slice, nan=0.0, posinf=255, neginf=0)

                max_val = np.max(image_slice)
                if max_val > 0 and not np.isnan(max_val):
                    image_slice = (image_slice / max_val * 255).astype(np.uint8)
                else:
                    # print(f"Warning: Max value is zero or NaN for {image} slice {i}, setting to black.")
                    image_slice = np.zeros_like(image_slice, dtype=np.uint8)
                
                tiff_output_dir = os.path.join(tiff_dir, image)
                tiff_output_dir = safe_makedirs(tiff_output_dir)
                
                tifffile.imwrite(os.path.join(tiff_output_dir, f"{image}_{i:04d}.tiff"), image_slice)
    
    print("Conversion complete.")

# # For parallel processing =================================================================================
# def process_image_slice(image_slice, image, i, tiff_output_dir):
#     """Process a single image slice and save as TIFF."""
#     if np.isnan(image_slice).any() or np.isinf(image_slice).any():
#         image_slice = np.nan_to_num(image_slice, nan=0.0, posinf=255, neginf=0)

#     max_val = np.max(image_slice)
#     if max_val > 0 and not np.isnan(max_val):
#         image_slice = (image_slice / max_val * 255).astype(np.uint8)
#     else:
#         image_slice = np.zeros_like(image_slice, dtype=np.uint8)

#     output_path = os.path.join(tiff_output_dir, f"{image}_{i:04d}.tiff")
#     tifffile.imwrite(output_path, image_slice)

# def process_image(args):
#     """Process a single Zarr image, converting all slices to TIFF."""
#     image, zarr_dir, tiff_dir = args
#     image_path = os.path.join(zarr_dir, image)
#     zarr_input = os.path.join(image_path, f"{image}.zarr")

#     if not os.path.exists(zarr_input):
#         print(f"Skipping {image}: Zarr file not found.")
#         return

#     image_zarr = zarr.open(zarr_input, mode='r')
#     tiff_output_dir = safe_makedirs(os.path.join(tiff_dir, image))

#     num_workers = min(multiprocessing.cpu_count(), image_zarr.shape[0])
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         futures = [
#             executor.submit(process_image_slice, image_zarr[i], image, i, tiff_output_dir)
#             for i in range(image_zarr.shape[0])
#         ]
#         for _ in tqdm(futures, desc=f"Converting {image}", leave=False):
#             _.result()  # Ensures errors are caught

# def convert_zarr_to_tiff(zarr_dir, tiff_dir, image_name=None):
#     """Convert all Zarr images in a directory to TIFF format using parallel processing."""
#     print(f"Original Zarr Directory: {zarr_dir}")
#     print(f"Original TIFF Directory: {tiff_dir}")

#     if not is_valid_zarr_directory(zarr_dir, image_name):
#         print("Invalid Zarr directory. Aborting conversion.")
#         return

#     tiff_dir = safe_makedirs(tiff_dir)
#     image_list = image_name if image_name is not None else os.listdir(zarr_dir)

#     args_list = [(img, zarr_dir, tiff_dir) for img in image_list]

#     with ProcessPoolExecutor() as executor:
#         list(tqdm(executor.map(process_image, args_list), total=len(image_list), desc="Processing images"))

#     print("Conversion complete.")

# # =============================================================================================================

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