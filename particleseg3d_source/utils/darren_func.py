import os
import zarr
import tifffile
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from p_tqdm import p_map  # Parallel tqdm for multiprocessing


def safe_makedirs(path):
    """Safely create directories, handling edge cases."""
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
    """Clean up path formatting."""
    return path.strip('"')

def is_valid_zarr_directory(zarr_dir, image_name=None):
    """Check if a Zarr directory is valid."""
    if not os.path.exists(zarr_dir):
        print(f"Error: Zarr directory {zarr_dir} does not exist.")
        return False
    if not os.path.isdir(zarr_dir):
        print(f"Error: {zarr_dir} is not a directory.")
        return False

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

def remap_labels_visualization(image):
    unique_labels = np.unique(image)  # Get unique labels
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background (assuming it's 0)

    max_value = np.iinfo(image.dtype).max  # Get max value for dtype (e.g., 65535 for uint16)
    new_labels = np.linspace(max_value-len(unique_labels), max_value, len(unique_labels), dtype=image.dtype)  # Create ascending values starting at max_value-len(unique labels)

    # Create a mapping dictionary
    label_map = {old: new for old, new in zip(unique_labels, new_labels)}

    # Apply mapping
    remapped_image = np.copy(image)
    for old_label, new_label in label_map.items():
        remapped_image[image == old_label] = new_label

    return remapped_image

def remap_labels_ascending(image):
    unique_labels = np.unique(image)  # Get unique labels
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background (assuming it's 0)

    max_value = np.iinfo(image.dtype).max  # Get max value for dtype (e.g., 65535 for uint16)
    new_labels = np.linspace(1, max_value, len(unique_labels), dtype=image.dtype)  # Create ascending values starting at max_value-len(unique labels)

    # Create a mapping dictionary
    label_map = {old: new for old, new in zip(unique_labels, new_labels)}

    # Apply mapping
    remapped_image = np.copy(image)
    for old_label, new_label in label_map.items():
        remapped_image[image == old_label] = new_label

    return remapped_image

def remap_labels_binary(image):
    # Convert image to binary (0 for background, 1 for foreground)
    # This assumes the foreground is any non-zero value.

    image = np.asarray(image) # Ensure the image is a numpy array

    # Convert image to binary where non-zero values are set to max_value (e.g., 255 for uint8)
    binary_image = np.where(image != 0, 255, 0).astype(np.uint8)

    return binary_image

def process_image_slice(image_slice, image, i, tiff_output_dir):
    """Process a single image slice and save it as a TIFF."""
    tifffile.imwrite(os.path.join(tiff_output_dir, f"{image}_{i:04d}.tiff"), image_slice)

def process_image(image, zarr_dir, tiff_dir, to_binary):
    """Process a single Zarr image, converting all slices to TIFF."""
    image_path = os.path.join(zarr_dir, image)
    zarr_input = os.path.join(image_path, f"{image}.zarr")

    if not os.path.exists(zarr_input):
        print(f"\nSkipping {image}: Zarr file not found.")
        return

    image_zarr = zarr.open(zarr_input, mode='r')
    if to_binary:
        image_zarr = remap_labels_binary(image_zarr)
    tiff_output_dir = safe_makedirs(os.path.join(tiff_dir, image))

    for i in tqdm(range(image_zarr.shape[0]), desc=f"Converting {image}", leave=False):
        process_image_slice(image_zarr[i], image, i, tiff_output_dir)

def convert_zarr_to_tiff(zarr_dir, tiff_dir, image_name=None, to_binary=False):
    """Convert all Zarr images in a directory to TIFF format using multiprocessing with tqdm."""
    print(f"Zarr Directory: {zarr_dir}")
    print(f"TIFF Directory: {tiff_dir}")

    safe_makedirs(tiff_dir)
    image_list = image_name if image_name is not None else os.listdir(zarr_dir)

    is_valid_zarr_directory(zarr_dir, image_list)

    p_map(lambda img: process_image(img, zarr_dir, tiff_dir, to_binary), image_list, num_cpus=multiprocessing.cpu_count())
    # p_map(lambda img: process_image(img, zarr_dir, tiff_dir), image_list, num_cpus=32)

    print("Conversion complete.")

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
class PathMaster:
    def __init__(self, dir_location, output_to_cloud, run_tag, is_original_data, weights_tag):
        self.dir_location = dir_location.lower()
        self.output_to_cloud = output_to_cloud
        self.run_tag = run_tag
        self.is_original_data = is_original_data
        self.weights_tag = weights_tag
        
        self.base_path = self.get_base_path()
        self.setup_paths()

    def get_base_path(self):
        paths = {
            'internal': r'C:\Senior_Design',
            'external': r'D:\Senior_Design',
            'cloud': r'C:\Users\dchen\OneDrive - University of Connecticut\Courses\Year 4\Fall 2024\BME 4900 and 4910W (Kumavor)\Python\Files',
            'refine': r'D:\Darren\Files'
        }
        if self.dir_location not in paths:
            raise ValueError('Invalid directory location type')
        return paths[self.dir_location]

    def setup_paths(self):
        self.base_database_path = os.path.join(self.base_path, 'database')
        self.base_output_path = os.path.join(self.base_path, 'outputs')
        self.base_weights_path = os.path.join(self.base_path, 'weights')
        
        if self.output_to_cloud:
            if self.dir_location == 'refine':
                self.base_output_path = os.path.join(r'D:\Darren\OneDrive - University of Connecticut\Courses\Year 4\Fall 2024\BME 4900 and 4910W (Kumavor)\Python\Files', 'outputs')
            else:
                self.base_output_path = os.path.join(r'C:\Users\dchen\OneDrive - University of Connecticut\Courses\Year 4\Fall 2024\BME 4900 and 4910W (Kumavor)\Python\Files', 'outputs')
        
        self.pred_zarr_path = os.path.join(self.base_output_path, 'zarr', self.run_tag)
        self.pred_tiff_path = os.path.join(self.base_output_path, 'tiff', self.run_tag)
        
        dataset_type = 'orignal_dataset' if self.is_original_data else 'tablet_dataset'
        self.grayscale_path = os.path.join(self.base_database_path, dataset_type, 'grayscale', 'dataset')
        self.weights_path = os.path.join(self.base_weights_path, self.weights_tag)
        
        # PSD Paths
        self.psd_path = os.path.join(self.base_output_path, 'metrics', 'particle_size_dist_v7')
        
        # Ground Truth Paths
        self.gt_sem_path = os.path.join(self.base_database_path, dataset_type, 'binary', 'tiff')
        self.gt_inst_path = os.path.join(self.base_database_path, dataset_type, 'instance', 'tiff')
        
        # Metrics Paths
        self.sem_metrics_path = os.path.join(self.base_output_path, 'metrics', 'semantic_metrics')
        
        # Check directories
        input_paths = [self.grayscale_path, self.weights_path, self.gt_sem_path, self.gt_inst_path]
        output_paths = [self.pred_zarr_path, self.pred_tiff_path, self.psd_path, self.sem_metrics_path]
        bad_input_paths = []
        bad_output_paths = []
        
        for input_path in input_paths:
            if not os.path.isdir(input_path):
                bad_input_paths.append(input_path)
        for output_path in output_paths:
            if not os.path.isdir(output_path):
                bad_output_paths.append(output_path)
                os.makedirs(output_path)
        
        if len(bad_input_paths) > 0:
            raise ValueError('The following input paths were not defined:', bad_input_paths)
        else:
            print('All input paths were properly created')
        
        if len(bad_output_paths) > 0:
            print('The following output paths were just created:', bad_output_paths)
        else:
            print('All input paths were properly defined')