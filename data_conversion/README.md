# Data Conversion Scripts
Darren, last edit 04/21/2025.

This directory contains a collection of Jupyter notebooks for converting between various biomedical imaging formats commonly used in 3D segmentation workflows. These scripts are intended to simplify the preprocessing and visualization of volumetric data for machine learning applications such as ParticleSeg3D.

# Available Conversion Scripts
## MHD to TIFF
- **`mhd_to_tiff.ipynb`**  
- Converts MetaImage (`.mhd`/`.raw`) volumes into a stack of TIFF images.

## NIfTI to TIFF
- **`nifti_to_tiff.ipynb`**  
- Converts NIfTI (`.nii` or `.nii.gz`) files to a TIFF stack.

## TIFF to NIfTI
- **`tiff_to_nifti.ipynb`**  
- Converts a directory of TIFF slices into a single NIfTI file.

### Parameters
Run by using `convert_tiff_to_nifti(load_dir: str, save_dir: str, spacing: Tuple[float, float, float], img_type='instance')`.

- **`load_dir`** (`str`):  
  Path to the directory containing the input TIFF slices.

- **`save_dir`** (`str`):  
  Path where the resulting NIfTI file will be saved.

- **`spacing`** (`Tuple[float, float, float]`):  
  Physical spacing of the image in the format `(z, y, x)`, typically in millimeters. Required for correct voxel scaling in 3D space.

- **`img_type`** (`str`, optional):  
  Type of image being converted. Can be `'instance'`, `'grayscale'`, `'bordercore'`, etc., depending on your dataset. Default is `'instance'`.

## TIFF to Video
- **`tiff_to_video.ipynb`**  
- Creates a video (e.g., `.mp4`) from a stack of TIFF images for visualization.

### Parameters
Run by using `convert_tiff_to_video(tiff_path: str, img_name: list[str], img_shape: list[tuple], file_format: str = 'mp4', video_length: int = 100, fps: int = None)`.

- **`tiff_path`** (`str`):  
  Base path to the folder containing TIFF image subdirectories.

- **`img_name`** (`list[str]`):  
  List of subfolder names (within `tiff_path`) that contain the TIFF image stacks to convert.

- **`img_shape`** (`list[tuple]`):  
  List of image dimensions for each folder in `img_name`. Each entry should be a tuple `(height, width)`.

- **`file_format`** (`str`, optional):  
  Format of the output video — either `'mp4'` or `'gif'`. Default is `'mp4'`.

- **`video_length`** (`int`, optional):  
  Desired length of the video in seconds. Used only if `fps` is not specified. Default is `100`.

- **`fps`** (`int`, optional):  
  Frames per second for the output video. If provided, overrides `video_length`.

## TIFF to Zarr
- **`tiff_to_zarr.ipynb`**  
- Converts TIFF stacks to the Zarr format for efficient storage and scalable access.