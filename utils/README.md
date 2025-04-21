# Utility Functions  
Darren, last edit 04/21/2025.

This directory contains a collection of Jupyter notebooks used throughout the ParticleSeg3D pipeline for data preparation, visualization, and dataset integrity checks. Each notebook is self-contained and provides specific utility functions that support training, evaluation, or inspection of 3D segmentation datasets.

# `colormapping_bordercore_instance.ipynb`
Visualizes instance masks with distinct colors for different components in border-core representations and instance segmentations. Useful for quality checking and visual debugging of labeled masks.

# `data_augmentation.ipynb`
Applies a configurable pipeline of 3D data augmentation techniques to a volume. All augmentations are optional and controlled via a configuration dictionary. The following augmentations are supported:

- **Shear Crop**: Randomly crops the volume along a specified dimension with a shear effect.
  - Parameters: `crop_factor`, `crop_dim`

- **Scale**: Scales the volume along a specified axis.
  - Parameters: `scale_factor`, `axis`, `order`

- **Zoom (Fixed)**: Applies a fixed zoom in/out to the entire volume.
  - Parameters: `zoom_factor`, `order`

- **Mirror**: Flips the volume along the specified axis.
  - Parameter: `mode` (axis)

- **Rotate**: Applies 90° rotations in 3D space.
  - Parameters: `num_rot`, `rot_axes`

- **Random Erasing**: Erases a random cuboid region from the volume.
  - Parameter: `seed` (for reproducibility), `num_cubes`, `min_size`, `max_size`

# `dataset_exploration.ipynb`
Provides an overview of the metadata of MHD images.

# `generate_split.ipynb`
Creates a five-fold cross-validation split formatted to work with the nnU-Net library. Ensures reproducibility of experiments.

# `global_mean_std.ipynb`
Computes the global mean and standard deviation of voxel intensities across entire volumetric images, which is important for data normalization for inference and training.

# `instance2border_core.ipynb`
Converts instance segmentation masks into semantic masks with explicit border and core labels.

# `read_pickle.ipynb`
Reads and inspects `.pkl` files commonly used by nnU-Net to store metadata and preprocessing results.

# `semantic_to_instance.ipynb`
Generates instance segmentation from semantic segmentation input—useful when ground truth instance segmentation images are unavailable.

## **Conversion Process:**

1. **Semantic to Binary Mask**  
   Convert the semantic segmentation into a binary mask (particle vs. background).  
   - The target voxel value corresponding to the particle class must be provided to correctly extract the binary mask.

2. **Binary Mask to Border-Core Representation**  
   Generate a border-core map by:
   - Eroding each particle to create the **core**.
   - Dilating the core to define the **border** region.  
   - The border thickness is adjustable via the `border_thickness` parameter in:
     ```python
     process_tiff_stack_dir(input_dir: str, border_thickness=1)
     ```
     This value is passed to:
     ```python
     generate_bordercore(image, border_thickness=1)
     ```

3. **Border-Core to Instance Segmentation**  
   Reconstruct individual particles by dilating each core to fill its corresponding border.  
   - Ensure consistency in core and border label values using:
     ```python
     border_core_component2instance_dilation(patch: np.ndarray, core_label: int = 255, border_label: int = 127)
     ```
   - This function mirrors the version used in `\particleseg3d_source\inference`, with one key difference:  
     **`remove_small_cores` is disabled** in this notebook implementation to preserve all instances.

# `tiff_viewer.ipynb`
Loads and interactively explores multi-slice TIFF images in 3D. A helpful tool for manually checking raw or segmented volumes.

# `verify_dataset_integrity.ipynb`
Checks for missing files, incorrect label values, and adherence to the expected directory structure. Helps catch formatting issues before training.
