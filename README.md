# Senior_Design

**How to run:**
1. Set directory location
    - Specify each 'base_path' to the correct path.
    - ```
        # Set path to the base directory
        if dir_location.lower() == 'internal':
            base_path = r'\path_to_internal_drive'
        elif dir_location.lower() == 'external':
            base_path = r'\path_to_external_drive'
        elif dir_location.lower() == 'cloud':
            base_path = r'\path_to_cloud'
2. Set directory location type (not case sensitive)
    - `'internal'` for an internal hard drive or SSD
    - `'external'` for an external hard drive or SSD
    - `'cloud'` for OneDrive
3. Set output to cloud override
    - If it is desired to save the output to the cloud location regardless of the directory location type, then set `output_to_cloud` to `True`. Set to `False` otherwise.
4. Set dataset
    - If the tablet dataset from DigiM is desired, set `is_original_data` to `False`. If the dataset used by the authors of ParticleSeg3D is desired, set to `True`.
5. Specify weights to use
    - Set `weights_tag` to the correct name of the weights that are desired. If the originally trained model is desired, set to `'original_particle_seg'`.
6. Specify run tag
    - Set `'run_tag'` to the identifier desired for the specified run and its outputs.
7. Ready to go!
    - Can run all cells to run the model as well as convert results to TIFF or can run cell one by one.

**Note**
Root directory must be setup as follows:
```
.
├── database
    ├── original_dataset
        ├── dataset
            ├── images
                ├── Ore6.zarr
                ├── Slag3.zarr
                ├── ...
            ├── metadata.json
    ├── tablet_dataset
        ├── dataset
            ├── images
                ├── 1_Microsphere_Grayscale.zarr
                ├── 2_Tablet_Grayscale.zarr
                ├── 3_SprayDriedDispersion_Grayscale.zarr
                ├── ...
            ├── metadata.json
├── outputs
    ├── run_tag
    ├── ...
├── weights
    ├── orignal_particle_seg
    ├── weights_tag
    ├── ...