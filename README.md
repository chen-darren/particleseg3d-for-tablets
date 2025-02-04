# Senior_Design

## How to Run
### Configurables
1. Set directory location type (not case sensitive)
    - `'internal'` for an internal hard drive or SSD
    - `'external'` for an external hard drive or SSD
    - `'cloud'` for OneDrive
    - `'refine'` for REFINE Lab PC
2. Set output to cloud override
    - If it is desired to save the output to the cloud location regardless of the directory location type, then set `output_to_cloud` to `True`. Set to `False` otherwise.
3. Set dataset
    - If the tablet dataset from DigiM is desired, set `is_original_data` to `False`. If the dataset used by the authors of ParticleSeg3D is desired, set to `True`.
4. Specify weights to use
    - Set `weights_tag` to the correct name of the weights that are desired. If the originally trained model is desired, set to `'original_particle_seg'`.
5. Specify run tag
    - Set `run_tag` to the identifier desired for the specified run and its outputs.
6. Specify conda environment
    - Set `conda_env` to the name of your conda environment (should not change between runs)

### Set Paths
- Set directory location
    - Specify each 'base_path' to the correct path.
    - ```
        # Set path to the base directory
        if dir_location.lower() == 'internal':
            base_path = r'\path\to\internal\drive'
        elif dir_location.lower() == 'external':
            base_path = r'\path\to\external\drive'
        elif dir_location.lower() == 'cloud':
            base_path = r'\path\to\cloud'
        elif dir_location.lower() == 'refine':
            base_path = r'\path\to\refine'

### Ready to Go!
- Can run all cells to run the model as well as convert results to TIFF or can run cell one by one.

## Important Notes
- Anaconda and its virtual environments should be used.
- Root directory must be setup as follows:
```
.
├── database
    ├── original_dataset
        ├── grayscale
            ├── dataset
                ├── images
                    ├── Ore6.zarr
                    ├── Slag3.zarr
                    ├── ...
                ├── metadata.json
        ├── segmented
    ├── tablet_dataset
        ├── grayscale
            ├── dataset
                ├── images
                    ├── 1_Microsphere_Grayscale.zarr
                    ├── 2_Tablet_Grayscale.zarr
                    ├── 3_SprayDriedDispersion_Grayscale.zarr
                    ├── ...
                ├── metadata.json
        ├── segmented
├── outputs
    ├── run_tag
    ├── ...
├── weights
    ├── orignal_particle_seg
    ├── weights_tag
    ├── ...