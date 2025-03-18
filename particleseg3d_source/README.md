# ParticleSeg3D Source Code Implementation

## How to Run
### Set Correct Paths
Go to `setup_paths` in `particleseg3d_source/utils/darren_func.py`
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

### Configurables
Go to the calling of the `main` function in `particleseg3d_source/main.py`
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
6. Specify metadata
    - Set `metadata` to the identifier for the desired .json file (i.e. `tab40_gen35_clar35`). Defaults to `metadata` for `metadata.json`.
7. Specify names of images (Optional)
    - Set `name` to the names of the desired images to be run. If nothing is passed, then automatically runs all images in the input directory.
8. Specifiy type of strategy to use for multi-GPU processing (if at all)
    - `'dp'` for DataParallel
    - `'ddp'` for DistributedDataParallel with GLOO Backend (NCCL is not supported on Windows)
    - `'singleGPU'` for no multi-GPU processing
9. Specify the folds desired for the ensemble
    - Set `folds` to the desired tuple/list containing `0, 1, 2, 3, and/or 4`. Defaults to all folds: `(0, 1, 2, 3, 4)`.
10. Specify whether the TIFF output should be saved as a binary mask
    - Set `is_binary` to `True` to convert the Zarr into a binary mask TIFF stack
11. Specify whether or not metrics shoudl be computed
    - Set `metrics` to `True` to compute and save metrics

### Ready to Go!
- Can run `main.py` to perform inference.

## Important Notes
- Anaconda and its virtual environments should be used.
    - See `/conda_env/conda_env_source_notes.txt` for updated details
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
                ├── metadata
                    ├── run_tag.json
                    ├── ...
        ├── segmented
    ├── tablet_dataset
        ├── grayscale
            ├── dataset
                ├── images
                    ├── 1_Microsphere_Grayscale.zarr
                    ├── 2_Tablet_Grayscale.zarr
                    ├── 3_SprayDriedDispersion_Grayscale.zarr
                    ├── ...
                ├── metadata
                    ├── run_tag.json
                    ├── ...
        ├── segmented
├── outputs
    ├── run_tag
    ├── ...
├── weights
    ├── orignal_particle_seg
    ├── weights_tag
    ├── ...
