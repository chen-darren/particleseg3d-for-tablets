# ParticleSeg3D Source Code
Darren, last edit 04/22/2025.

This directory includes the original ParticleSeg3D source code, along with custom additions and revisions made to improve and optimize the inference pipeline. These modifications include both standalone scripts (e.g., for quantitative analysis) and direct edits to the original source code.

# Performing Inference
## Set Correct Paths
Edit the `PathMaster` class located in: `/particleseg3d_source/utils/darren_func.py`

In the `get_base_path(self)` method, update each `'base_path'` to match your directory structure:
```python
paths = {
    'internal': r'C:\Senior_Design',
    'external': r'D:\Senior_Design',
    'cloud': r'C:\Users\dchen\OneDrive - University of Connecticut\Courses\Year 4\Fall 2024\BME 4900 and 4910W (Kumavor)\Python\Files',
    'refine': r'D:\Darren\Files'
}
```

Update the `output_to_cloud` paths in `setup_paths(self)` to the correct paths.
```python
if self.output_to_cloud:
            if self.dir_location == 'refine':
                self.base_output_path = os.path.join(r'D:\Darren\OneDrive - University of Connecticut\Courses\Year 4\Fall 2024\BME 4900 and 4910W (Kumavor)\Python\Files', 'outputs')
            else:
                self.base_output_path = os.path.join(r'C:\Users\dchen\OneDrive - University of Connecticut\Courses\Year 4\Fall 2024\BME 4900 and 4910W (Kumavor)\Python\Files', 'outputs')
```

## Modify Weight Loading Functions
The code is configured to use transfer learning weights specific to this project. If you choose to use your own weights or trainer, modify the default trainer string in the `main.setup_model` function:
```python
def setup_model(model_dir: str, folds: List[int], strategy: str = 'singleGPU', trainer: str = "nnUNetTrainerV2_ParticleSeg3D_DarrenSGD_CUDAErrorSkip__nnUNetPlansv2.1")
```

The default setting uses the final checkpoint. If you wish to switch to the best model checkpoint (not suggested), modify the default value of the checkpoing string:
```python
def __init__(self, model_dir: str, folds: Optional[Tuple[int, int, int, int, int]] = None, nnunet_trainer: str = "nnUNetTrainerV2__nnUNetPlansv2.1", 
             configuration: str = "3D", tta: bool = True, checkpoint: str = "model_final_checkpoint")
```

## Configurables
The following configurations control various aspects of the run, such as directory location, dataset selection, and multi-GPU processing.

### **1. Passing Parameters through the `main()` Function**

In this method, you set your configuration parameters directly when calling the `main()` function.

```python
def main(dir_location, output_to_cloud=False, is_original_data=False, weights_tag='Task502_Manual_Split_TL_Fold0', run_tag='No Run Tag Inputted', metadata='metadata', name=None, strategy='singleGPU', folds=(0, 1, 2, 3, 4), to_binary=False, psd=True, metrics=True):
    pathmaster = func.PathMaster(dir_location, output_to_cloud, run_tag, is_original_data, weights_tag)
    names = run_inference(pathmaster.grayscale_path, pathmaster.pred_zarr_path, pathmaster.weights_path, pathmaster.run_tag, metadata, name, strategy, folds=folds)
    func.convert_zarr_to_tiff(pathmaster.pred_zarr_path, pathmaster.pred_tiff_path, names, to_binary)
    if psd:
        part_size_dist.psd(pathmaster.pred_tiff_path, pathmaster.run_tag, names, pathmaster.psd_path)
    if metrics:
        sem_metrics.save_metrics(pathmaster.gt_sem_path, pathmaster.gt_inst_path, pathmaster.pred_tiff_path, pathmaster.sem_metrics_path, pathmaster.run_tag, names)
```

### **2. Using the `if __name__ == "__main__"`: Entry Point
Alternatively, you can define all configurations at the start of your script using the standard Python entry point. This method is useful for scripting, batch processing, or testing different configurations easily.

When calling `main()`, ensure the following configurable parameters are set. If not specified, the function uses the default values:
1. **Directory Location** (`dir_location`):  
   - **Description**: Specifies the location type for the dataset and outputs.
   - **Options**:  
     - `'internal'` for an internal hard drive or SSD
     - `'external'` for an external hard drive or SSD
     - `'cloud'` for OneDrive
     - `'refine'` for REFINE Lab PC
   - **Default**: No default value. Must be set.

2. **Output to Cloud Override** (`output_to_cloud`):  
   - **Description**: If `True`, saves output to the cloud location regardless of the directory location type.
   - **Default**: `False`

3. **Dataset** (`is_original_data`):  
   - **Description**: Choose between DigiM tablet dataset or the original ParticleSeg3D authors' dataset.
   - **Options**:  
     - `False` for DigiM tablet dataset
     - `True` for the original authors' dataset
   - **Default**: `False`

4. **Weights** (`weights_tag`):  
   - **Description**: Tag for the weights to use.
   - **Default**: `'Task502_Manual_Split_TL_Fold0'`

5. **Run Tag** (`run_tag`):  
   - **Description**: Custom tag for the run, used for naming output files.
   - **Default**: `'No Run Tag Inputted'`

6. **Metadata** (`metadata`):  
   - **Description**: Specify the metadata identifier for the `.json` file.
   - **Example**: `'tab40_gen35_clar35'` which corresponds to the `tab40_gen35_clar35.json` metadata file.
   - **Default**: `'metadata'`

7. **Image Names** (`name`):  
   - **Description**: Optional list of image names to run. If `None`, all images in the input directory will be processed.
   - **Default**: `None`

8. **Strategy** (`strategy`):  
   - **Description**: Defines the multi-GPU processing strategy.
   - **Options**:  
     - `'dp'` for DataParallel
     - `'ddp'` for DistributedDataParallel (not validated, no guarantee it works)
     - `'singleGPU'` for single GPU processing
   - **Default**: `'singleGPU'`

9. **Folds** (`folds`):  
   - **Description**: A tuple/list containing the folds to use in the ensemble for inference.
   - **Default**: `(0, 1, 2, 3, 4)` (All folds)

10. **Binary Mask Output** (`to_binary`):  
    - **Description**: If `True`, converts the TIFF output to a binary mask TIFF stack.
    - **Default**: `False`

11. **PSD Computation** (`psd`):  
    - **Description**: Set to `True` to compute and save the particle size distributions (PSDs) for analysis.
    - **Default**: `True`

12. **Metrics Computation** (`metrics`):  
    - **Description**: Set to `True` to compute and save the semantic metrics for quantitative analysis.
    - **Default**: `True`

## Ready to Go!
Once all the configurable parameters, paths, directories, and other necessary settings have been correctly configured, you are ready to run the main script.

Simply execute `/particleseg3d_source/main.py` to start inference.

# Revisions and Optimizations to the Original Source Code
Please note that additional revisions or optimizations may have been made to the source code that are not explicitly documented here, as not every change was meticulously tracked.
### 1. **Python Integration and CLI-Free Execution**
- Converted the original `inference.py` into `main.py`, allowing the code to be run directly in a Python IDE such as VS Code without relying on the command line.

### 2. **Multi-GPU Processing Support**
- Modified `/particleseg3d_source/main.py` to enable multi-GPU inference using `DataParallel`.
- `DistributedDataParallel` (DDP) was added as an option but remains untested.

### 3. **TIFF Output Support**
- Enhanced `/particleseg3d_source/main.py` to save predicted segmentations as TIFF slices in addition to Zarr format.
- This makes visual inspection of outputs easier, without needing specialized software.

### 4. **File Access Conflict Resolution in Parallel Execution**
- `DataParallel` caused inconsistent file access conflicts in `/particleseg3d_source/inference/aggregator.py`.
- Introduced a try-except block to:
  - Wait a random interval (2–5 seconds) on `PermissionError`
  - Retry up to 50 times
- This solution allows GPUs to stagger access, mitigating conflicts.

### 5. **Image-Specific Z-Score Intensity Normalization**
- Replaced the static normalization constants with dynamic, image-specific z-score normalization.
- Implemented directly in `/particleseg3d_source/main.py`.

### 6. **Enhanced Dataset Preprocessing for nnU-Net**
- Reworked `/particleseg3d_source/train/preproccessing.py` to:
  - Use image-specific z-score normalization
  - Avoid out-of-memory (OOM) issues during resampling by:
    - Splitting large images in half with 25% overlap
    - Running resampling on two GPUs
    - Moving non-computation tensors to CPU
  - Enable dataset preprocessing for training nnU-Net with border-core and instance segmentation even when only binary or border-core labels are available

### 7. **Metrics and PSD Computation Integration**
- Implemented semantic segmentation metrics and particle size distribution (PSD) computation.
- Integrated both into the inference pipeline via `/particleseg3d_source/main.py`.

### 8. **Customized Weight Loading**
- Adjusted weight loading in `/particleseg3d_source/main.py` to support transfer learning models.
- In `/particleseg3d_source/inference/model_nnunet.py`, changed the checkpoint default from `"model_best"` to `"model_final_checkpoint"`, as the final model generally performs better.

### 9. **Postprocessing Performance Optimization**

**a. Performance Bottleneck**
- Major slowdown identified in converting border-core to instance segmentation—especially with:
  - Large connected patches
  - High particle counts

**b. Optimization 1: Skip Small Core Removal**
- Disabled the small core removal step:
  - Minor impact on accuracy (±1e-5 to 1e-4 in metrics)
  - Reduced runtime by 50%+ for bottlenecked tablets

**c. Optimization 2: Code-level Improvements**
- Replaced `scipy.ndimage.label` with `cc3d.connected_components`
- Replaced per-label masking/writing with subregion masking/writing
- Achieved ~10× speedup for bottlenecked tablets
- These changes **did not affect segmentation results**

# Important Setup Notes
## Virtual Environment
Anaconda and its virtual environments should be used.
- See `/setup/conda_env/conda_env_source_notes.txt` for details

## Directory Structure
The directories for the database, outputs, and weights should be organized as follows:  
> Note: Some subdirectories within the `outputs` directory may be automatically generated at runtime.  
> Additionally, within `/database/tablet_dataset`, only the `grayscale`, `binary`, and `instance` subdirectories are strictly required for inference:  
> - `grayscale` is used as the input to the model  
> - `instance` is required for Particle Size Distribution (PSD) analysis  
> - `binary` is used for computing semantic segmentation metrics  
```
.
├── database
    ├── tablet_dataset
        ├── grayscale
            ├── dataset
                ├── images
                    ├── 2_Tablet.zarr
                    ├── 4_GenericD12.zarr
                    ├── ...
                ├── metadata
                    ├── tab30_gen35.json
                    ├── tab40.json
                    ├── metadata.json
                    ├── ...
        ├── segmented
            ├── tiff
                ├── 2_Tablet
                    ├── slice_0000.tiff
                    ├── slice_0001.tiff
                    ├── ...
                ├── 4_GenericD12
                    ├── ...
        ├── binary
            ├── tiff
                ├── 2_Tablet
                    ├── slice_0000.tiff
                    ├── slice_0001.tiff
                    ├── ...
                ├── 4_GenericD12
                    ├── ...
        ├── instance
            ├── tiff
                ├── 2_Tablet
                    ├── slice_0000.tiff
                    ├── slice_0001.tiff
                    ├── ...
                ├── 4_GenericD12
                    ├── ...
        ├── bordercore
            ├── tiff
                ├── 2_Tablet
                    ├── slice_0000.tiff
                    ├── slice_0001.tiff
                    ├── ...
                ├── 4_GenericD12
                    ├── ...
    ├── ...
├── outputs
    ├── metrics
        ├── particle_size_dist
        ├── semantic_metrics
    ├── tiff
    ├── zarr
    ├── video
├── weights
    ├── Task502_Manual_Split_TL_Fold0
        ├── nnUNetTrainerV2_ParticleSeg3D_DarrenSGD_CUDAErrorSkip__nnUNetPlansv2.1
            ├── fold_0
            ├── fold_1
            ├── fold_2
            ├── fold_3
            ├── fold_4
            ├── plans.pkl
    ├── ...
```
# Transfer Learning Procedure

> **Note**: For all steps related to transfer learning, please follow the original procedure outlined in the [ParticleSeg3D GitHub repository](https://github.com/MIC-DKFZ/ParticleSeg3D) and its corresponding [ParticleSeg3D branch within the nnU-Net repository](https://github.com/MIC-DKFZ/nnUNet/tree/ParticleSeg3D).

Although this project includes revised preprocessing and training scripts (e.g., for memory handling, normalization, and compatibility), they are compatible with the original methodology. By following the procedures defined in the original repositories—especially for data structuring and training workflows—the modified scripts in this repository can be used effectively to perform transfer learning or train new models.
