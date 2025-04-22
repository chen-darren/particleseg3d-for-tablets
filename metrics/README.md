# Metrics for Quantitative Analysis
Darren, last edit 04/22/2025.

This directory contains notebooks designed to analyze the quality of predicted segmentations. Two types of quantitative analyses are provided: Particle Size Distribution (PSD), which focuses on the size and shape of each predicted particle, and Segmentation Metrics, which assess the quality of the binary mask, distinguishing between particle and background.

# Particle Size Distribution (PSD)
Please note that this code is located in `/particleseg3d_source/metrics/particle_size_distribution.py`, but is presented here in the form of a Jupyter Notebook.

## Functionality
- **Computes PSD metrics**: Calculates **volume**, **sphericity**, **surface area**, and **diameter** for each particle.

- **Filters PSD data**: Applies different filters:
    - **Raw**: No filtering.
    - **IQR-filtered**: Filters out outliers based on the Interquartile Range (IQR).
        - The threshold factor can be set to any float value.
            - Modify the parameter in `filter_by_iqr(args)` to adjust the threshold.
    - **Threshold-filtered**: Filters based on a specified threshold.
        - The threshold can be set for any of the four PSD metrics (volume, sphericity, surface area, and diameter).
            - Modify the parameter in `filter_by_threshold(args)` to adjust the threshold.

- **Saves outputs**:
  - **Histograms** for each particle metric (volume, sphericity, surface area, diameter) in both raw and filtered forms.
  - **CSV Tables** containing raw, IQR-filtered, and threshold-filtered PSD metrics.
  - **Comparison plots** comparing the raw/IQR-filtered/threshold-filtered results to their corresponding ground truth data.
  - **Summary metrics** for each of the filtering methods.

## Parameters
Run PSD by using `psd(input_dir, run_tag, names, save_dir, save=True)`.

- **`input_dir` (str)**:  
  Path to the directory containing the TIFF files for the segmented data that you want to analyze.

- **`run_tag` (str)**:  
  A string tag associated with the run, used to organize output folders and differentiate between different analyses (e.g., 'ground_truth' or other specific identifiers).

- **`names` (list of str)**:  
  A list of subdirectories or filenames to process. Each entry corresponds to a specific image to be analyzed.

- **`save_dir` (str)**:  
  Path to the directory where the output (CSV files, histograms, comparison plots) will be saved.

- **`save` (bool, default=True)**:  
  A flag to indicate whether to save the output files (CSV, histograms, comparison plots, etc.). If set to `False`, the analysis will be performed but not saved.

## Outputs
- **Histograms**:  
  Plots for each particle metric (volume, sphericity, surface area, and diameter) in raw, IQR-filtered, and threshold-filtered forms. These histograms are saved in the specified `save_dir` under subdirectories named `histogram`.

- **CSV Tables**:  
  Tables of raw, IQR-filtered, and threshold-filtered PSD metrics are saved in CSV format. These are stored under subdirectories named `table` within the `save_dir`.

- **Comparison Plots**:  
  Plots comparing raw, IQR-filtered, and threshold-filtered distributions to their ground truth counterparts. These comparison plots are saved in the `comparison` subdirectory within the `save_dir`.

- **Summary Metrics**:  
  For each of the filtering methods (raw, IQR-filtered, and threshold-filtered), summary statistics of the PSD metrics are saved.

All the results are organized by the `run_tag` and the image name in the directory structure, making it easy to distinguish between different analyses and images.

# Semantic Metrics
Please note that this code is located in `/particleseg3d_source/metrics/semantic_metrics.py`, but is presented here in the form of a Jupyter Notebook.

## Functionality
- **Computes Semantic Metrics**: Calculates standard performance metrics for predicted segmentation results.
    - **IoU** (Intersection over Union)
    - **Dice coefficient**
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **Specificity**
    - **False Positive Rate (FPR)**
    - **False Negative Rate (FNR)**

- **Saves Outputs**:
    - **CSV Tables** containing the computed metrics for each image.
    - **Updates Existing Results**: If a previous results file exists, new results are added or updated. If no file exists, a new one is created.
    - **Results File**: The results are saved in a CSV file, `semantic_metrics.csv`, containing all metrics computed for each image.

## Parameters
Run Semantic Metrics by using `save_metrics(gt_sem_path, gt_inst_path, pred_path, results_path, run_tag, img_names)`.

- **`gt_sem_path` (str)**:  
  Path to the directory containing the ground truth semantic segmentation data.

- **`gt_inst_path` (str)**:  
  Path to the directory containing the ground truth instance segmentation data.

- **`pred_path` (str)**:  
  Path to the directory containing the predicted segmentation results.

- **`results_path` (str)**:  
  Path to the directory where the results (CSV file) will be saved.

- **`run_tag` (str)**:  
  A string tag associated with the run, used to organize output and differentiate between different analyses.

- **`img_names` (list of str)**:  
  A list of image names or subdirectories to process. Each entry corresponds to a specific image to be analyzed for semantic metrics.

## Outputs
- **`semantic_metrics.csv`**: A CSV file that contains the following columns:
    - `run_tag`: The tag associated with the current run.
    - `image_name`: The name of the image being processed.
    - Metrics: **IoU**, **Dice**, **Accuracy**, **Precision**, **Recall**, **Specificity**, **FPR**, and **FNR** for each image.
  
If an existing CSV file is found, the new metrics are appended and updated; if no file exists, a new file is created.
