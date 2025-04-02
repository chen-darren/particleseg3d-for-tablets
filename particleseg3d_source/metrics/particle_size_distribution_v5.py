import os
import numpy as np
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis
from skimage import measure
from scipy.spatial.distance import pdist
from math import sqrt


def load_data_from_dir(input_dir):
    """Load 3D instance segmentation from a directory of 2D TIFF slices."""
    file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.tiff')])
    if not file_list:
        raise ValueError(f"No TIFF images found in {input_dir}")
    return np.stack([tiff.imread(os.path.join(input_dir, f)) for f in file_list])

def compute_psd(segmentation):
    """Compute volume, surface area, diameter, and sphericity for each particle in the 3D segmentation."""
    labels = segmentation
    props = regionprops(labels)

    def compute_metrics(prop):
        if prop.label == 0:  # Exclude background
            return None

        instance_label = prop.label  # Instance ID
        volume = prop.area  # Voxel count as volume
        
        bbox = prop.bbox
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        surface_area = (
            np.sum(segmentation[min_x+1:max_x, min_y:max_y, min_z:max_z] != 
                   segmentation[min_x:max_x-1, min_y:max_y, min_z:max_z]) +
            np.sum(segmentation[min_x:max_x, min_y+1:max_y, min_z:max_z] != 
                   segmentation[min_x:max_x, min_y:max_y-1, min_z:max_z]) +
            np.sum(segmentation[min_x:max_x, min_y:max_y, min_z+1:max_z] != 
                   segmentation[min_x:max_x, min_y:max_y, min_z:max_z-1])
        )
        
        # Get other regionprops properties
        diameter = prop.equivalent_diameter_area # (6 * volume / np.pi) ** (1/3)

        # Compute sphericity: the ratio of the surface area of a sphere with the same volume as the object to the surface area of the object itself
        sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area if surface_area > 0 else 0
        if sphericity > 1:
            sphericity = 0

        return instance_label, volume, surface_area, diameter, sphericity
    
    results = Parallel(n_jobs=-1)(delayed(compute_metrics)(prop) for prop in props)
    results = [r for r in results if r is not None]  # Remove None values

    if results:
        instance_labels, volumes, surface_areas, diameters, sphericities = zip(*results)
    else:
        instance_labels, volumes, surface_areas, diameters, sphericities = ([], [], [], [], [])

    return [np.array(instance_labels), np.array(volumes), np.array(surface_areas), np.array(diameters), np.array(sphericities)]

def filter_by_iqr(raw_psd_metrics, threshold_factor=7.5):
    """Remove outliers using the IQR method and return filtered data."""
    if len(raw_psd_metrics[0]) == 0:
        return raw_psd_metrics  # Return empty if no data

    Q1 = np.percentile(raw_psd_metrics[1], 25)
    Q3 = np.percentile(raw_psd_metrics[1], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold_factor * IQR
    upper_bound = Q3 + threshold_factor * IQR

    mask = (raw_psd_metrics[1] >= lower_bound) & (raw_psd_metrics[1] <= upper_bound)
    return [raw_psd_metrics[0][mask], raw_psd_metrics[1][mask], raw_psd_metrics[2][mask], raw_psd_metrics[3][mask], raw_psd_metrics[4][mask]]

def filter_by_threshold(raw_psd_metrics, threshold=('diameter', 50)):
    """Filter particles based on a specified threshold for various metrics."""
    # Apply the appropriate threshold based on the specified metric
    if threshold[0] == 'volume':
        mask = raw_psd_metrics[1] <= threshold[1]
    elif threshold[0] == 'surface area':
        mask = raw_psd_metrics[2] <= threshold[1]
    elif threshold[0] == 'diameter':
        mask = raw_psd_metrics[3] <= threshold[1]
    else:
        raise ValueError('The threshold type must be one of the following: `volume`, `surface area`, or `diameter`!')
    
    # Return the filtered data based on the threshold
    return [raw_psd_metrics[0][mask], raw_psd_metrics[1][mask], raw_psd_metrics[2][mask], raw_psd_metrics[3][mask], raw_psd_metrics[4][mask]], threshold

def bin_data(data, bin_edges):
    """Bin the data according to the bin edges and return the bin counts."""
    counts, _ = np.histogram(data, bins=bin_edges)
    return counts

def save_psd_to_csv(psd_metrics, save_path):
    """Apply outlier removal using IQR and save results to CSV."""
    df = pd.DataFrame({
        "Instance": psd_metrics[0],
        "Volume": psd_metrics[1],
        "Surface Area": psd_metrics[2],
        "Diameter": psd_metrics[3],
        "Sphericity": psd_metrics[4]
    })
    df.to_csv(save_path, index=False)
    print(f"Saved PSD as CSV to {save_path}")

def save_histogram(data, bin_edges, title, xlabel, save_path):
    """Save histogram plot with scaled x-axis, excluding zero values."""
    data = data[data > 0]  # Exclude zero values
    if len(data) == 0:
        print(f"Warning: No valid data for histogram {title}. Skipping.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bin_edges, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel("Particle Count")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_binned_psd(psd_metrics, save_path, num_bins=50, threshold=None):
    """Save binned Particle Size Distribution (PSD) to CSV for multiple metrics."""
    volume_min = 0
    surface_area_min = 0
    diameter_min = 0
    sphericity_min = 0
    
    # Range from 0 to 1
    sphericity_max = 1
    
    if threshold is None:
        # Calculate the range for each dataset and round accordingly
        volume_max = np.ceil(max(psd_metrics[1]))   # Round up
        surface_area_max = np.ceil(max(psd_metrics[2]))   # Round up
        diameter_max = np.ceil(max(psd_metrics[3]))   # Round up
    else:
        if threshold[0] == 'volume':
            volume_max = threshold[1]
            diameter_max = np.ceil((6 * volume_max / np.pi) ** (1/3))
            surface_area_max = np.ceil(4 * np.pi * (diameter_max / 2) ** 2)
        elif threshold[0] == 'surface area':
            surface_area_max = threshold[1]
            diameter_max = np.ceil(np.sqrt(surface_area_max / (4 * np.pi)) * 2)
            volume_max = np.ceil((4/3) * np.pi * (diameter_max / 2) ** 3)
            print('Please note that thresholding based on surface area may lead to odd binned histograms due to the very approximate estimation of surface area that is not directly related to volume or diameter.')
        elif threshold[0] == 'diameter':
            diameter_max = threshold[1]
            volume_max = np.ceil((4/3) * np.pi * (diameter_max / 2) ** 3)
            surface_area_max = np.ceil(4 * np.pi * (diameter_max / 2) ** 2)
        else:
            raise ValueError('The threshold type must be one of the following: `volume`, `surface area`, or `diameter`!')

    def create_bins(min_val, max_val, num_bins):
        bin_range = max_val - min_val
        if bin_range >= num_bins:
            return np.linspace(min_val, max_val, num_bins + 1, dtype=int)
        else:
            extra_bins = num_bins - bin_range  # Add extra bins if range is small
            return np.linspace(min_val, max_val + extra_bins, num_bins + 1, dtype=int)

    def create_log_bins(min_val, max_val, num_bins, epsilon=1e-6):
        # Shift values to avoid log(0) error
        min_val = max(min_val, epsilon)
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        log_bins = np.logspace(log_min, log_max, num_bins + 1)
        return np.unique(log_bins)
    
    def create_power_bins(min_val, max_val, num_bins, power=2):
        # Create bins using a power law scaling
        bins = np.linspace(min_val ** power, max_val ** power, num_bins + 1) ** (1 / power)
        return np.unique(np.round(bins).astype(int))

    # Create bins for volume and surface area using logarithmic spacing
    volume_bins = create_bins(volume_min, volume_max, num_bins)
    surface_area_bins = np.linspace(surface_area_min, surface_area_max, num_bins + 1)
    diameter_bins = np.linspace(diameter_min, diameter_max, num_bins + 1)
    sphericity_bins = np.linspace(sphericity_min, sphericity_max, num_bins + 1)

    volume_counts = bin_data(psd_metrics[1], volume_bins)
    surface_area_counts = bin_data(psd_metrics[2], surface_area_bins)
    diameter_counts = bin_data(psd_metrics[3], diameter_bins)
    sphericity_counts = bin_data(psd_metrics[4], sphericity_bins)

    volume_bin_ranges = [f"{v1}.0-{v2}.0" for v1, v2 in zip(volume_bins[:-1], volume_bins[1:])]
    surface_area_bin_ranges = [f"{s1}-{s2}" for s1, s2 in zip(surface_area_bins[:-1], surface_area_bins[1:])]
    diameter_bin_ranges = [f"{d1}-{d2}" for d1, d2 in zip(diameter_bins[:-1], diameter_bins[1:])]
    sphericity_bin_ranges = [f"{d1}-{d2}" for d1, d2 in zip(sphericity_bins[:-1], sphericity_bins[1:])]
    
    binned_df = pd.DataFrame({
        "Volume Bin Range": volume_bin_ranges,
        "Volume Count": volume_counts,
        "Surface Area Bin Range": surface_area_bin_ranges,
        "Surface Area Count": surface_area_counts,
        "Diameter Bin Range": diameter_bin_ranges,
        "Diameter Count": diameter_counts,
        "Sphericity Bin Range": sphericity_bin_ranges,
        "Sphericity Count": sphericity_counts
    })

    binned_df.to_csv(save_path, index=False)
    print(f"Saved binned PSD results to {save_path}")

    return [volume_bins, surface_area_bins, diameter_bins, sphericity_bins]

def save_summary_metrics(psd_metrics, psd_metric_names, save_dir, run_tag, name, prefix):
    """Compute and save summary metrics (mean, std, median, IQR, percentiles, skewness, kurtosis) for PSD data."""
    
   # Compute basic statistics
    total_particles = len(psd_metrics[0])
    total_volume = np.sum(psd_metrics[1])
    total_surface_area = np.sum(psd_metrics[2])
    
    summary = {
        "Total Particles": total_particles,
        "Total Volume": total_volume,
        "Total Surface Area": total_surface_area
    }
    
    if len(psd_metrics) - 1 != len(psd_metric_names):
        raise RuntimeError("Mismatch between the number of PSD metrics and metric names.")
    
    for psd_metric, psd_metric_name in zip(psd_metrics[1:], psd_metric_names):
        skewness = skew(psd_metric)
        kurt = kurtosis(psd_metric)
        percentiles = np.percentile(psd_metric, [25, 50, 75])

        # Add calculated metrics to summary
        summary.update({
            f"Mean {psd_metric_name}": np.mean(psd_metric),
            f"Std {psd_metric_name}": np.std(psd_metric),
            f"Median {psd_metric_name}": np.median(psd_metric),
            f"IQR {psd_metric_name}": percentiles[2] - percentiles[0],
            
            f"{psd_metric_name} 25th Percentile": percentiles[0],
            f"{psd_metric_name} 50th Percentile (Median)": percentiles[1],
            f"{psd_metric_name} 75th Percentile": percentiles[2],
            
            f"{psd_metric_name} Skewness": skewness,
            f"{psd_metric_name} Kurtosis": kurt,
        })
    
    # Convert the summary dictionary to a DataFrame
    summary_df = pd.DataFrame(summary, index=[0])

    # Create the directory to save the summary file
    summary_folder = os.path.join(save_dir, "summary", run_tag, name)
    os.makedirs(summary_folder, exist_ok=True)

    # Define the path to save the summary CSV file
    summary_path = os.path.join(summary_folder, f"{name}_{prefix}_summary.csv")
    
    # Save the summary DataFrame to CSV
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Saved summary metrics for {name} ({prefix}) to {summary_path}")

def psd(input_dir, run_tag, names, save_dir, save=True):
    """Analyze 3D instance segmentation and compute particle size distribution (PSD)."""
    for name in names:
        segmentation = load_data_from_dir(os.path.join(input_dir, name))
        raw_psd_metrics = compute_psd(segmentation)

        if save:
            table_folder = os.path.join(save_dir, "table", run_tag, name)
            hist_folder = os.path.join(save_dir, "histogram", run_tag, name)
            os.makedirs(table_folder, exist_ok=True)
            os.makedirs(hist_folder, exist_ok=True)

            # Save original CSV
            orignal_csv_path = os.path.join(table_folder, f"{name}_raw_psd.csv")
            save_psd_to_csv(raw_psd_metrics, orignal_csv_path)

            # Save IQR-filtered CSV
            iqr_filt_psd_metrics = filter_by_iqr(raw_psd_metrics)
            iqr_filt_csv_path = os.path.join(table_folder, f"{name}_iqr_filt_psd.csv")
            save_psd_to_csv(iqr_filt_psd_metrics, iqr_filt_csv_path)

            # Save threshold-filtered CSV
            thresh_filt_psd_metrics, threshold = filter_by_threshold(raw_psd_metrics)
            thresh_filt_csv_path = os.path.join(table_folder, f"{name}_thresh_filt_psd.csv")
            save_psd_to_csv(thresh_filt_psd_metrics, thresh_filt_csv_path)

            # Save binned PSD CSVs
            raw_psd_bins = save_binned_psd(raw_psd_metrics, os.path.join(table_folder, f"{name}_raw_binned_psd.csv"), num_bins=75)
            iqr_filt_psd_bins = save_binned_psd(iqr_filt_psd_metrics, os.path.join(table_folder, f"{name}_iqr_filt_binned_psd.csv"), num_bins=75)
            thresh_filt_psd_bins = save_binned_psd(thresh_filt_psd_metrics, os.path.join(table_folder, f"{name}_thresh_filt_binned_psd.csv"), num_bins=75, threshold=threshold)

            # Save histograms
            psd_metric_names = ['Volume', 'Surface Area', 'Diameter', 'Sphericity']
            psd_metric_tags = ['volume', 'surface', 'diameter', 'sphericity']

            if len(psd_metric_names) != len(psd_metric_tags):
                raise RuntimeError("Mismatch between the number of PSD metric names and tags.")
            
            # Raw PSD values
            if not (len(raw_psd_metrics) - 1 == len(raw_psd_bins) == len(psd_metric_names) == len(psd_metric_tags)):
                raise RuntimeError("Mismatch between the number of raw PSD metrics, bins, names, and tags.")
            for raw_psd_metric, raw_psd_bin, psd_metric_name, psd_metric_tag in zip(raw_psd_metrics[1:], raw_psd_bins, psd_metric_names, psd_metric_tags):
                if np.max(raw_psd_metric) <= 1:   
                    title = f"{psd_metric_name}"
                else:
                    title = f"{psd_metric_name} (voxels)"
                save_histogram(raw_psd_metric, raw_psd_bin, f"Raw Particle {psd_metric_name} Distribution", title, os.path.join(hist_folder, f"{name}_raw_{psd_metric_tag}_hist.png"))
                
            # IQR-filterd PSD values
            if not (len(iqr_filt_psd_metrics) - 1 == len(iqr_filt_psd_bins) == len(psd_metric_names) == len(psd_metric_tags)):
                raise RuntimeError("Mismatch between the number of IQR-filtered PSD metrics, bins, names, and tags.")
            for iqr_filt_psd_metric, iqr_filt_psd_bin, psd_metric_name, psd_metric_tag in zip(iqr_filt_psd_metrics[1:], iqr_filt_psd_bins, psd_metric_names, psd_metric_tags):
                if np.max(iqr_filt_psd_metric) <= 1:   
                    title = f"{psd_metric_name}"
                else:
                    title = f"{psd_metric_name} (voxels)"
                save_histogram(iqr_filt_psd_metric, iqr_filt_psd_bin, f"IQR-Filtered Particle {psd_metric_name} Distribution", title, os.path.join(hist_folder, f"{name}_iqr_filt_{psd_metric_tag}_hist.png"))
                
            # Threshold-filterd PSD values
            if not (len(thresh_filt_psd_metrics) - 1 == len(thresh_filt_psd_bins) == len(psd_metric_names) == len(psd_metric_tags)):
                raise RuntimeError("Mismatch between the number of threshold-filtered PSD metrics, bins, names, and tags.")
            for thresh_filt_psd_metric, thresh_filt_psd_bin, psd_metric_name, psd_metric_tag in zip(thresh_filt_psd_metrics[1:], thresh_filt_psd_bins, psd_metric_names, psd_metric_tags):
                if np.max(thresh_filt_psd_metric) <= 1:   
                    title = f"{psd_metric_name}"
                else:
                    title = f"{psd_metric_name} (voxels)"
                save_histogram(thresh_filt_psd_metric, thresh_filt_psd_bin, f"Threshold-Filtered Particle {psd_metric_name} Distribution", title, os.path.join(hist_folder, f"{name}_thresh_filt_{psd_metric_tag}_hist.png"))

            # Save summary metrics
            lists_of_psd_metrics = [raw_psd_metrics, iqr_filt_psd_metrics, thresh_filt_psd_metrics]
            prefixes = ['raw', 'iqr_filt', 'thresh_filt']

            if len(lists_of_psd_metrics) != len(prefixes):
                raise RuntimeError("Mismatch between the number of PSD metric lists and prefixes.")
            
            for list_of_psd_metrics, prefix in zip(lists_of_psd_metrics, prefixes):
                save_summary_metrics(list_of_psd_metrics, psd_metric_names, save_dir, run_tag, name, prefix=prefix)

            print(f"Saved results for {name}.\n")