import os
import numpy as np
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis

def load_data_from_dir(input_dir):
    """Load 3D instance segmentation from a directory of 2D TIFF slices."""
    file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.tiff')])
    if not file_list:
        raise ValueError(f"No TIFF images found in {input_dir}")
    return np.stack([tiff.imread(os.path.join(input_dir, f)) for f in file_list])

def compute_psd(segmentation):
    """Compute volume, surface area, and diameter for each particle in the 3D segmentation."""
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
        diameter = (6 * volume / np.pi) ** (1/3)  # Spherical equivalent diameter
        return instance_label, volume, surface_area, diameter

    results = Parallel(n_jobs=-1)(delayed(compute_metrics)(prop) for prop in props)
    results = [r for r in results if r is not None]  # Remove None values
    instance_labels, volumes, surface_areas, diameters = zip(*results) if results else ([], [], [], [])

    return np.array(instance_labels), np.array(volumes), np.array(surface_areas), np.array(diameters)

def filter_by_iqr(instance_labels, volume, surface_areas, diameters, threshold_factor=7.5):
    """Remove outliers using the IQR method and return filtered data."""
    if len(volume) == 0:
        return instance_labels, volume, surface_areas, diameters  # Return empty if no data

    Q1 = np.percentile(volume, 25)
    Q3 = np.percentile(volume, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold_factor * IQR
    upper_bound = Q3 + threshold_factor * IQR

    mask = (volume >= lower_bound) & (volume <= upper_bound)
    return instance_labels[mask], volume[mask], surface_areas[mask], diameters[mask]

def filter_by_volume(instance_labels, volumes, surface_areas, diameters, volume_threshold=6000):
    """Filter particles based on a volume threshold."""
    mask = volumes <= volume_threshold
    return instance_labels[mask], volumes[mask], surface_areas[mask], diameters[mask]

def bin_data(data, bin_edges):
    """Bin the data according to the bin edges and return the bin counts."""
    counts, _ = np.histogram(data, bins=bin_edges)
    return counts

def save_psd_to_csv(instance_labels, volumes, surface_areas, diameters, save_path):
    """Apply outlier removal using IQR and save results to CSV."""
    df = pd.DataFrame({
        "Instance": instance_labels,
        "Volume": volumes,
        "Surface Area": surface_areas,
        "Diameter": diameters
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

def save_binned_psd(volumes, surface_areas, diameters, save_path, num_bins=50, fixed=False):
    """Save binned Particle Size Distribution (PSD) to CSV."""
    if not fixed:
        # Calculate the range for each dataset and round accordingly
        volume_min = np.floor(min(volumes))  # Round down
        volume_max = np.ceil(max(volumes))   # Round up
        surface_area_min = np.floor(min(surface_areas))  # Round down
        surface_area_max = np.ceil(max(surface_areas))   # Round up
        diameter_min = np.floor(min(diameters))  # Round down
        diameter_max = np.ceil(max(diameters))   # Round up
    else:
        volume_min = 0
        volume_max = 6000
        surface_area_min = 0
        surface_area_max = 6000
        diameter_min = 0
        diameter_max = 25

    def create_bins(min_val, max_val, num_bins):
        bin_range = max_val - min_val
        if bin_range >= num_bins:
            return np.linspace(min_val, max_val, num_bins + 1, dtype=int)
        else:
            extra_bins = num_bins - bin_range  # Add extra bins if range is small
            return np.linspace(min_val, max_val + extra_bins, num_bins + 1, dtype=int)

    volume_bins = create_bins(volume_min, volume_max, num_bins)
    surface_area_bins = create_bins(surface_area_min, surface_area_max, num_bins)
    diameter_bins = np.linspace(diameter_min, diameter_max, num_bins + 1)

    volume_counts = bin_data(volumes, volume_bins)
    surface_area_counts = bin_data(surface_areas, surface_area_bins)
    diameter_counts = bin_data(diameters, diameter_bins)

    volume_bin_ranges = [f"{v1}.0-{v2}.0" for v1, v2 in zip(volume_bins[:-1], volume_bins[1:])]
    surface_area_bin_ranges = [f"{s1}.0-{s2}.0" for s1, s2 in zip(surface_area_bins[:-1], surface_area_bins[1:])]
    diameter_bin_ranges = [f"{d1}.0-{d2}.0" for d1, d2 in zip(diameter_bins[:-1], diameter_bins[1:])]

    binned_df = pd.DataFrame({
        "Volume Bin Range": volume_bin_ranges,
        "Volume Count": volume_counts,
        "Surface Area Bin Range": surface_area_bin_ranges,
        "Surface Area Count": surface_area_counts,
        "Diameter Bin Range": diameter_bin_ranges,
        "Diameter Count": diameter_counts
    })

    binned_df.to_csv(save_path, index=False)
    print(f"Saved binned PSD results to {save_path}")

    return volume_bins, surface_area_bins, diameter_bins

def save_summary_metrics(volumes, surface_areas, diameters, save_dir, run_tag, name, prefix="raw"):
    """Compute and save summary metrics (mean, std, median, IQR, percentiles, skewness, kurtosis) for PSD data."""
    
    # Compute basic summary statistics
    total_particles = len(volumes)
    total_volume = np.sum(volumes)
    total_surface_area = np.sum(surface_areas)
    total_diameter = np.sum(diameters)
    
    avg_volume = np.mean(volumes)
    avg_surface_area = np.mean(surface_areas)
    avg_diameter = np.mean(diameters)
    
    median_volume = np.median(volumes)
    volume_skewness = skew(volumes)
    volume_kurtosis = kurtosis(volumes)
    
    volume_percentiles = np.percentile(volumes, [25, 50, 75])
    surface_area_percentiles = np.percentile(surface_areas, [25, 50, 75])
    diameter_percentiles = np.percentile(diameters, [25, 50, 75])
    
    # Create a dictionary with all the summary metrics
    summary = {
        # Basic statistics
        "Mean Volume": np.mean(volumes),
        "Std Volume": np.std(volumes),
        "Median Volume": np.median(volumes),
        "IQR Volume": np.percentile(volumes, 75) - np.percentile(volumes, 25),

        "Mean Surface Area": np.mean(surface_areas),
        "Std Surface Area": np.std(surface_areas),
        "Median Surface Area": np.median(surface_areas),
        "IQR Surface Area": np.percentile(surface_areas, 75) - np.percentile(surface_areas, 25),

        "Mean Diameter": np.mean(diameters),
        "Std Diameter": np.std(diameters),
        "Median Diameter": np.median(diameters),
        "IQR Diameter": np.percentile(diameters, 75) - np.percentile(diameters, 25),
        
        # Additional metrics
        "Total Particles": total_particles,
        "Total Volume": total_volume,
        "Total Surface Area": total_surface_area,
        "Total Diameter": total_diameter,
        "Average Volume": avg_volume,
        "Average Surface Area": avg_surface_area,
        "Average Diameter": avg_diameter,
        "Median Volume": median_volume,
        "Volume Skewness": volume_skewness,
        "Volume Kurtosis": volume_kurtosis,
        
        # Individual percentiles for volume, surface area, and diameter
        "Volume 25th Percentile": volume_percentiles[0],
        "Volume 50th Percentile (Median)": volume_percentiles[1],
        "Volume 75th Percentile": volume_percentiles[2],
        
        "Surface Area 25th Percentile": surface_area_percentiles[0],
        "Surface Area 50th Percentile (Median)": surface_area_percentiles[1],
        "Surface Area 75th Percentile": surface_area_percentiles[2],
        
        "Diameter 25th Percentile": diameter_percentiles[0],
        "Diameter 50th Percentile (Median)": diameter_percentiles[1],
        "Diameter 75th Percentile": diameter_percentiles[2]
    }
    
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
        instance_labels, volumes, surface_areas, diameters = compute_psd(segmentation)

        if save:
            table_folder = os.path.join(save_dir, "table", run_tag, name)
            hist_folder = os.path.join(save_dir, "histogram", run_tag, name)
            os.makedirs(table_folder, exist_ok=True)
            os.makedirs(hist_folder, exist_ok=True)

            # Save original CSV
            orignal_csv_path = os.path.join(table_folder, f"{name}_raw_psd.csv")
            save_psd_to_csv(instance_labels, volumes, surface_areas, diameters, orignal_csv_path)

            # Save IQR-filtered CSV
            iqr_filt_instance_labels, iqr_filt_volumes, iqr_filt_surface_areas, iqr_filt_diameters = filter_by_iqr(instance_labels, volumes, surface_areas, diameters)
            iqr_filt_csv_path = os.path.join(table_folder, f"{name}_iqr_filt_psd.csv")
            save_psd_to_csv(iqr_filt_instance_labels, iqr_filt_volumes, iqr_filt_surface_areas, iqr_filt_diameters, iqr_filt_csv_path)

            # Save volume-filtered CSV
            vol_filt_instance_labels, vol_filt_volumes, vol_filt_surface_areas, vol_filt_diameters = filter_by_volume(instance_labels, volumes, surface_areas, diameters)
            vol_filt_csv_path = os.path.join(table_folder, f"{name}_vol_filt_psd.csv")
            save_psd_to_csv(vol_filt_instance_labels, vol_filt_volumes, vol_filt_surface_areas, vol_filt_diameters, vol_filt_csv_path)

            # Save binned PSD CSVs
            raw_volume_bins, raw_surface_area_bins, raw_diameter_bins = save_binned_psd(volumes, surface_areas, diameters, os.path.join(table_folder, f"{name}_raw_binned_psd.csv"), num_bins=75)
            iqr_filt_volume_bins, iqr_filt_surface_area_bins, iqr_filt_diameter_bins = save_binned_psd(iqr_filt_volumes, iqr_filt_surface_areas, iqr_filt_diameters, os.path.join(table_folder, f"{name}_iqr_filt_binned_psd.csv"), num_bins=75)
            vol_filt_volume_bins, vol_filt_surface_area_bins, vol_filt_diameter_bins = save_binned_psd(vol_filt_volumes, vol_filt_surface_areas, vol_filt_diameters, os.path.join(table_folder, f"{name}_vol_filt_binned_psd.csv"), num_bins=75, fixed=True)

            # Save histograms
            save_histogram(volumes, raw_volume_bins, "Raw Particle Volume Distribution", "Volume (voxels)", os.path.join(hist_folder, f"{name}_raw_volume_hist.png"))
            save_histogram(surface_areas, raw_surface_area_bins, "Raw Particle Surface Area Distribution", "Surface Area (voxels)", os.path.join(hist_folder, f"{name}_raw_surface_hist.png"))
            save_histogram(diameters, raw_diameter_bins, "Raw Particle Diameter Distribution", "Diameter (voxels)", os.path.join(hist_folder, f"{name}_raw_diameter_hist.png"))

            # Save filtered histograms
            save_histogram(iqr_filt_volumes, iqr_filt_volume_bins, "IQR-Filtered Particle Volume Distribution", "Volume (voxels)", os.path.join(hist_folder, f"{name}_iqr_filt_volume_hist.png"))
            save_histogram(iqr_filt_surface_areas, iqr_filt_surface_area_bins, "IQR-Filtered Particle Surface Area Distribution", "Surface Area (voxels)", os.path.join(hist_folder, f"{name}_iqr_filt_surface_hist.png"))
            save_histogram(iqr_filt_diameters, iqr_filt_diameter_bins, "IQR-Filtered Particle Diameter Distribution", "Diameter (voxels)", os.path.join(hist_folder, f"{name}_iqr_filt_diameter_hist.png"))
            save_histogram(vol_filt_volumes, vol_filt_volume_bins, "Volume-Filtered Particle Volume Distribution", "Volume (voxels)", os.path.join(hist_folder, f"{name}_vol_filt_volume_hist.png"))
            save_histogram(vol_filt_surface_areas, vol_filt_surface_area_bins, "Volume-Filtered Particle Surface Area Distribution", "Surface Area (voxels)", os.path.join(hist_folder, f"{name}_vol_filt_surface_hist.png"))
            save_histogram(vol_filt_diameters, vol_filt_diameter_bins, "Volume-Filtered Particle Diameter Distribution", "Diameter (voxels)", os.path.join(hist_folder, f"{name}_vol_filt_diameter_hist.png"))

            # Save summary metrics
            save_summary_metrics(volumes, surface_areas, diameters, save_dir, run_tag, name, prefix="raw")
            save_summary_metrics(iqr_filt_volumes, iqr_filt_surface_areas, iqr_filt_diameters, save_dir, run_tag, name, prefix="iqr_filt")
            save_summary_metrics(vol_filt_volumes, vol_filt_surface_areas, vol_filt_diameters, save_dir, run_tag, name, prefix="vol_filt")

            print(f"Saved results for {name}.\n")