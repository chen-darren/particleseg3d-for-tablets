import os
import numpy as np
import pandas as pd
import tifffile as tiff

def calculate_metrics(gt_mask, pred_mask):
    intersection = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask | pred_mask)
    total_gt = np.sum(gt_mask)
    total_model = np.sum(pred_mask)
    true_positive = intersection
    false_positive = np.sum(pred_mask) - true_positive
    false_negative = np.sum(gt_mask) - true_positive
    true_negative = np.sum(~gt_mask & ~pred_mask)

    iou = intersection / union if union > 0 else np.nan
    dice = (2 * intersection) / (total_gt + total_model) if (total_gt + total_model) > 0 else np.nan
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else np.nan
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else np.nan
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else np.nan
    fpr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else np.nan
    fnr = false_negative / (false_negative + true_positive) if (false_negative + true_positive) > 0 else np.nan

    return iou, dice, accuracy, precision, recall, specificity, fpr, fnr

def process_image(img, gt_sem_path, gt_inst_path, pred_path, run_tag):
    gt_sem_files = sorted([f for f in os.listdir(os.path.join(gt_sem_path, img)) if f.endswith('.tiff')])
    # gt_inst_files = sorted([f for f in os.listdir(os.path.join(gt_inst_path, img)) if f.endswith('.tiff')])
    pred_files = sorted([f for f in os.listdir(os.path.join(pred_path, img)) if f.endswith('.tiff')])

    gt_sem_image = np.stack([tiff.imread(os.path.join(gt_sem_path, img, f)) for f in gt_sem_files])
    # gt_inst_image = np.stack([tiff.imread(os.path.join(gt_inst_path, img, f)) for f in gt_inst_files])
    pred_image = np.stack([tiff.imread(os.path.join(pred_path, img, f)) for f in pred_files])
    
    gt_sem_mask = gt_sem_image > 0
    pred_sem_mask = pred_image > 0
    
    iou, dice, accuracy, precision, recall, specificity, fpr, fnr = calculate_metrics(gt_sem_mask, pred_sem_mask)
    
    return [run_tag, img, iou, dice, accuracy, precision, recall, specificity, fpr, fnr]

def save_metrics(gt_sem_path, gt_inst_path, pred_path, results_path, run_tag, img_names):
    results = [process_image(img, gt_sem_path, gt_inst_path, pred_path, run_tag) for img in img_names]
    df = pd.DataFrame(results, columns=['run_tag', 'image_name', 'iou', 'dice', 'accuracy', 'precision', 'recall', 'specificity', 'fpr', 'fnr'])
    
    results_file = os.path.join(results_path, 'semantic_metrics.csv')
    
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        existing_df = existing_df.loc[:, ~existing_df.columns.str.contains('^Unnamed')]
        existing_df[['run_tag', 'image_name']] = existing_df[['run_tag', 'image_name']].astype(str)
        df[['run_tag', 'image_name']] = df[['run_tag', 'image_name']].astype(str)
        
        existing_df.set_index(['run_tag', 'image_name'], inplace=True)
        df.set_index(['run_tag', 'image_name'], inplace=True)
        
        existing_df.update(df)
        updated_df = pd.concat([existing_df, df[~df.index.isin(existing_df.index)]])
        updated_df.reset_index(inplace=True)
    else:
        updated_df = df
    
    updated_df.to_csv(results_file, index=False)
    return updated_df