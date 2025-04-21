# Weights  
Darren, last edit 04/21/2025.

This directory contains trained model weights for all five folds of the transfer learning process using the original Fold 0 weights from ParticleSeg3D. The five-fold cross-validation split was manually implemented, where Aug. 1–5 correspond to validation splits for Folds 0–4 respectively.

# Training Details
- **Trainer Used**: `nnUNetTrainerV2_ParticleSeg3D_DarrenSGD_CUDAErrorSkip`  
  This is a modified trainer that:
  - Builds off `nnUNetTrainerV2_ParticleSeg3D_DarrenSGD`.
  - Safely skips optimizer steps where gradients are NaN or zero (which rarely occurs) instead of terminating training.

- **Augmented Data**:  
  All models were trained via five-fold cross-validation using only augmented images.

- **Ground Truth**:  
  Ground truth instance and border-core segmentations were generated from binary masks using: `\particleseg3d_source\train\preprocess.py`.

# Final vs. Best Model
Both the "best" (lowest validation loss) and "final" (last epoch) model checkpoints are saved, but **the final model is generally recommended**.

- According to the maintainer of the official [nnU-Net GitHub repository](https://github.com/MIC-DKFZ/nnUNet), the final model often outperforms the best model in practice.
- The final models were used in the optimized ParticleSeg3D inference pipeline for this project.
