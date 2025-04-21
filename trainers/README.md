# Trainers for Transfer Learning
Darren, last edit 04/21/2025.

This directory contains custom nnU-Net trainer classes tailored for transfer learning for tablet images. The trainers are categorized based on the optimizer used: **SGD** or **Adam**.

# SGD-Based Trainers
These trainers use Stochastic Gradient Descent (SGD) with customized learning rate settings and checkpointing behavior.

### `nnUNetTrainerV2_ParticleSeg3D_DarrenSGD`
- Inherits from `nnUNetTrainerV2_ParticleSeg3D` (original ParticleSeg3D trainer).

- Modifications:
  - Input directories are revised to fit the current transfer learning setup.
  - Checkpoints are saved every **5 epochs** (instead of the default).
  - Initial learning rate reduced from `0.01` to `0.001` (1/10 of the original).

### `nnUNetTrainerV2_ParticleSeg3D_DarrenSGD_CUDAErrorSkip`
- Inherits from `nnUNetTrainerV2_ParticleSeg3D_DarrenSGD`.

- Adds a safety mechanism to **skip optimizer steps** if the gradients contain `inf` or `NaN`.
  - Prevents rare CUDA errors from stopping training.

## Adam-Based Trainers
These trainers leverage the Adam optimizer with learning rate scheduling.

### `nnUNetTrainerV2_ParticleSeg3D_DarrenAdam`
- Analogous to the SGD version, but uses Adam optimizer.

- Modifications:
  - Initial learning rate reduced from `3e-4` to `3e-5` (1/10 of the original).
  - Implements a **ReduceLROnPlateau** scheduler, consistent with legacy nnU-Net implementations that used Adam.

### `nnUNetTrainerV2_ParticleSeg3D_DarrenAdam_CUDAErrorSkip`
- Inherits from `nnUNetTrainerV2_ParticleSeg3D_DarrenAdam`.

- Also implements the optimizer step skipping logic for handling rare gradient anomalies (e.g., `inf`, `NaN`).
