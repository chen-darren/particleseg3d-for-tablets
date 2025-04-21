# particleseg3d-for-tablets
Darren, last edit 04/21/2025.

This repository includes our retrained nnU-Net models (via transfer learning), newly developed trainer classes to support transfer learning, an optimized inference pipeline, ParticleSeg3D source code [1] (with optimizations), data loading scripts [1] (with minor revisions), and comprehensive documentation for access/use. Additionally, it contains scripts for quantitative analysis, including segmentation metrics and particle size distribution calculations. This project was developed as part of the University of Connecticut (UConn) Senior Design program in the Department of Biomedical Engineering for the 2024–2025 academic year by Darren Chen of Team 15.

# Abstract
Variability in particle composition within pharmaceutical tablets can significantly impact drug release and therapeutic efficacy. Ensuring a uniform particle size distribution is critical for consistent dosage delivery. Traditional quality control methods often rely on destructive laboratory techniques that cannot be applied to intact tablets. To address these limitations, machine learning has been applied to tablet imaging. However, existing methods typically either analyze only a single representative slice or treat 3D volumetric data as stacks of 2D images. Both approaches overlook crucial spatial relationships between particles—reducing segmentation accuracy and compromising the reliability of particle characterization. This project addresses that limitation by applying ParticleSeg3D, a machine learning–based 3D instance segmentation pipeline, to a novel use case: volumetric particle analysis in pharmaceutical tablets. We optimized the pipeline’s preprocessing, postprocessing, and features, and applied transfer learning to adapt it for analyzing 3D X-ray microscopy tablet images. Using tablets supplied by our sponsor, DigiM, we evaluated the optimized pipeline across several key metrics. Our pipeline achieved a mean Dice coefficient of 0.9041, mean recall of 0.9802, mean specificity of 0.9843, and particle count error of 6.33%, demonstrating strong segmentation performance and quantitative reliability. Although precision (0.8390) was lower than other metrics, the model consistently captured both particle shape and volume distributions. To our knowledge, this represents one of the first applications of 3D instance segmentation for pharmaceutical tablet analysis. By enabling non-destructive, automated, and high-throughput particle detection, ParticleSeg3D offers a valuable tool for enhancing quality control and providing deeper insight into tablet microstructure.

# Project Breakdown
## Data Conversion
- Go to `README.me` in the `data_conversion` folder.
## Metrics for Quantitative Analysis
- Go to `README.me` in the `metrics` folder.
## ParticleSeg3D Pipeline Source Code
- Go to `README.me` in the `particleseg3d_source` folder.
## Virtual Environment Setup
- Go to `README.me` in the `setup` folder.
## Trainers for Transfer Learning of nnU-Net
- Go to `README.me` in the `trainers` folder.
## Miscellaneous Useful Functions
- Go to `README.me` in the `utils` folder.

# References
This work builds heavily upon and is derived from the original implementations of ParticleSeg3D [1] and nnU-Net [2].
- [1] ParticleSeg3D:
  - K. Gotkowski, S. Gupta, J. R. A. Godinho, C. G. S. Tochtrop, K. H. Maier-Hein, and F. Isensee, “ParticleSeg3D: A Scalable Out-of-the-Box Deep Learning Segmentation Solution for Individual Particle Characterization from Micro CT Images in Mineral Processing and Recycling,” Dec. 14, 2023, arXiv: arXiv:2301.13319. doi: 10.48550/arXiv.2301.13319.
  - Code: https://github.com/MIC-DKFZ/ParticleSeg3D
- [2] nnU-Net:
  - F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” Nat Methods, vol. 18, no. 2, pp. 203–211, Feb. 2021, doi: 10.1038/s41592-020-01008-z.
  - Code (ParticleSeg3D branch): https://github.com/MIC-DKFZ/nnUNet/tree/ParticleSeg3D
