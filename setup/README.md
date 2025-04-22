# Virtual Environment Setup
Darren, last edit 04/22/2025.

This directory contains resources and instructions to help you set up the environment necessary to run **ParticleSeg3D**, which supports inference on both **Windows** and **Linux** systems. However, **all training tasks related to nnU-Net must be performed on Linux**.

# Summary
- `conda_env/`  
  Contains setup instructions for the required Conda environment.

- `linux_setup/`  
  Contains resources and documentation for setting up a Linux environment via WSL 2 on a Windows machine.

# `conda_env/`
This folder includes:

- `conda_env_source_notes.txt`  
  A text file with:
  - A complete list of required Python libraries and dependencies for ParticleSeg3D.
  - Copy-and-paste-ready commands to create and activate the Conda environment.

You can use this file to quickly replicate the correct environment by running the listed commands in your terminal.

# `linux_setup/`
This folder includes:

- `linux_conda_setup.txt`  
  Step-by-step notes for setting up a Linux environment on Windows using WSL 2.  
  Covers:
  - Enabling WSL and installing Ubuntu.
  - Installing Anaconda and creating the Conda environment (see `conda_env_notes.txt`).
  - Handling GeodisTK installation issues.
  - Optimizing `.wslconfig` for memory and swap usage to reduce SSD wear.
  - Adjusting Linux swappiness and installing monitoring tools (`htop`, `iotop`).
  - Fixing potential SSL/TLS errors when using `Invoke-WebRequest`.

- `reclaim_vhd_space.txt`  
  A collection of terminal commands to help you **reclaim VHD disk space** for WSL if your virtual disk grows too large.

- PDFs within `linux_setup/`  
  A set of reference documents downloaded from various websites that explain:
  - Additional WSL configuration steps.
  - Linux terminal basics.
  - Common troubleshooting and tips for WSL 2 performance.

# Notes
- You can perform **inference** with ParticleSeg3D on either Windows or Linux.

- **Training** with nnU-Net must be done in a **Linux environment** due to framework and performance limitations.

- These setup resources are tailored for Windows users looking to create a compatible Linux environment using WSL 2.


