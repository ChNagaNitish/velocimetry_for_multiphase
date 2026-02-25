#!/bin/bash
# 
#SBATCH -t 0-02:00:00 
#SBATCH -N 1 
#SBATCH --account=cavitation 
#SBATCH --partition=a30_normal_q 
#SBATCH --gres=gpu:1 
#SBATCH --mail-user=your-email@vt.edu 
#SBATCH --mail-type=ALL 
#SBATCH --job-name=velocimetry
 
# Loading required modules 
module purge 
module reset 
module load GCC/13.3.0 Python/3.12.3-GCCcore-13.3.0 
module load CUDA/12.6.0 

# Activate environment
source "$HOME/NAME-OF-THE-ENV/bin/activate" 

# ===============================================
# ALGORITHM EXECUTION EXAMPLES
# ===============================================
# Uncomment the algorithm you wish to run below. 
# Make sure to replace the placeholder paths with your actual video directory!

# 1. RAFT (Deep Learning Optical Flow)
# Highly accurate, leverages the requested A30 GPU automatically. Requires an optical flow .pth model weight.
python3 tracker.py --method raft --model weights/raft-sintel.pth --path /home/your-pid/experiment/48.avi --batch_size 4 --use_clahe --throat_loc 331 85

# 2. OpenPIV (Classical Particle Image Velocimetry)
# Operates strictly on the CPU. Automatically allocates worker pools matching your SLURM --cpus-per-task limit.
# Note: When running OpenPIV, you must uncomment the SBATCH command below to request CPU cores from ARC.
# #SBATCH --cpus-per-task=16
# python3 tracker.py --method openpiv --path /home/your-pid/experiment/48.avi --batch_size 16 --use_clahe --throat_loc 331 85

# 3. Farneback (Classical Optical Flow)
# Fast, baseline optical flow based on OpenCV's internal implementations.
# python3 tracker.py --method farneback --path /home/your-pid/experiment/48.avi --use_clahe --throat_loc 331 85
