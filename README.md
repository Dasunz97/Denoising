# U-net based Self-Supervised Denoiser for Low Dose 4D-STEM Data SrTiO<sub>3</sub>

This repository provides a **self-supervised denoising pipeline** for low-dose 4D-STEM images of SrTiO<sub>3</sub> using a U-Net-based deep learning architecture with iterative refinement. The approach leverages the blind-spot/neighbours method for denoising, where the network predicts each pixel from its spatial neighbors, enabling training without clean ground truth. All implementations are in **PyTorch** and **PyTorch Lightning**.

---

<p align="center"><img width="2933" height="866" alt="denoised1" src="https://github.com/user-attachments/assets/6f9e6849-0e46-4d42-b0ab-1c169a5d4411" /><br/><strong>Denoising Comparison</strong></p>


*Left: Raw low-dose input. Center: Denoised output after 10 cycles. Right: High-dose (reference, not used for training).*

---

## Key Features

- **U-Net Architecture:**  
  A U-Net style convolutional neural network processes each patch, taking as input the eight neighboring patches (blind-spot), and predicts the center.

- **Iterative Refinement:**  
  The denoising process is performed over multiple cycles (default: 10). After each cycle, predictions are fed back into the training set, gradually improving denoising performance.

- **Self-Supervised (Blind-Spot) Learning:**  
  The model is trained using only *noisy* low-dose data, leveraging spatial neighbours. High-dose data is used only for qualitative comparison, not for training or validation.

- **4D-STEM SrTiO<sub>3</sub> Data:**  
  The script is tailored for 4D scanning transmission electron microscopy (4D-STEM) data of strontium titanate (SrTiO<sub>3</sub>), but can be adapted to other similar datasets.

- **PyTorch Lightning Framework:**  
  Training and evaluation are managed with PyTorch Lightning for ease of experimentation and reproducibility.

---

## Method Overview

The denoising is achieved by:

1. **Blind-Spot Training:**  
   For each patch, the network receives only the eight surrounding neighbour patches (excluding the center), ensuring no direct access to the noisy target.

2. **Iterative Self-Refinement:**  
   - The network is trained for a number of epochs.
   - The denoised output is combined with the raw data and fed again as input for another training cycle.
   - This process is repeated (default: 10 cycles) to refine the denoised result.

3. **Loss Functions:**  
   - Initial epochs use z-normalized MSE loss for warmup.
   - After a warmup, Poisson negative log-likelihood loss is used, with auxiliary terms for mean and sum constraints, reflecting the statistics of electron-counting data.

---

## Usage

**Requirements:**
- PyTorch
- PyTorch Lightning
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install torch pytorch-lightning numpy matplotlib
```

**Edit Paths:**
```python
LOW_PATH  = '/path/to/low_dose.npy'    # Noisy input data
HIGH_PATH = '/path/to/high_dose.npy'   # High-dose, for comparison only
OUT_DIR   = '/path/to/output_dir'      # Output directory
```
**Run the script:**
```bash
python denoise.py
```

**Outputs:**
- `denoised_cycle_{c}.npy`: Denoised data after each cycle
- `SrTiO3_denoised.npy`: Final denoised cube
- `unet_denoiser.pt`: Trained model weights
- `denoised.png`: Comparison plot (shown above)

---

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Unsupervised deep denoising for four-dimensional scanning transmission electron microscopy](https://doi.org/10.1038/s41524-024-01428-x)

---
