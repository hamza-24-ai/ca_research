<p align="center">
  <img src="assets/banner.svg" alt="Does Lower Precision Always Help? banner" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.14.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Colab"/>
  <img src="https://img.shields.io/badge/Course-Computer%20Architecture-1F3A5F" alt="Course"/>
</p>

# Does Lower Precision Always Help?
### A Cross-Architecture Study of CPU, GPU, and TPU Inference

**Author:** Muhammad Hamza, Computer Science, NUCES Faisalabad (hamzacs1028@gmail.com)

This project asks whether lowering numeric precision (FP16 or BF16 instead of FP32) speeds up model inference, and whether the gain depends on the model and the hardware. We benchmark three models on an Intel Xeon CPU, an NVIDIA Tesla T4 GPU, and a Google TPU v5e.

---

## Key Findings

| Finding | Result |
|---|---|
| FP16 on the T4 (DistilBERT, batch 32) | **4.7x faster** than FP32 |
| FP16 on the T4 (LSTM, batch 32) | **1.25x faster** than FP32 |
| BF16 on the TPU v5e | Never slower than FP32 |
| BF16 on the CPU (Broadwell) | **5 to 14x slower**, no native hardware support |
| Accuracy change, all cases | Within **0.2 percentage points** |

**Conclusion:** Lower precision does not always speed up inference. The gain depends on how much matrix math the model contains and whether the hardware supports the format.

---

## Hardware

| Device | Configuration | Peak used for comparison |
|---|---|---|
| CPU | Intel Xeon @ 2.20 GHz (Broadwell), 1 physical core | 70.4 GFLOP/s FP32 |
| GPU | NVIDIA Tesla T4, driver 580.82.07 | 8.14 TFLOPS FP32 |
| TPU | Google TPU v5e, one chip | 197 TFLOP/s BF16 |

## Models

| Model | Family | Parameters | Compute | Test data |
|---|---|---|---|---|
| MobileNetV2 | CNN | 3.5M | 0.60 GFLOPs / image | Imagenette2-320 (1,000 images, 10 classes) |
| DistilBERT | Transformer | 67M | 10.9 GFLOPs / sequence | SST-2 validation (872 sentences) |
| LSTM | Recurrent | 16.9M | 26 MFLOPs / window | WikiText-2 (10,000 windows) |

---

## Repository Structure

```
CA_Project/
├── make_roofline.ipynb        # Roofline plots for CPU, GPU, TPU
├── assets/banner.svg          # README banner
├── *_cpu.csv, *_gpu.csv, *_tpu.csv   # Raw measurements per device
│   ├── matmul_*.csv           # Matrix multiplication (n = 256 to 2048)
│   ├── bandwidth_*.csv        # Memory copy / scale-and-add bandwidth
│   ├── precision_*.csv        # FP32 vs FP16 vs BF16
│   ├── transfer_*.csv         # Host-to-device transfer time (GPU, TPU)
│   ├── distilbert_*.csv, lstm_*.csv, mobilenetv2_*.csv   # Inference results
│   └── lstm_train_*.csv       # LSTM training (2,000 steps)
├── Roofline_CPU.png, Roofline_GPU.png, Roofline_TPU.png
├── CPU_Broadwell_Core_Diagram.svg, GPU_Turing_SM_Diagram.svg
├── Research_Paper_CPU_GPU_TPU.pdf
└── Task*_Complete_Doc.pdf
```

---

## Reproduction Steps

### 1. Environment
Run on Google Colab (or Kaggle) with the following runtimes:
- **CPU:** Colab standard CPU runtime
- **GPU:** Runtime > Change runtime type > T4 GPU
- **TPU:** Runtime > Change runtime type > TPU v5e

### 2. Install dependencies
```bash
pip install torch==2.14.1 torchvision transformers datasets matplotlib pandas
# TPU only:
pip install torch_tpu==0.2.0
```

### 3. Set the seed
All experiments use seed `0`:
```python
import torch, random, numpy as np
torch.manual_seed(0); random.seed(0); np.random.seed(0)
```

### 4. Timing rules
- Warm-up: 3 untimed runs per configuration
- Timed runs: 100 for batch size 1 (for P95/P99), 10 for larger batches
- **GPU:** call `torch.cuda.synchronize()` before stopping the timer
- **TPU:** end each timed call with `.sum().item()` to force completion
- Only the forward pass is timed. Model loading and compilation are excluded.

### 5. Run the experiments
Run the notebooks or scripts for each device, saving output to the CSV files listed above. Each CSV corresponds to one device, one test, and one model.

### 6. Generate the Roofline plots
Open `make_roofline.ipynb` and run all cells. It reads the `matmul_*.csv` and `bandwidth_*.csv` files and writes `Roofline_*.png`.

---

## Model Weights and Data
- **MobileNetV2:** torchvision ImageNet-pretrained weights
- **DistilBERT:** Hugging Face checkpoint fine-tuned on SST-2
- **LSTM:** trained from scratch on WikiText-2; the trained weights are in `lstm_wikitext2.pt`
- **Imagenette:** https://github.com/fastai/imagenette

---

## Limitations
- End-to-end pipeline (preprocessing included) was not fully timed.
- Only the LSTM was trained on all three devices.
- The CPU is a shared Colab VM; hardware counters were not available.
- TPU FP32 peak is not published; the BF16 peak is used as a reference only.
- Speed tests used synthetic inputs.

---

## References
1. Intel, *Intel 64 and IA-32 Architectures Optimization Reference Manual*.
2. NVIDIA, *NVIDIA Turing GPU Architecture Whitepaper*, 2018.
3. NVIDIA, *NVIDIA T4 Tensor Core GPU Datasheet*, 2019.
4. Google Cloud, *Cloud TPU v5e documentation*.
5. N. P. Jouppi et al., "In-datacenter performance analysis of a tensor processing unit," ISCA 2017.
6. S. Williams, A. Waterman, D. Patterson, "Roofline," *Commun. ACM*, 2009.
7. M. Sandler et al., "MobileNetV2," CVPR 2018.
8. J. Howard, "Imagenette," fast.ai.
9. J. L. Hennessy and D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed., 2017.
10. V. Sanh et al., "DistilBERT," arXiv:1910.01108, 2019.
11. A. Wang et al., "GLUE," EMNLP BlackboxNLP, 2018.
12. S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 1997.

---

## License
Academic project. Datasets and pretrained weights are subject to their original licenses.
