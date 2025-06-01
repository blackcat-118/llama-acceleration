# Llama 3B Acceleration

This project focuses on accelerating the inference speed of the LLaMA 3B Instruct small language model (SLM) while maintaining acceptable accuracy, defined as a perplexity (PPL) ≤ 11.5. We explore and implement optimization techniques aimed at reducing latency and improving throughput without significantly sacrificing model performance. 

## 📦 Features

- Model Distillation
- HQQ
- Lora Fine-tuning

## 🚀 Getting Started

### Prerequisites

Python version: 3.10.12

```bash
# Install Environment
pip install -r requirements.txt
```

### Run Model Distillation
```bash
python3 distil_logits.py
```

### Fine-tuned Quantized Model and Evaluation
```bash
python3 result.py
```
