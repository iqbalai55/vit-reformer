# Vision Reformer (ViR)

This repository contains an experimental implementation of a **Vision Transformer variant that incorporates Reformer-inspired attention mechanisms** for facial recognition. The project explores whether ideas from the Reformer architecture, particularly **Locality Sensitive Hashing (LSH) attention**, can improve computational efficiency when applied to Vision Transformer pipelines. In theory, Reformer reduces the complexity of self-attention from *O(n²)* to *O(n log n)*. However, the hashing and sorting operations used in LSH attention introduce additional computational overhead, which only becomes beneficial when the number of tokens is sufficiently large.

In typical Vision Transformer configurations, token counts are relatively small. For example, an image of **224×224 pixels with a patch size of 16×16** produces **196 tokens plus a CLS token**, meaning the overhead from the original LSH implementation may reduce its practical efficiency. To address this issue, this project implements a **simplified LSH attention module** designed to reduce unnecessary overhead while preserving the locality-based attention mechanism. The resulting architecture is referred to as **Vision Reformer (ViR)**.

The experiment uses a subset of the **Indonesian Muslim Student Face Dataset (IMSFD)** consisting of **10 classes** for multi-class facial recognition. Images are resized to 224×224 and processed through patch embeddings before being passed into transformer layers that replace standard self-attention with the modified LSH attention module. The implementation was developed using **Python and PyTorch**, with training performed on AWS and evaluation conducted on Google Colab. Model performance is evaluated using accuracy, precision, recall, and F1-score, along with computational metrics such as training time and GPU memory usage.



# Experiment Setup

The Vision Reformer (ViR) model is compared with a **standard Vision Transformer (ViT)** under the same experimental configuration. Images are resized to **224×224** with a **patch size of 16×16**, producing 196 tokens and one CLS token. Two learning rates were tested during training, **0.001 and 0.0001**, to observe their effect on model performance. Evaluation metrics include **accuracy, precision, recall, and F1-score**, while additional analysis measures **training time and GPU memory usage** to examine computational efficiency.



# Results

### Model Performance

| Model | Learning Rate | Accuracy | Precision | Recall | F1-score |
| ----- | ------------- | -------- | --------- | ------ | -------- |
| ViR   | 0.0001        | 0.9463   | 0.9589    | 0.9463 | 0.9477   |
| ViT   | 0.0001        | 0.9411   | –         | –      | –        |

### Training Time

| Model | Training Time (seconds) |
| ----- | ----------------------- |
| ViR   | 41.076                  |
| ViT   | 82.368                  |

### GPU Memory Usage

| Batch Size | ViR Memory Saving vs ViT |
| ---------- | ------------------------ |
| 16         | 78 MB                    |
| 32         | 39 MB                    |

The results show that the **Vision Reformer model achieves comparable classification performance while improving training efficiency**, reducing training time by nearly half and demonstrating slightly lower GPU memory consumption.



# Disclaimer

The experiments presented in this repository were conducted **only on a subset of the IMSFD dataset**, and the results may not necessarily generalize to other datasets or tasks. Therefore, this implementation should be viewed primarily as an **experimental exploration of efficient attention mechanisms in Vision Transformer architectures**.


# Example Usage

```python
from model import ViR

model = ViR(
    img_size=224,
    patch_size=16,
    num_classes=10,
    dim=768,
    depth=12,
    heads=8
)

output = model(images)
```

For feature extraction:

```python
features = model.extract_features(images)
```

