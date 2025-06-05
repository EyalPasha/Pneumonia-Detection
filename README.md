# Pneumonia Detection from Chest X-Rays Using Special Approaches 🫁🧠

![image](https://github.com/EyalPasha/ML-Basics/assets/51478907/519c5f18-5345-4f1c-b498-22f415427703)

This project detects pneumonia in chest X-ray images using convolutional neural networks (CNNs) and transfer learning approaches like VGG-16 and DenseNet-121. It includes a full pipeline from data preprocessing to evaluation and interpretability with t-SNE visualizations.

---

## 📂 Dataset

* **Source:** Guangzhou Women and Children’s Medical Center
* **Categories:** Pneumonia / Normal
* **Format:** 5,863 anterior-posterior X-ray images (JPEG)
* **Structure:** Images are organized into `train`, `val`, and `test` folders

Dataset:

* Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 🧠 Techniques Used

| Technique                | Description                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------- |
| **CNN from Scratch**     | Custom Conv→ReLU→MaxPool architecture trained on raw data                             |
| **Transfer Learning**    | VGG-16 (frozen layers) and DenseNet-121 (partially fine-tuned) pretrained on ImageNet |
| **Data Augmentation**    | Real-time augmentation using `ImageDataGenerator` (rotation, zoom, flips, shifts)     |
| **t-SNE** Embedding      | Visual inspection of model representations                                            |
| **k-NN** Classifier      | Sanity check on learned feature embeddings                                            |
| **Model Saving/Loading** | Reusable architecture using `Model.save()` and `.h5` serialization                    |
| **Callbacks**            | EarlyStopping for generalization control                                              |

---

## 📊 Results Summary

| Model                            | Classes | Test Accuracy | Precision | Recall | F1   |
| -------------------------------- | ------- | ------------- | --------- | ------ | ---- |
| Baseline CNN                     | 2       | 0.84          | 0.85      | 0.82   | 0.83 |
| VGG-16 (frozen)                  | 2       | **0.89**      | 0.90      | 0.88   | 0.89 |
| DenseNet-121 (partial fine-tune) | 3       | 0.33          | 0.33      | 0.32   | 0.32 |

Training curves and classification reports are available in the notebook.

---

## 🧪 Libraries & Tools

* **TensorFlow / Keras** – deep learning models, transfer learning
* **scikit-learn** – t-SNE, k-NN, metrics
* **matplotlib** – plots and visualizations
* **tqdm** – progress bars
* **NumPy** – matrix operations
* **Google Colab** – runtime environment

---

## 📈 Visualizations

* Training/Validation Loss and Accuracy plots
* t-SNE 2D embedding projections
* Confusion matrices and classification reports

---


## 🧭 Future Work

* 🔬 Add Grad-CAM visualizations for explainability
* 🔁 Explore ensembling VGG + DenseNet
* 🧪 Run hyperparameter sweeps (optimizers, learning rates)
* 📚 Pre-train on unlabeled CXR images using self-supervision

---

## 📖 References

* Kermany et al., “Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning”, Cell 2018 ([link](https://www.cell.com/cell/fulltext/S0092-8674%2818%2930154-5))
* He et al., “Deep Residual Learning for Image Recognition”, CVPR 2016
* Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions”, CVPR 2017
