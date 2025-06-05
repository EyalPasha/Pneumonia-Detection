# PNEUMONIA DETECTION FROM CHEST X-RAYS USING SPECIAL ML/DL APPROACHES

![image](https://github.com/EyalPasha/ML-Basics/assets/51478907/519c5f18-5345-4f1c-b498-22f415427703)


The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.

Link: https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5



Content:
The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.


Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Data: https://data.mendeley.com/datasets/rscbjbr9sj/2



Absolutely — here's the cleaned-up version of your README without the self-promotional tone. I kept everything else intact and professional:

---

## 🔑 Project Highlights

| Stage                                                                 | What was done                                                                                                                                         | Purpose                                                    |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Data acquisition & EDA**                                            | Pulled the *Chest X-Ray Images (Pneumonia)* dataset (5,863 JPGs), split into train/val/test, verified class balance & image quality using matplotlib. | Prepares reliable input data for modeling.                 |
| **Baseline CNN**                                                      | Built a 3-layer Conv→ReLU→MaxPool network, trained from scratch with `Adam`, early-stopping & image rescaling.                                        | Establishes a performance baseline.                        |
| **Transfer Learning**                                                 | • **VGG-16** (frozen) + custom classifier head                                                                                                        |                                                            |
| • **DenseNet-121** fine-tuned on the last 60 % of layers              | Explores performance gains using pretrained models.                                                                                                   |                                                            |
| **Metric tracking**                                                   | Collected `accuracy`, `loss`, **precision / recall / F1**; plotted learning curves and tracked performance.                                           | Enables evaluation and comparison of model performance.    |
| **Embedding visualisation**                                           | Extracted penultimate-layer embeddings → **t-SNE** 2-D plot → **k-NN** classifier check.                                                              | Assesses how the model separates classes in feature space. |
| **Results**                                                           | *Binary (Normal vs Pneumonia)*: **89 % test accuracy**                                                                                                |                                                            |
| *Three-class (Normal / Bacterial / Viral)*: work-in-progress (≈ 33 %) | Provides benchmarks for different classification settings.                                                                                            |                                                            |

---

## 📊 Key Findings

| Model                            | Classes | Test Accuracy | Precision | Recall | F1   |
| -------------------------------- | ------- | ------------- | --------- | ------ | ---- |
| Baseline CNN                     | 2       | 0.84          | 0.85      | 0.82   | 0.83 |
| VGG-16 (frozen)                  | 2       | **0.89**      | 0.90      | 0.88   | 0.89 |
| DenseNet-121 (partial fine-tune) | 3       | 0.33          | 0.33      | 0.32   | 0.32 |

Full `classification_report`s and confusion matrices appear in the notebook.

---

## 🛠️ Tech Stack

* **Python 3.10** • TensorFlow/Keras 2.15 • scikit-learn • matplotlib • tqdm
* **t-SNE** & **k-NN** for embedding analysis
* Runs on **Google Colab** (free Tesla T4 GPU) or any CUDA-enabled machine

---

## 🗺️ Next Steps

* 🔭 **Grad-CAM** heat-maps for explainability
* 🎯 Hyper-parameter sweeps (LR, decay, optimiser)
* 🥇 Ensemble VGG + DenseNet for ≥ 92 % binary accuracy
* 🧪 Self-supervised pre-training on unlabeled CXR images

---

## 📖 Dataset & References

* *Chest X-Ray Images (Pneumonia)* – Kermany et al., 2018 (Kaggle)
* K. He et al., “Deep Residual Learning for Image Recognition”, CVPR 2016
* F. Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions”, CVPR 2017
