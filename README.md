# YogaSync - Detailed Analysis and Classification of Yoga Poses 

This project aims to classify and correct yoga postures using deep learning techniques. We utilize an ensemble of pretrained CNN models (Xception, MobileNetV2, and InceptionV3) enhanced with soft attention mechanisms to improve performance and interpretability.

## 📊 Results

An ablation study comparing individual and ensemble models on the Yoga Posture Dataset showed that our **Majority Voting based Ensemble with Soft Attention** achieved the best performance:

| Model | Accuracy (%) | Precision | Recall | F1-Score |
|-------|--------------|-----------|--------|----------|
| Xception | 68.00 | 0.68 | 0.60 | 0.60 |
| MobileNetV2 | 61.56 | 0.61 | 0.58 | 0.59 |
| InceptionV3 | 59.07 | 0.59 | 0.56 | 0.53 |
| Product-Rule Ensemble | 68.00 | 0.68 | 0.65 | 0.67 |
| Weighted Average Ensemble | 69.75 | 0.69 | 0.68 | 0.68 |
| **Majority Voting Ensemble** | **71.00** | **0.71** | **0.73** | **0.69** |

## 🧠 Model Architecture

- **Base Models**: Xception, MobileNetV2, InceptionV3  
- **Attention Mechanism**: Soft Attention Layer for each base model  
- **Ensemble Techniques**: Product Rule, Weighted Average, and Majority Voting  

## 🗂 Dataset

We used a publicly available Yoga Posture dataset consisting of labeled images across various poses. If you want to use the same dataset, you can find it in the following link:
(https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset)

## 🚀 Setup Instructions
# 1. Clone the repository:

git clone https://github.com/your-username/yoga-pose-detection.git
cd yoga-pose-detection
# 2. Install dependencies:

pip install -r requirements.txt

#(Optional) If using Git LFS:

git lfs install
git lfs pull



## 🙋‍♀️ Acknowledgements
Special thanks to the open-source community, TensorFlow/Keras contributors, and the creators of the Yoga Posture Dataset.



