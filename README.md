# 🧠 Custom Image Classification using ResNet (PyTorch)

This project builds a **custom image classification system** using a manually implemented **ResNet architecture** (with Residual Blocks).  
The dataset consists of user-defined categories, and the model is designed to classify real-world images—including live webcam frames—into one of the trained classes.

---

## 📌 Project Overview

The project implements the full pipeline required to train a deep neural network:

- Loading and structuring dataset images into memory  
- Mapping string class names into numerical labels  
- Applying image preprocessing and normalization  
- Building a custom **ResNet** model using residual blocks  
- Training, tuning, and validating the model  
- Testing final accuracy on a held-out test set  
- Performing **real-time predictions** using webcam-captured frames  

This setup is ideal for building personalized multi-class image classification systems.

---

## 📂 Dataset Structure

The dataset is organized into multiple folders, each representing a class, e.g.:

The script dynamically:

- Reads all images inside each folder  
- Assigns labels based on folder names  
- Maps class names to numerical IDs  
- Stores everything into a unified dataset format  

---

## 🔄 Preprocessing

Each image passes through a transform pipeline:

- Convert numpy → PIL  
- Resize to **64×64**  
- Convert to tensor  
- Normalize using mean/std = 0.5 per channel  

This ensures consistent formatting and stable gradients during training.

Dataset is then split into:

- **Train Set** (70%)
- **Dev/Validation Set** (15%)
- **Test Set** (15%)

---

## 🧱 Model Architecture — Custom ResNet

The project implements:

### **ResidualBlock**
- Two 3×3 convolution layers  
- Batch normalization  
- Identity shortcut  
- Optional projection shortcut (when shape mismatch)  
- Final ReLU  

### **ResNet Backbone**
- Initial 7×7 convolution  
- Max pooling  
- Residual Layer 1 (64 → 64)  
- Residual Layer 2 (64 → 128)  
- Adaptive average pooling  
- Fully connected output layer  

The model is configured to support **10 output classes** (can be changed as needed).

---

## 🏋️ Training Details

- **Epochs:** 15  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam (with weight decay)  
- **Device:** MPS (Mac GPU) or CPU  
- **Batch Size:** 64  

Training loop logs:

- Average training loss  
- Validation accuracy at the end of every epoch  

A line plot of loss over epochs is also generated.

---

## 📈 Model Evaluation

The model is evaluated on:

### ✔️ Validation (Dev) Set  
Used for tuning and checking overfitting.

### ✔️ Test Set  
Final evaluation on unseen images.

The script computes final accuracy and prints the result.

---

## 📹 Real-Time Webcam Prediction

The project includes a **live prediction pipeline**:

1. Capture 2 frames per second  
2. Save them to a temporary folder  
3. Apply the same preprocessing transform  
4. Pass each frame through the trained ResNet model  
5. Print predicted class label  
6. Clean up all temporary files  

This allows creating your own:

- Gesture classifier  
- Object recognizer  
- Real-time AI assistant  

---

## 💾 Saving the Model

Trained weights are saved using:

This enables easy reuse for:

- Deployment  
- Fine-tuning  
- Building a GUI or API interface  

---

## 🛠️ Technologies Used

- **PyTorch**
- **OpenCV**
- **Torchvision**
- **Scikit-Learn**
- **NumPy**
- **Seaborn** (visualization)

---

## 🚀 Future Enhancements

- Add deeper ResNet layers (ResNet-18/34/50 style)  
- Use data augmentation for improved generalization  
- Add confusion matrix and per-class accuracy  
- Build a Streamlit/Gradio UI  
- Implement Grad-CAM for interpretability  

---

## 🙌 Acknowledgements

- PyTorch community  
- OpenCV contributors  
- Inspiration from original ResNet paper  

---

If you want, I can generate:

✅ A cleaner GitHub-style README  
✅ Folder structure documentation  
✅ A downloadable `.md` version with badges and screenshots  
