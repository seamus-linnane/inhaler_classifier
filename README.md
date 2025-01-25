
# Inhaler Type Identifier

An AI-powered project that uses deep learning to classify inhaler types from images. This project leverages transfer learning with pre-trained ResNet models for accurate and efficient image classification.

---

## Features

- **Transfer Learning**: Utilizes ResNet-18, with options for ResNet-152, pre-trained on ImageNet.
- **Customizable Classes**: Adapts to datasets with user-defined class counts.
- **Device Compatibility**: Automatically uses MPS (for Apple Silicon), CUDA, or CPU for training.
- **Data Augmentation**: Applies transformations like resizing, cropping, and normalization for robust training.
- **Metrics Visualization**: Generates plots for training and validation losses and accuracies.

---

## Directory Structure

```plaintext
inhaler-type-identifier/
├── data/
│   ├── train/          # Training dataset
│   ├── validate/       # Validation dataset
├── models/             # Folder for saving trained models
├── README.md           # Project documentation
├── main.py             # Main script for training and evaluation
└── requirements.txt    # Project dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- `torch`, `torchvision`, `matplotlib`, and other Python libraries listed in `requirements.txt`.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/seamus-linnane/inhaler_classifier.git
   cd inhaler-type-identifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the datasets are organized as follows:
   ```plaintext
   inhaler_2_train/
   ├── train/
   │   ├── Class1/
   │   ├── Class2/
   ├── validate/
   │   ├── Class1/
   │   ├── Class2/
   ```

---

## How to Run

1. **Train the Model**:
   - Edit `train_path` and `val_path` in `main.py` to point to your dataset locations.
   - Run the script:
     ```bash
     python main.py
     ```

2. **Fine-Tune the Entire Network**:
   - After training the last layer, the entire network is fine-tuned for more epochs for better accuracy.

3. **Save the Model**:
   - The trained model weights will be saved in the `models/` directory.

---

## Visualization

- Training and validation losses:
  ![Loss Plot](example_loss_plot.png)

- Training and validation accuracies:
  ![Accuracy Plot](example_accuracy_plot.png)

---

## Results

- **Dataset**: Describe  dataset here.
- **Accuracy**: Document accuracy achieved during testing.
- **Inference**: Discuss performance on unseen images.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- **PyTorch**: For the deep learning framework.
- **ImageNet**: For pre-trained model weights.
"""