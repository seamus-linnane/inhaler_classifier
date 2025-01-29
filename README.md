# Inhaler Type Identifier

An AI-powered project that uses deep learning to classify inhaler types from images. This project leverages transfer learning with pre-trained ResNet models for accurate and efficient image classification.

---

## 🚀 Features

- **Transfer Learning**: Utilizes ResNet-18, with options for ResNet-152, pre-trained on ImageNet.
- **Customizable Classes**: Dynamically detects class names and counts from `classes.txt`.
- **Device Compatibility**: Automatically uses MPS (for Apple Silicon), CUDA, or CPU for training.
- **Data Preprocessing**: 
  - Validates images before training, removing corrupted or non-standard files.
  - Standardizes images to **256x224 resolution** before training.
- **Error Handling**: 
  - Ensures missing datasets trigger appropriate preprocessing steps.
  - Prevents training with corrupt or missing data.
- **Metrics Visualization**: Generates plots for **training and validation loss/accuracy**.

---

## 📂 Directory Structure

```plaintext
inhaler_classifier/
├── data/
│   ├── raw/            # Original images before processing
│   ├── processed/      # Preprocessed datasets (train/test/validate)
│   │   ├── train/
│   │   ├── validate/
│   │   ├── test/
│   ├── classes.txt     # Auto-generated class details
├── models/             # Folder for saving trained models
├── src/                # Source code
│   ├── main.py         # Main execution pipeline
│   ├── class_counter.py# Counts classes and generates `classes.txt`
│   ├── prep_data.py    # Splits raw data into train/validate/test sets
│   ├── standardise_images.py # Standardizes images and removes corrupt files
│   ├── inhaler_finetune.py  # Model training and fine-tuning
├── README.md           # Project documentation
├── requirements.txt    # Project dependencies
```

## 🔧 Installation

Prerequisites
	•	Python 3.9+
	•	torch, torchvision, matplotlib, and other Python libraries listed in requirements.txt.

Setup
	1.	Clone the Repository:

      git clone https://github.com/seamus-linnane/inhaler_classifier.git
      cd inhaler_classifier

	2.	Install Dependencies:

      pip install -r requirements.txt

	3.	Ensure Your Dataset is Organized as Follows:  
   These folders should contain your images before processing.

      data/raw/  
      ├── named_class_1/  
      ├── named_class_2/  
         etc  


🔄 Workflow

1️⃣ Automatically Prepare the Dataset

Run main.py to execute the full pipeline, including dataset validation, standardization, and training:

python src/main.py

The script will:  
	•	Check if classes.txt exists (if not, generate it).
	•	Verify that data/raw/ contains valid images.
	•	Convert raw data into train/test/validate sets.
	•	Standardize images and remove corrupt files.
	•	Train the model using transfer learning.

2️⃣ Train the Model

Training is included in main.py, but you can also run fine-tuning separately:

python src/inhaler_finetune.py

This will:  
	•	Freeze pre-trained ResNet layers.
	•	Train the classifier head for 6 epochs.
	•	Fine-tune the entire network for 20 additional epochs.
	•	Save the trained model in models/.

📊 Visualization

Training & Validation Loss

Training & Validation Accuracy

🏆 Results

- **Dataset Used**: Images were scraped from the web, categorized into `ellipta` and `mdi` classes.
- **Final Accuracy**: 
  - **Training Accuracy**: ~94.5% (based on final epoch)
  - **Validation Accuracy**: ~93% (based on final epoch)
  - Training and validation curves show **good generalization with minimal overfitting**.
- **Performance on Unseen Data**: 
  - **Not yet tested**. Next steps include evaluating the model on completely unseen inhaler images to measure real-world performance.

🛠️ Troubleshooting

Common Issues & Fixes

Issue	Solution
PIL.UnidentifiedImageError	Run python src/standardise_images.py to clean dataset
No images found in processed	Ensure prep_data.py has run correctly
MPS TypeError (float64)	Ensure inhaler_finetune.py uses .float() instead of .double()

📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

❤️ Acknowledgments
	•	PyTorch: For the deep learning framework.
	•	ImageNet: For pre-trained ResNet weights.
	•	Apple MPS Acceleration: Optimized for Apple Silicon.

   ## ✅ To-Do
- [ ] Evaluate model performance on unseen inhaler images.
- [ ] Expand dataset with more inhaler types.
- [ ] Implement data augmentation for robustness.

📌 Next Steps   
	•	Evaluate on additional test datasets.

🔥 Want to Contribute?

Pull requests are welcome! For major changes, please open an issue first to discuss. 🚀