import os

# Define the project root dynamically
project_root = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the project root
raw_data_path = os.path.join(project_root, "data/raw")
processed_data_path = os.path.join(project_root, "data/processed")
classes_file = os.path.join(project_root, "data/classes.txt")
src_path = os.path.join(project_root, "src")

def run_class_counter():
    """Ensure classes.txt is up to date."""
    if not os.path.exists(classes_file):
        print("classes.txt not found. Running class_counter.py to generate it...")
        os.system(f"python {os.path.join(src_path, 'class_counter.py')}")
    else:
        print("Validating classes.txt...")
        os.system(f"python {os.path.join(src_path, 'class_counter.py')} --validate")

def run_prep_data():
    """Ensure processed data is ready."""
    if not (os.path.exists(os.path.join(processed_data_path, "train")) and
            os.path.exists(os.path.join(processed_data_path, "validate")) and
            os.path.exists(os.path.join(processed_data_path, "test"))):
        print("Processed data not found. Running prep_data.py...")
        os.system(f"python {os.path.join(src_path, 'prep_data.py')}")
    else:
        print("Processed data already exists. Skipping prep_data.py.")

def run_standardise_images():
    """Ensure images in processed data are standardised."""
    print("Checking image standardisation...")
    os.system(f"python {os.path.join(src_path, 'standardise_images.py')}")

def run_finetune():
    """Run the model fine-tuning script."""
    print("Starting fine-tuning...")
    os.system(f"python {os.path.join(src_path, 'inhaler_finetune.py')}")

if __name__ == "__main__":
    print("Starting the project pipeline...")
    run_class_counter()
    run_prep_data()
    run_standardise_images()
    run_finetune()
    print("Pipeline completed successfully!")