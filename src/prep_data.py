import os
import shutil
import random

# Dynamically set project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define paths
classes_file = os.path.join(project_root, "data/classes.txt")
raw_data_path = os.path.join(project_root, "data/raw")
processed_data_path = os.path.join(project_root, "data/processed")


# Ensure classes.txt exists
if not os.path.exists(classes_file):
    print(f"Error: '{classes_file}' not found. Please run 'class_counter.py' via main.py first.")
    exit()

# Read classes.txt into a dictionary
with open(classes_file, "r") as file:
    class_details = dict(line.strip().split(": ") for line in file)

# Convert numeric values to integers where applicable
for key, value in class_details.items():
    if value.isdigit():
        class_details[key] = int(value)

# Ensure min_class_count exists
min_class_count = class_details.get("min_class_count")
if not min_class_count:
    print("Error: 'min_class_count' not found in classes.txt.")
    exit()

# Get the folder names directly from raw_data_path
# Retrieve class names from classes.txt instead of raw_data_path
num_classes = class_details.get("class_number")
class_folders = [class_details[f"class_{i}_name"] for i in range(1, num_classes + 1)]


# Prepare cohorts
cohorts = ["train", "test", "validate"]
split_ratios = {"train": 0.7, "test": 0.2, "validate": 0.1}



# Function to check if processed data matches expectations
def check_processed_data():
    print("Verifying processed data...")

    for cohort in cohorts:
        cohort_path = os.path.join(processed_data_path, cohort)
        
        # Check if cohort folders exist
        if not os.path.exists(cohort_path):
            print(f"❌ ERROR: {cohort} folder is missing.")
            return False

        for class_name in class_folders:
            class_path = os.path.join(cohort_path, class_name)
            
            # Check if class subfolders exist inside train/test/validate
            if not os.path.exists(class_path):
                print(f"❌ ERROR: {cohort}/{class_name} folder is missing.")
                return False

            # Count the actual number of image files
            actual_files = [
                f for f in os.listdir(class_path) 
                if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            actual_count = len(actual_files)

            # Calculate the expected number of files
            expected_count = int(min_class_count * split_ratios[cohort])

            # Debugging output
            print(f"🔍 Checking {cohort}/{class_name} → Expected: {expected_count}, Found: {actual_count}")

            # If the actual file count does not match, force re-run
            if actual_count != expected_count:
                print(f"❌ MISMATCH in {cohort}/{class_name}: Expected {expected_count}, Found {actual_count}.")
                return False

    print("✅ Processed data is complete and valid.")
    return True

# Check if processed data needs to be prepared
if check_processed_data():
    print("Data is already organized into train, test, and validate cohorts. Skipping prep_data.py.")
    exit()

# Create directories for cohorts and classes
for cohort in cohorts:
    cohort_path = os.path.join(processed_data_path, cohort)
    os.makedirs(cohort_path, exist_ok=True)
    for class_name in class_folders:
        os.makedirs(os.path.join(cohort_path, class_name), exist_ok=True)

# Copy files into cohorts
for class_name in class_folders:
    class_folder = os.path.join(raw_data_path, class_name)
    files = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
    
    # Shuffle and limit to min_class_count
    random.shuffle(files)
    files = files[:min_class_count]
    
    # Determine split counts
    train_count = int(min_class_count * split_ratios["train"])
    test_count = int(min_class_count * split_ratios["test"])
    validate_count = min_class_count - train_count - test_count
    
    # Distribute files
    split_counts = {"train": train_count, "test": test_count, "validate": validate_count}
    start_idx = 0
    
    for cohort in cohorts:
        cohort_path = os.path.join(processed_data_path, cohort, class_name)
        end_idx = start_idx + split_counts[cohort]
        for file_name in files[start_idx:end_idx]:
            src_file = os.path.join(class_folder, file_name)
            dst_file = os.path.join(cohort_path, file_name)
            shutil.copy2(src_file, dst_file)
        start_idx = end_idx

print("Data organized successfully into train, test, and validate cohorts!")