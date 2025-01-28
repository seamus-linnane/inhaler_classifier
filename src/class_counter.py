import os

# Define the relative path to the raw data folder
raw_data_path = "./data/raw/"

# Define the output file for class details
output_file = "./data/classes.txt"

# Check if the raw data folder exists
if not os.path.exists(raw_data_path):
    print(f"Error: The folder '{raw_data_path}' does not exist.")
    exit()

# List all subdirectories in the raw data folder
subfolders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]

# Initialize a dictionary to hold class details
class_details = {}
class_counts = []

# Populate the dictionary with class names and counts
for idx, folder in enumerate(subfolders, start=1):
    folder_path = os.path.join(raw_data_path, folder)
    item_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    class_details[f"class_{idx}_name"] = folder.lower()
    class_details[f"class_{idx}_count"] = item_count
    class_counts.append(item_count)

# Add the total number of classes
class_details["class_number"] = len(subfolders)

# Add the smallest class count
class_details["min_class_count"] = min(class_counts)

# Print the results
print(f"Number of classes: {class_details['class_number']}")
for i in range(1, class_details["class_number"] + 1):
    print(f"Class {i}: {class_details[f'class_{i}_name']} ({class_details[f'class_{i}_count']} items)")
print(f"Minimum class count: {class_details['min_class_count']}")

# Write the details to the classes.txt file
with open(output_file, "w") as file:
    file.write(f"class_number: {class_details['class_number']}\n")
    for i in range(1, class_details["class_number"] + 1):
        file.write(f"class_{i}_name: {class_details[f'class_{i}_name']}\n")
        file.write(f"class_{i}_count: {class_details[f'class_{i}_count']}\n")
    file.write(f"min_class_count: {class_details['min_class_count']}\n")

print(f"Class details saved to: {output_file}")