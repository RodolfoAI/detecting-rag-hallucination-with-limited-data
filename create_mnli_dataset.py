from datasets import load_dataset, DatasetDict
from huggingface_hub import login

# 1. Authenticate with Hugging Face Hub (REQUIRED for push_to_hub)
login() 
# Uncomment and run this if you haven't logged in recently.

# 2. Load the entire DatasetDict
print("Loading the MultiNLI dataset...")
try:
    full_dataset_dict = load_dataset("nyu-mll/multi_nli")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have a stable connection. Exiting.")
    exit()

# 3. Filter for the desired splits (train and validation_matched)
print("Filtering for 'train' and 'validation_matched' splits...")
target_splits = ['train', 'validation_matched']
dataset_to_process = DatasetDict({
    split: full_dataset_dict[split]
    for split in target_splits
    if split in full_dataset_dict
})

# 4. Define the transformation function
def format_and_combine_text(example):
    """
    Combines 'premise' and 'hypothesis' into a single 'text' column 
    and keeps 'pairID' and 'label'.
    """
    # Create the new 'text' column in the requested format
    example['text'] = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}"
    return example

# Define the columns we want to remove from the original dataset
columns_to_remove = [
    'promptID', 
    'premise', 
    'premise_binary_parse', 
    'premise_parse', 
    'hypothesis', 
    'hypothesis_binary_parse', 
    'hypothesis_parse', 
    'genre'
]

# 5. Apply the transformation and remove columns across the filtered splits
print("Applying transformation and removing unnecessary columns...")
modified_dataset_dict = dataset_to_process.map(
    format_and_combine_text,
    remove_columns=columns_to_remove
)

# 6. Inspect the modified dataset structure
print("\n--- Modified Dataset Structure ---")
print(modified_dataset_dict)
print("\n--- Example of a Processed Row (Train Split) ---")
print(modified_dataset_dict['train'][0]) 

# 7. Upload the modified dataset to the Hugging Face Hub
# **IMPORTANT: Replace "YOUR_USERNAME/your_new_dataset_name" with your actual repo path**
repo_id = "Revesis/multi_nli_setfit_formatted" 

try:
    print(f"\nUploading modified dataset to: {repo_id}")
    # The DatasetDict will push the 'train' and 'validation_matched' splits
    modified_dataset_dict.push_to_hub(repo_id)
    print(f"\n✅ Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_id}")
except Exception as e:
    print(f"\n❌ Error during push_to_hub. Check your login status and repo_id format. Error: {e}")