from datasets import load_dataset, DatasetDict
from huggingface_hub import login

# 1. Authenticate
login()

# 2. Load the RAGTruth dataset
print("Loading the RAGTruth dataset...")
try:
    full_dataset_dict = load_dataset("wandb/RAGTruth-processed")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 3. Filter for train and test splits
target_splits = ['train', 'test']
dataset_to_process = DatasetDict({
    split: full_dataset_dict[split]
    for split in target_splits
    if split in full_dataset_dict
})

# 4. Define preprocessing function
def preprocess_example(example):
    """
    - Keeps 'id' as 'pairID'
    - Computes binary label
    - Combines 'context' and 'output' into MNLI-style 'text'
    - Cleans trailing '\n Output:' from context if present
    """
    # Keep id as pairID
    example['pairID'] = example['id']
    
    # Compute binary label
    example['label'] = int(any(example['hallucination_labels_processed'].values()))
    
    # Clean context string
    context_clean = example['context'].replace('\noutput:', '').strip()
    
    # Combine context and output
    example['text'] = f"Premise: {context_clean} Hypothesis: {example['output']}"
    return example

# 5. Apply preprocessing
print("Applying preprocessing...")
modified_dataset_dict = dataset_to_process.map(preprocess_example)

# 6. Remove unnecessary columns
columns_to_remove = [
    'id', 'query', 'context', 'output', 'task_type', 'quality',
    'model', 'temperature', 'hallucination_labels', 'hallucination_labels_processed', 'input_str'
]
modified_dataset_dict = modified_dataset_dict.remove_columns(columns_to_remove)

# 7. Inspect the processed dataset
print("\n--- Processed Dataset Structure ---")
print(modified_dataset_dict)
print("\n--- Example of a Processed Row (Train Split) ---")
print(modified_dataset_dict['train'][0])

# 8. Upload to Hugging Face Hub
repo_id = "Revesis/rag_truth_hallucination_binary"
try:
    print(f"\nUploading processed dataset to: {repo_id}")
    modified_dataset_dict.push_to_hub(repo_id)
    print(f"\n✅ Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_id}")
except Exception as e:
    print(f"\n❌ Error during push_to_hub. Error: {e}")
