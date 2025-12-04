from datasets import load_dataset, DatasetDict
from huggingface_hub import login

login()

print("Loading the RAGTruth dataset...")
try:
    full_dataset_dict = load_dataset("wandb/RAGTruth-processed")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

target_splits = ['train', 'test']
dataset_to_process = DatasetDict({
    split: full_dataset_dict[split]
    for split in target_splits
    if split in full_dataset_dict
})

def preprocess_example(example):
    """
    - Keeps 'id' as 'pairID'
    - Computes binary label
    - Combines 'context' and 'output' into MNLI-style 'text'
    - Cleans trailing '\n Output:' from context if present
    """
    example['pairID'] = example['id']
    
    example['label'] = int(any(example['hallucination_labels_processed'].values()))
    
    context_clean = example['context'].replace('\noutput:', '').strip()
    
    example['text'] = f"Premise: {context_clean} Hypothesis: {example['output']}"
    return example

print("Applying preprocessing...")
modified_dataset_dict = dataset_to_process.map(preprocess_example)

columns_to_remove = [
    'id', 'query', 'context', 'output', 'task_type', 'quality',
    'model', 'temperature', 'hallucination_labels', 'hallucination_labels_processed', 'input_str'
]
modified_dataset_dict = modified_dataset_dict.remove_columns(columns_to_remove)

print("\n--- Processed Dataset Structure ---")
print(modified_dataset_dict)
print("\n--- Example of a Processed Row (Train Split) ---")
print(modified_dataset_dict['train'][0])

repo_id = "Revesis/rag_truth_hallucination_binary"
try:
    print(f"\nUploading processed dataset to: {repo_id}")
    modified_dataset_dict.push_to_hub(repo_id)
    print(f"\nSuccessfully uploaded dataset to: https://huggingface.co/datasets/{repo_id}")
except Exception as e:
    print(f"\nError during push_to_hub. Error: {e}")
