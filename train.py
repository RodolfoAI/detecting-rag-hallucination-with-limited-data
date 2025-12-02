from datasets import load_dataset, concatenate_datasets, Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from transformers import BitsAndBytesConfig
from sentence_transformers.losses import BatchAllTripletLoss
import torch
from sklearn.metrics import classification_report
import numpy as np

EVAL_SAMPLES = [2, 3, 5, 10, 20]
FULL_TRAIN_SAMPLES = 20

BALANCED_TEST_SIZE_PER_CLASS = 338 # half of 675, rounded up for each class, this represents 50% of the original test set

# Load the full dataset (splits)
rag_dataset = load_dataset("Revesis/rag_truth_hallucination_binary")

# Create a Balanced Full Training Dataset (50/50) ---
train_pos = rag_dataset["train"].filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(FULL_TRAIN_SAMPLES // 2))
train_neg = rag_dataset["train"].filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(FULL_TRAIN_SAMPLES // 2))

# Combine, shuffle, and store as the source for your subsets
full_rag_train_dataset = concatenate_datasets([train_pos, train_neg]).shuffle(seed=42)

# Create a Balanced Test Dataset ---
test_pos = rag_dataset["test"].filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(BALANCED_TEST_SIZE_PER_CLASS))
test_neg = rag_dataset["test"].filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(BALANCED_TEST_SIZE_PER_CLASS))

rag_test_dataset = concatenate_datasets([test_pos, test_neg]).shuffle(seed=42)

results = {}
previous_sample_count = 0

# Iterate through the desired sample counts
for current_sample_count in EVAL_SAMPLES:
    print(f"\n--- Training with {current_sample_count} Samples ---")

    # Select the subset of the training data
    rag_train_subset = full_rag_train_dataset.select(range(current_sample_count))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2", 
        use_differentiable_head=True, head_params={"out_features": 2}).to(device)
    # All tested models
    # google-bert/bert-base-cased
    # kathaem/bert-base-cased-sentence-transformer-mnli-en
    # sentence-transformers/all-mpnet-base-v2
    # lizchu414/mpnet-base-all-nli-triplet

    args = TrainingArguments(
    batch_size=32,
    num_epochs=1,
    num_iterations=1,
    sampling_strategy="unique",
    logging_steps=1,
    end_to_end=True,
    loss = BatchAllTripletLoss, # delete this to use cosine similarity (non triplet)
    body_learning_rate=1e-5, 
    head_learning_rate=1e-2, 
    l2_weight=0.0, # Weight decay on **both** the body and head. If `None`, will use 0.01.
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=rag_train_subset,
        metric="accuracy", 
        column_mapping={"text": "text", "label": "label"}
    )

    # Unfreeze the entire model for end-to-end training
    trainer.unfreeze(keep_body_frozen=False)

    trainer.train()


    # Get predictions on the test set
    test_texts = rag_test_dataset["text"]

    # The output will be a PyTorch tensor, likely on the CUDA device if available
    predictions_tensor = model.predict(test_texts)

    # Move the predictions tensor to CPU and convert to a NumPy array
    predictions = predictions_tensor.cpu().numpy()

    true_labels = np.array(rag_test_dataset["label"])

    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)

    metrics = {
        'accuracy': report['accuracy'],
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
        'f1_macro': report['macro avg']['f1-score'],
    }


    # Store and print results
    results[current_sample_count] = metrics
    print(f"Evaluation Metrics after {current_sample_count} samples: {metrics}")

    # Track the number of samples used
    previous_sample_count = current_sample_count

print("\n" + "="*80)
print("All Results Summary:")
print("| Samples | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |")
print("|---------|----------|-----------------|--------------|----------------|")
for samples, metrics in results.items():
    print(f"| {samples:<7} | {metrics['accuracy']:^8.4f} | {metrics['precision_macro']:^15.4f} | {metrics['recall_macro']:^12.4f} | {metrics['f1_macro']:^14.4f} |")

# Print peak memory usage (from the last run)
peak_memory_bytes = torch.cuda.max_memory_allocated()
peak_memory_gb = peak_memory_bytes / (1024 ** 3)
print(f"\nPeak Memory Usage (Last Run): {peak_memory_gb:.2f} GB")

# Disconnect runtime automatically
#from google.colab import runtime
#runtime.unassign()