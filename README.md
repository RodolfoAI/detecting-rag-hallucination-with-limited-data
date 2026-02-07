Grounded LLM outputs are prevalent in contemporary AI applications, particularly
those encompassing RAG with indexed document databases and agent-based search
via web search. However, despite the advancement of capabilities in LLMs and
improvements in information retrieval techniques, intrinsic hallucination remains
a significant concern in these systems, as hallucinations undermine user trust, especially when they arise from within the model itself. The standard approach for
hallucination detection and other quality benchmarks of deployed LLMs is the LLM
as a judge method. However, this method is not accessible to everyone due to its
relatively high operational costs, along with other limitations. This paper explores an efficient solution to this problem by training an embedding model using the SetFit library. SetFit fine-tunes the embedding model using contrastive loss and adds a classification head. This approach, using only three training samples achieves an F1 score of 71.87% on the RAG Truth test set adapted for binary classification using a 0.1 billion parameter model. This performance is superior to using gpt-4-turbo as a judge with a custom prompt, which achieves 63.4% and is also comparable to other similar low-parameter models, such as lettucedetect-base-v1 which achieves 76.07%, when trained on the full 15.1k sample dataset. Our approach enables the creation of hallucination detection models with just a few samples while mainting computational resources low.

[Paper](hallucinations_llm_rodolfo.pdf)