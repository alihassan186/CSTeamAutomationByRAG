# Example document

This is a small document bundled with the project.

## What is RAG?

Retrieval-Augmented Generation (RAG) is a pattern where an LLM answers questions using **retrieved context** from your own data.

Key steps:
1. Split documents into chunks.
2. Convert chunks into embeddings and store them in a vector database.
3. At query time, retrieve the most relevant chunks.
4. Provide them to the LLM in the prompt to generate a grounded answer.

## Why use RAG?

- Up-to-date answers (you can refresh the index)
- Better grounding and fewer hallucinations
- Ability to use private domain knowledge
