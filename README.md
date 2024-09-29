# Clustering-Based Summarization with GPT-4o-Mini and Text-Embedding-3-Large

This project implements a general-purpose approach to summarizing large datasets using clustering and retrieval techniques. It uses the **text-embedding-3-large** model for embeddings and the **gpt-4o-mini** model for generating summaries.

## Features

- **Embedding Generation**: Converts textual data into high-dimensional vector embeddings using OpenAI's `text-embedding-3-large` model.
- **Clustering**: Uses the KMeans algorithm to cluster data into topics.
- **Top-N Retrieval**: Retrieves the top `N` most representative data points closest to each cluster centroid.
- **Summarization**: Combines the top `N` data points from all clusters and generates a single summary using GPT-4o-mini.

## Workflow

1. **Data Input**: The pipeline reads textual data from a CSV or TXT file.
2. **Embedding Generation**: Each text entry is embedded into vector space using OpenAI's `text-embedding-3-large` model.
3. **Clustering**: KMeans is used to group the embeddings into clusters/topics.
4. **Top-N Retrieval**: For each cluster, the top `N` data points closest to the centroid are retrieved.
5. **Summarization**: The retrieved data points from all clusters are concatenated and passed to GPT-4o-mini for summarization in a single request.

## Requirements

Install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```
