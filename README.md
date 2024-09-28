# Clustering-Based Summarization with GPT-4o-Mini and Text-Embedding-3-Large

This project implements a general-purpose approach to summarizing large datasets using clustering and retrieval techniques. It uses the **text-embedding-3-large** model for embeddings and the **gpt-4o-mini** model for generating summaries.

## Features

- **Embedding**: Data entries are transformed into vector embeddings using OpenAI's `text-embedding-3-large` model.
- **Clustering**: The embeddings are clustered into topics using KMeans.
- **Retrieval**: The top `N` data points closest to each cluster centroid are retrieved.
- **Summarization**: GPT-4o-mini generates a summary of the top data points for each topic.

## How It Works

1. **Input Data**: The script reads textual data from either a CSV or a TXT file.
2. **Embeddings**: The textual data is embedded using the `text-embedding-3-large` model via OpenAI's API.
3. **Clustering**: KMeans clustering groups the embeddings into topics.
4. **Retrieval**: For each topic, the closest data points to the cluster centroids are retrieved.
5. **Summarization**: GPT-4o-mini is used to summarize the top data points for each cluster.

## Requirements

Install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```
