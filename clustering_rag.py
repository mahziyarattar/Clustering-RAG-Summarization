import openai
import requests
import json
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

# Set your OpenAI API key
openai.api_key = 'your-api-key-here'

# Step 1: Read Data from CSV or Text File
def read_data(file_path, file_type='csv', column_name=None):
    """
    Reads text data from a CSV or text file.

    Args:
        file_path (str): The path to the CSV or text file.
        file_type (str): The type of the file ('csv' or 'txt').
        column_name (str): The column name in the CSV file containing the text data (only for CSV).

    Returns:
        data (list of str): The list of textual data entries.
    """
    if file_type == 'csv':
        if column_name is None:
            raise ValueError("Please specify the column name containing the text data in the CSV file.")
        df = pd.read_csv(file_path)
        data = df[column_name].dropna().tolist()
    elif file_type == 'txt':
        with open(file_path, 'r') as f:
            data = f.readlines()
        data = [line.strip() for line in data if line.strip()]
    else:
        raise ValueError("Unsupported file type. Please use 'csv' or 'txt'.")
    
    return data

# Step 2: Get embeddings using text-embedding-3-large model
def get_embeddings(data):
    """
    Retrieve embeddings from the OpenAI API using the text-embedding-3-large model.
    
    Args:
        data (list of str): The list of textual data to be embedded.
    
    Returns:
        embeddings (list of list of float): The list of embeddings.
    """
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=data
    )
    embeddings = [item['embedding'] for item in response['data']]
    return embeddings

# Step 3: KMeans clustering to group data into topics
def cluster_data(embeddings, num_clusters=10):
    """
    Apply KMeans clustering to the embeddings to group the data into topics.

    Args:
        embeddings (list of list of float): The list of embeddings to be clustered.
        num_clusters (int): The number of clusters/topics.

    Returns:
        labels (list of int): The cluster labels for each data point.
        centroids (list of list of float): The centroids of each cluster.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    return labels, centroids

# Step 4: Retrieve top N closest data points to each centroid
def get_top_chunks(embeddings, centroids, labels, top_n=3):
    """
    Retrieve the top N data points closest to each cluster's centroid.

    Args:
        embeddings (list of list of float): The list of embeddings.
        centroids (list of list of float): The list of centroid vectors.
        labels (list of int): The cluster labels for each data point.
        top_n (int): The number of top closest points to retrieve for each cluster.

    Returns:
        top_chunks (list of list of int): Indices of the top closest data points for each cluster.
    """
    top_chunks = []
    for i in range(len(centroids)):
        cluster_points = np.array([embeddings[j] for j in range(len(embeddings)) if labels[j] == i])
        centroid = centroids[i].reshape(1, -1)
        distances = cdist(cluster_points, centroid, 'euclidean')
        closest_indices = np.argsort(distances.flatten())[:top_n]
        top_chunks.append(closest_indices)
    return top_chunks

# Step 5: Summarize top chunks using GPT-4o-mini
def summarize_chunks(data, top_chunks, labels):
    """
    Summarize each cluster by selecting the top closest data points and generating a summary.

    Args:
        data (list of str): The original data entries.
        top_chunks (list of list of int): The indices of the top closest points for each cluster.
        labels (list of int): The cluster labels for each data point.

    Returns:
        summaries (list of str): The generated summaries for each cluster.
    """
    summaries = []
    for i, chunk_indices in enumerate(top_chunks):
        cluster_data = [data[j] for j in range(len(data)) if labels[j] in chunk_indices]
        cluster_text = " ".join(cluster_data)
        summary = generate_gpt_summary(cluster_text)
        summaries.append(summary)
    return summaries

# Function to generate summary using GPT-4o-mini
def generate_gpt_summary(text):
    """
    Use GPT-4o-mini to summarize a chunk of data.

    Args:
        text (str): The text to summarize.

    Returns:
        summary (str): The generated summary.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
        ],
        "max_completion_tokens": 150
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"Error in GPT-4o-mini API request: {response.text}")

# Step 6: Main function to execute the entire pipeline
def summarize_data(file_path, file_type='csv', column_name=None, num_clusters=10, top_n=3):
    """
    Complete pipeline to summarize the data based on clustering and retrieval.

    Args:
        file_path (str): The path to the input file (CSV or TXT).
        file_type (str): The type of file ('csv' or 'txt').
        column_name (str): The column name in the CSV file containing the text data.
        num_clusters (int): The number of clusters/topics.
        top_n (int): The number of top closest points to retrieve for each cluster.

    Returns:
        summaries (list of str): The summaries for each cluster/topic.
    """
    # Step 1: Read data from the file
    data = read_data(file_path, file_type, column_name)

    # Step 2: Get embeddings for the data
    embeddings = get_embeddings(data)

    # Step 3: Cluster data into topics
    labels, centroids = cluster_data(embeddings, num_clusters)

    # Step 4: Retrieve the top closest data points to centroids
    top_chunks = get_top_chunks(embeddings, centroids, labels, top_n)

    # Step 5: Summarize the top chunks
    summaries = summarize_chunks(data, top_chunks, labels)

    return summaries

# Example usage:
if __name__ == "__main__":
    # Example usage for CSV file
    file_path = "example_data.csv"
    summaries = summarize_data(file_path, file_type='csv', column_name='text_column', num_clusters=5, top_n=3)

    # Print the summaries for each cluster
    for i, summary in enumerate(summaries):
        print(f"Summary for Topic {i+1}:\n{summary}\n")

    # Example usage for text file
    file_path = "example_data.txt"
    summaries = summarize_data(file_path, file_type='txt', num_clusters=5, top_n=3)

    # Print the summaries for each cluster
    for i, summary in enumerate(summaries):
        print(f"Summary for Topic {i+1}:\n{summary}\n")
