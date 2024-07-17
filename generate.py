import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import umap
import plotly.graph_objects as go


def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def process_nouns(file_path, n_nouns: int, n_neighbors: int):
    # Load pre-trained model and tokenizer
    print("Loading pre-trained BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Read nouns from file
    print("Reading nouns from file...")
    with open(file_path, "r") as f:
        nouns = [line.strip() for line in f if line.strip()]

    nouns = nouns[:n_nouns]

    # Calculate embeddings
    print("Calculating BERT embeddings")
    embeddings = [get_bert_embedding(noun, model, tokenizer) for noun in nouns]

    # Calculate pairwise similarities
    print("Calculating pairwise similarities")
    similarities = cosine_similarity(embeddings)

    # Find N closest neighbors for each noun
    print("Generating graph")
    graph = {"nodes": [], "links": []}
    for i, noun in enumerate(nouns):
        graph["nodes"].append({"id": noun})
        neighbors = similarities[i].argsort()[-n_neighbors - 1 : -1][
            ::-1
        ]  # exclude self
        for neighbor in neighbors:
            graph["links"].append(
                {
                    "source": noun,
                    "target": nouns[neighbor],
                    "similarity": float(
                        similarities[i][neighbor]
                    ),  # Convert to float for JSON serialization
                }
            )

    # Save graph to JSON file
    with open("graph_data.json", "w") as f:
        json.dump(graph, f, indent=4)

    # Apply UMAP and visualize
    visualize_umap(embeddings, nouns)


def visualize_umap(embeddings, labels):
    print("Applying UMAP and creating visualization...")

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)

    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings_array)

    # Create Plotly scatter plot
    fig = go.Figure(
        data=go.Scatter(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            mode="markers+text",
            marker=dict(
                size=10,
                # color=np.arange(len(labels)),  # color by index
                # colorscale="Viridis",
                # showscale=True,
            ),
            text=labels,
            textposition="top center",
        )
    )

    fig.update_layout(
        title="UMAP Projection of BERT Embeddings",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        height=800,
        width=1000,
    )

    # Save the plot as an HTML file
    fig.write_html("umap_visualization.html")
    print("Visualization saved as 'umap_visualization.html'")


if __name__ == "__main__":
    process_nouns(
        "./nouns.txt", n_nouns=100, n_neighbors=3
    )  # You can adjust n_neighbors as needed
