import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import umap
import plotly.graph_objects as go
import click


def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def load_model_and_tokenizer():
    print("Loading pre-trained BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return model, tokenizer


def process_nouns(file_path, n_nouns: int):
    model, tokenizer = load_model_and_tokenizer()

    # Read nouns from file
    print("Reading nouns from file...")
    with open(file_path, "r") as f:
        nouns = [line.strip() for line in f if line.strip()]

    nouns = nouns[:n_nouns]

    # Calculate embeddings
    print("Calculating BERT embeddings")
    embeddings = [get_bert_embedding(noun, model, tokenizer) for noun in nouns]

    return embeddings, nouns


def visualize_umap(embeddings, labels, output_path):
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
            marker=dict(size=10),
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
    fig.write_html(output_path)
    print(f"Visualization saved as '{output_path}'")


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--n_nouns", default=100, help="Number of nouns to process")
def umap_visualization(input_file, output_file, n_nouns):
    """Generate UMAP visualization for nouns."""
    embeddings, nouns = process_nouns(input_file, n_nouns)
    visualize_umap(embeddings, nouns, output_file)


@cli.command()
@click.option("--phrase1", required=True, help="First phrase for comparison")
@click.option("--phrase2", required=True, help="Second phrase for comparison")
def cosine_sim(phrase1, phrase2):
    """Compute cosine similarity between two phrases."""
    model, tokenizer = load_model_and_tokenizer()

    embedding1 = get_bert_embedding(phrase1, model, tokenizer)
    embedding2 = get_bert_embedding(phrase2, model, tokenizer)

    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    click.echo(f"Cosine similarity between '{phrase1}' and '{phrase2}': {similarity}")


if __name__ == "__main__":
    cli()
