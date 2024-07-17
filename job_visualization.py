import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import numpy as np
import umap
import plotly.graph_objects as go
import click
import os


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df["Average Salary"] = (
        df["Salary Range From"].astype(float) + df["Salary Range To"].astype(float)
    ) / 2
    return df


def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Use CLS token


def load_model_and_tokenizer():
    print("Loading pre-trained BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return model, tokenizer


def process_job_descriptions(df, model, tokenizer, cache_dir):
    print("Calculating BERT embeddings for job descriptions...")
    embeddings = []

    os.makedirs(cache_dir, exist_ok=True)

    for idx, desc in tqdm(
        enumerate(df["Job Description"]),
        total=len(df),
        desc="Generating 'Job Description' embeddings",
    ):
        cache_file = os.path.join(cache_dir, f"embedding_{idx}.npy")

        if os.path.exists(cache_file):
            embedding = np.load(cache_file)
        else:
            embedding = get_bert_embedding(desc, model, tokenizer)
            np.save(cache_file, embedding)

        embeddings.append(embedding)

    return np.array(embeddings)


def visualize_umap(embeddings, df, output_path):
    print("Applying UMAP and creating visualization...")
    reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    fig = go.Figure(
        data=go.Scatter(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            mode="markers",
            marker=dict(
                size=8,
                color=df["Average Salary"],
                colorscale="Viridis",
                colorbar=dict(title="Average Salary"),
                showscale=True,
            ),
            text=df["Business Title"],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="UMAP Projection of Job Descriptions (BERT Embeddings)",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        height=800,
        width=1000,
    )

    fig.write_html(output_path)
    print(f"Visualization saved as '{output_path}'")


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--cache-dir",
    type=click.Path(),
    default="embedding_cache",
    help="Directory to store embedding cache",
)
def main(input_file, output_file, cache_dir):
    """Generate UMAP visualization for job descriptions."""
    df = load_and_preprocess_data(input_file)
    model, tokenizer = load_model_and_tokenizer()
    embeddings = process_job_descriptions(df, model, tokenizer, cache_dir)
    visualize_umap(embeddings, df, output_file)


if __name__ == "__main__":
    main()
