import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import numpy as np
import umap
import plotly.graph_objects as go
import click
import os
import textwrap
from loguru import logger


def remove_non_ascii(text):
    return "".join([i if ord(i) < 128 else " " for i in str(text)])


def normalize_salary(row):
    if row["Salary Frequency"] == "Annual":
        return row["Salary Range From"], row["Salary Range To"]
    elif row["Salary Frequency"] == "Daily":
        return row["Salary Range From"] * 260, row["Salary Range To"] * 260
    elif row["Salary Frequency"] == "Hourly":
        return row["Salary Range From"] * 2080, row["Salary Range To"] * 2080
    else:
        return np.nan, np.nan


def load_and_preprocess_data(file_path, columns_to_normalize):
    logger.info(f"Loading and preprocessing data from {file_path}")
    df = pd.read_csv(file_path)

    for column in tqdm(columns_to_normalize, desc="Cleaning data"):
        if column in df.columns:
            df[column] = df[column].apply(remove_non_ascii)

    if all(
        col in columns_to_normalize
        for col in ["Salary Range From", "Salary Range To", "Salary Frequency"]
    ):
        df["Salary Range From"] = pd.to_numeric(
            df["Salary Range From"], errors="coerce"
        )
        df["Salary Range To"] = pd.to_numeric(df["Salary Range To"], errors="coerce")
        df[["Normalized Salary From", "Normalized Salary To"]] = df.apply(
            normalize_salary, axis=1, result_type="expand"
        )
        df["Average Salary"] = (
            df["Normalized Salary From"] + df["Normalized Salary To"]
        ) / 2
        logger.info("Salary normalization completed")
    else:
        df["Average Salary"] = (
            df["Salary Range From"].astype(float) + df["Salary Range To"].astype(float)
        ) / 2
        logger.info("Salary normalization skipped, using original values")

    return df


def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Use CLS token


def load_model_and_tokenizer():
    logger.info("Loading pre-trained BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    logger.info("BERT model and tokenizer loaded successfully")
    return model, tokenizer


def process_job_descriptions(df, model, tokenizer, cache_dir, col):
    logger.info(f"Calculating BERT embeddings for {col}")
    embeddings = []

    cache_dir = os.path.join(cache_dir, col)
    os.makedirs(cache_dir, exist_ok=True)

    for idx, desc in tqdm(
        enumerate(df[col]),
        total=len(df),
        desc=f"Generating '{col}' embeddings",
    ):
        cache_file = os.path.join(cache_dir, f"embedding_{idx}.npy")

        if os.path.exists(cache_file):
            embedding = np.load(cache_file)
        else:
            embedding = get_bert_embedding(desc, model, tokenizer)
            np.save(cache_file, embedding)

        embeddings.append(embedding)

    logger.info(f"Completed BERT embeddings for {col}")
    return np.array(embeddings)


def wrap_text(text, width=50):
    return "<br>".join(textwrap.wrap(text, width=width))


def visualize_umap(embeddings, df, output_path, embed_column):
    logger.info("Applying UMAP and creating visualization")
    reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42, n_jobs=1)
    embedding_2d = reducer.fit_transform(embeddings)

    hover_text = df.apply(
        lambda row: (
            f"Business Title: {wrap_text(row['Business Title'])}<br>"
            f"Civil Service Title: {wrap_text(row['Civil Service Title'])}<br>"
            f"Job Category: {wrap_text(row['Job Category'])}<br>"
            f"Salary Range: ${row['Normalized Salary From']:,.2f} - ${row['Normalized Salary To']:,.2f} (Annual)<br>"
            f"Average Salary: ${row['Average Salary']:,.2f}<br>"
            f"{'-' * 50}<br>"
            f"{embed_column}:<br>{wrap_text(row[embed_column])}"
            if "Normalized Salary From" in df.columns
            else f"Business Title: {wrap_text(row['Business Title'])}<br>"
            f"Civil Service Title: {wrap_text(row['Civil Service Title'])}<br>"
            f"Job Category: {wrap_text(row['Job Category'])}<br>"
            f"Salary Range: ${row['Salary Range From']:,.2f} - ${row['Salary Range To']:,.2f}<br>"
            f"Average Salary: ${row['Average Salary']:,.2f}<br>"
            f"{'-' * 50}<br>"
            f"{embed_column}:<br>{wrap_text(row[embed_column])}"
        ),
        axis=1,
    )

    fig = go.Figure(
        data=go.Scatter(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            mode="markers",
            marker=dict(
                size=6,
                color=df["Average Salary"],
                colorscale="Viridis",
                colorbar=dict(title="Average Salary"),
                showscale=True,
            ),
            text=hover_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=f"UMAP Projection of {embed_column} (BERT Embeddings)",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        height=800,
        width=1000,
    )

    fig.write_html(output_path)
    logger.info(f"Visualization saved as '{output_path}'")


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--cache-dir",
    type=click.Path(),
    default="embedding_cache",
    help="Directory to store embedding cache",
)
@click.option(
    "--embed-column",
    type=str,
    default="Preferred Skills",
    help="Column to use for embedding and projection",
)
@click.option(
    "--normalize",
    type=str,
    default="Business Title,Civil Service Title,Job Category,Salary Range From,Salary Range To,Salary Frequency,Preferred Skills",
    help="Comma-separated list of columns to normalize",
)
def main(input_file, output_file, cache_dir, embed_column, normalize):
    """Generate UMAP visualization for job descriptions."""
    logger.info("Starting job description visualization process")
    columns_to_normalize = normalize.split(",")
    df = load_and_preprocess_data(input_file, columns_to_normalize)
    model, tokenizer = load_model_and_tokenizer()
    embeddings = process_job_descriptions(
        df, model, tokenizer, cache_dir, col=embed_column
    )
    visualize_umap(embeddings, df, output_file, embed_column)
    logger.info("Job description visualization process completed")


if __name__ == "__main__":
    main()
