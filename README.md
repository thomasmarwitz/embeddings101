# Noun Processing with BERT, UMAP Visualization, and Cosine Similarity CLI

This project processes nouns using BERT embeddings, calculates similarities, visualizes the results using UMAP and Plotly, and provides a CLI for UMAP visualization and cosine similarity computation.

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

The script now provides two CLI commands:

### 1. UMAP Visualization

Generate a UMAP visualization for nouns:

```
python tool.py umap-visualization nouns.txt umap_visualization.html
```

Options:
- `--n_nouns INTEGER`: Number of nouns to process (default: 100)

Example:
```
python tool.py umap-visualization nouns.txt visualization.html --n_nouns 200 --n_neighbors 5
```

### 2. Cosine Similarity

Compute the cosine similarity between two phrases:

```
python tool.py cosine-sim --phrase1 TEXT --phrase2 TEXT
```

Options:
- `--phrase1 TEXT`: First phrase for comparison (required)
- `--phrase2 TEXT`: Second phrase for comparison (required)

Example:
```
python tool.py cosine-sim --phrase1 "hello world" --phrase2 "greetings earth"
```

This command will output the cosine similarity between the two phrases.

## Note

The UMAP visualization command requires a file named `nouns.txt` (or another specified input file) in the same directory as the script, with one noun per line.