# Noun Processing with BERT and UMAP Visualization

This project processes nouns using BERT embeddings, calculates similarities, and visualizes the results using UMAP and Plotly.

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

1. Ensure you have a file named `nouns.txt` in the same directory as the script, with one noun per line.

2. Run the script:
   ```
   python process_nouns.py
   ```

3. The script will generate two files:
   - `graph_data.json`: Contains the graph data of noun similarities.
   - `umap_visualization.html`: An interactive visualization of the UMAP projection.

You can open the `umap_visualization.html` file in a web browser to explore the visualization.