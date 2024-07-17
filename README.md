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

# Jobs Visualizing

# Job Description Visualization Tool

This tool creates a 2D visualization of job descriptions using BERT embeddings and UMAP projection. It processes a CSV file containing job data, embeds the job descriptions using BERT, projects them into 2D space using UMAP, and creates an interactive visualization where points are colored by the average salary.

## Features

- Load and preprocess job data from a CSV file
- Embed job descriptions using BERT (CLS token)
- Project embeddings to 2D using UMAP
- Create an interactive Plotly visualization
- Color-code points based on average salary

## Requirements

- Python 3.7+
- pandas
- torch
- transformers
- numpy
- umap-learn
- plotly
- click

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/job-description-visualization.git
   cd job-description-visualization
   ```

2. Install the required packages:
   ```
   pip install pandas torch transformers numpy umap-learn plotly click
   ```

## Usage

Run the script from the command line, providing the input CSV file and the desired output HTML file:

```
python job_visualization.py Jobs_NYC_Postings.csv jobs_vizualization.html
```

### Input CSV Format

```
>>> df.columns
Index(['Job ID', 'Agency', 'Posting Type', '# Of Positions', 'Business Title',
       'Civil Service Title', 'Title Classification', 'Title Code No', 'Level',
       'Job Category', 'Full-Time/Part-Time indicator', 'Career Level',
       'Salary Range From', 'Salary Range To', 'Salary Frequency',
       'Work Location', 'Division/Work Unit', 'Job Description',
       'Minimum Qual Requirements', 'Preferred Skills',
       'Additional Information', 'To Apply', 'Hours/Shift', 'Work Location 1',
       'Recruitment Contact', 'Residency Requirement', 'Posting Date',
       'Post Until', 'Posting Updated', 'Process Date'],
      dtype='object')
```

### Output

The script generates an HTML file containing an interactive Plotly visualization. Each point in the visualization represents a job description, with its position determined by the UMAP projection of its BERT embedding. The color of each point represents the average salary for that job.

## Customization

You can modify the `job_visualization.py` script to adjust various parameters:

- UMAP parameters (n_neighbors, n_components) in the `visualize_umap` function
- Plotly figure layout and styling in the `visualize_umap` function
- BERT model selection in the `load_model_and_tokenizer` function

## Contributing

Contributions to improve the tool are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.