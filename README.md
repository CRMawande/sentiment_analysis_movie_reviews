# Movie Review Sentiment Analysis using DistilBERT

## Project Overview
This project implements a Natural Language Processing (NLP) system for sentiment analysis on movie reviews using transformer-based models. The system classifies movie reviews as either positive or negative sentiment using DistilBERT, a distilled version of BERT optimized for computational efficiency.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Dataset
This project uses the **IMDB Large Movie Review Dataset** (Maas et al., 2011), which contains:
- 50,000 movie reviews (25,000 training, 25,000 testing)
- Binary sentiment labels (positive/negative)
- Balanced distribution across classes

**Citation:**
```
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). 
Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual 
Meeting of the Association for Computational Linguistics: Human Language Technologies 
(pp. 142-150).
```

**Dataset Source:** [Stanford AI Lab IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CRMawande/sentiment_analysis_movie_reviews.git
cd movie-sentiment-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the IMDB dataset:
```bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

## Project Structure
```
movie-sentiment-analysis/
│
├── movie_review_analysis.ipynb    
├── requirements.txt               
├── README.md                      
├── results/                       
│   ├── results/                   
│   └── results_small/            
├── logs/                        
└── data/                      
    └── aclImdb/
        ├── train/
        └── test/
```

## Usage

### Running the Notebook
1. Open Jupyter Notebook:
```bash
jupyter notebook movie_review_analysis.ipynb
```

2. Execute cells sequentially to:
   - Load and preprocess data
   - Train models on small (1,250 samples) and large (25,000 samples) datasets
   - Evaluate model performance
   - Run inference on new reviews

## Results

### Small Dataset (1,250 samples)
- **Training epochs:** 5
- **Final accuracy:** 88.72%
- **Best validation loss:** 0.4354

### Large Dataset (25,000 samples)
- **Training epochs:** 3
- **Final accuracy:** 92.60%
- **Test set performance:**
  - Precision (Negative): 93.91%
  - Precision (Positive): 91.36%
  - Recall (Negative): 91.10%
  - Recall (Positive): 94.10%
  - F1-score: 92.59%

### Key Findings
- Larger dataset improved accuracy by 3.88 percentage points
- Model generalizes well with balanced precision and recall
- Training time: ~78 minutes on GPU (T4)

## Model Architecture
- **Base Model:** DistilBERT (distilbert-base-uncased)
- **Parameters:** ~67 million
- **Classification Head:** Linear layer with 2 outputs (positive/negative)
- **Optimizer:** AdamW
- **Learning Rate:** 2e-5
- **Batch Size:** 16

## Performance Metrics Visualization
Refer to the notebook for detailed visualizations including:
- Training/validation loss curves
- Confusion matrices
- Per-class performance metrics

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Hugging Face for the Transformers library
- Stanford AI Lab for the IMDB dataset
- IU International University of Applied Sciences

## Contact
For questions or feedback, please open an issue on GitHub.

