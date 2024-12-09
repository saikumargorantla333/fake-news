
# Fake News Detection Using NLP Techniques

## Project Overview
This project addresses the issue of misinformation by building a system that classifies news articles as reliable or unreliable using machine learning techniques. The system combines advanced natural language processing (NLP) methods and a user-friendly deployment using Streamlit.

## Features
- **Data Preprocessing Pipeline**: Includes text cleaning, tokenization, stopword removal, stemming, and TF-IDF vectorization.
- **Baseline Models**: Implements Logistic Regression and Support Vector Machines (SVM) for initial benchmarks.
- **Advanced Models**: Uses CNN-RNN hybrid architectures and transformer-based models like BERT.
- **Deployment**: Streamlit-based web application for real-time predictions.

## Dataset
- **Source**: Publicly available dataset from Kaggle.
- **Features**: Includes textual data (e.g., article content) and labels indicating reliability (0 = Reliable, 1 = Unreliable).
- **Split**: 80% training, 20% testing.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/saikumargorantla333/fake-news
   ```
2. **Navigate to the project directory:**
   ```bash
   cd Fake-News-Detection
   ```
3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Preprocess the Data
Run the preprocessing pipeline to clean and prepare the data for modeling:
```bash
python src/preprocess.py
```

### 2. Train the Model
Train the baseline and advanced models:
```bash
python src/model.py
```

### 3. Evaluate the Model
Assess model performance using evaluation metrics:
```bash
python src/evaluate.py
```

### 4. Run the Web Application
Launch the Streamlit web application for real-time fake news classification:
```bash
streamlit run app/app.py
```

## Results
- **Accuracy**: 92%
- **Precision**: 89%
- **Recall**: 94%
- **F1-Score**: 91%

## Project Structure
```
Fake-News-Detection/
├── data/
│   ├── train.csv
│   ├── test.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── baseline_model.ipynb
│   ├── advanced_model.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── evaluate.py
│   ├── deploy.py
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
├── README.md
├── results/
│   ├── metrics.csv
│   ├── plots/
├── slides/
│   └── project_presentation.pptx
├── requirements.txt
└── LICENSE
```

## Contributions
We welcome contributions from the community! Feel free to fork the repository, submit issues, or create pull requests.

## Troubleshooting
- If you encounter a module not found error, ensure you have installed all dependencies in `requirements.txt`.
- For Streamlit issues, ensure you’re using the correct Python environment.

## References
- **Dataset**: [Kaggle Dataset for Fake News Detection](https://www.kaggle.com)
- **Libraries**: Pandas, Scikit-Learn, NLTK, TensorFlow, Streamlit
- **Papers**:
  1. "Attention Is All You Need" - Vaswani et al., 2017
  2. "BERT: Pre-training of Deep Bidirectional Transformers" - Devlin et al., 2019

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
