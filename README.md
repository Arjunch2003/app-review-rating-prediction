# Google Play Store App Review Rating Prediction

## 📋 Problem Statement

Predict the star rating (1-5) of Google Play Store application reviews based on review text and related attributes. This is a **multi-class classification problem** with a highly imbalanced dataset.

### Dataset Overview
- **Training Set**: 5,693 reviews with ratings
- **Test Set**: 1,424 reviews without labels (predictions required)
- **Target Variable**: Star Rating (1, 2, 3, 4, 5)
- **Evaluation Metric**: Weighted F1-Score

### Class Distribution (Training Data)
| Rating | Count | Percentage |
|--------|-------|-----------|
| 5-star | 2,923 | 51.34% |
| 1-star | 1,788 | 31.41% |
| 4-star | 611 | 10.73% |
| 3-star | 217 | 3.81% |
| 2-star | 154 | 2.71% |

![Rating Distribution](https://via.placeholder.com/600x250?text=Rating+Distribution+Chart)

**Challenge**: Severe class imbalance requiring specialized handling techniques.

---

## 🔧 Approach & Methodology

### 1. Text Preprocessing
Implemented a comprehensive text cleaning pipeline:

```python
TextPreprocessor
├── Convert to lowercase
├── Remove URLs & HTML tags
├── Remove special characters & numbers
├── Normalize whitespace
├── Tokenization (whitespace-based)
├── Stopword removal (English)
└── Lemmatization (WordNetLemmatizer)
```

**Combined Features**: Merged Review Title + Review Text for richer context

### 2. Feature Extraction
**TF-IDF Vectorization** with optimized parameters:
- Maximum features: 5,000
- N-gram range: Unigrams + Bigrams
- Minimum document frequency: 2
- Maximum document frequency: 0.9

**Output**:
- Training features: 5,693 × 5,000
- Test features: 1,424 × 5,000

### 3. Class Imbalance Handling
- **Balanced Class Weights**: `class_weight="balanced"` (automatic adjustment)
- **Stratified K-Fold**: Preserves class distribution across folds
- **Weighted F1-Score**: Primary evaluation metric accounting for imbalance

### 4. Validation Strategy

#### Nested Cross-Validation Framework
```
┌─ Outer CV (5-Fold Stratified) ────────────────────┐
│  Provides unbiased performance estimates           │
│                                                    │
│  ├─ Inner CV (3-Fold Stratified) ──────────┐     │
│  │  GridSearchCV for Hyperparameter Tuning │     │
│  │                                         │     │
│  │  • LogisticRegression (C: [0.01,1,10]) │     │
│  │  • RandomForest (8 hyperparameters)    │     │
│  └─────────────────────────────────────────┘     │
│                                                    │
│  └─ Evaluate on outer fold test set              │
└────────────────────────────────────────────────────┘
```

**Why Nested CV?**
- Inner CV: Hyperparameter optimization without bias
- Outer CV: Honest, unbiased performance estimation
- Prevents data leakage and overfitting evaluation

---

## 🤖 Models & Results

### Model Comparison

| Model | Accuracy | Weighted F1 | Macro F1 | Outer CV F1 |
|-------|----------|------------|----------|------------|
| Logistic Regression | 85.63% | 0.8655 | 0.8186 | 0.6606 ± 0.0064 |
| **Random Forest** ⭐ | 75.09% | 0.7660 | 0.6799 | **0.6745 ± 0.0067** |

![Logistic Regression Confusion Matrix](https://via.placeholder.com/600x500?text=Logistic+Regression+Confusion+Matrix)

### Best Model: Random Forest
**Selected based on Outer CV F1-Score (most reliable estimate)**

#### Hyperparameters
- Number of trees: 200
- Max depth: None (full tree)
- Max features: log2
- Minimum samples per leaf: 2
- Minimum samples to split: 2
- Class weight: Balanced
- Random state: 42

#### Per-Class Performance
| Rating | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1-star | 0.8964 | 0.8468 | 0.8709 | 1,788 |
| 2-star | 0.5647 | 0.8506 | 0.6788 | 154 |
| 3-star | 0.5714 | 0.7189 | 0.6367 | 217 |
| 4-star | 0.3412 | 0.5679 | 0.4263 | 611 |
| 5-star | 0.8570 | 0.7277 | 0.7870 | 2,923 |

![Confusion Matrix - Random Forest](https://via.placeholder.com/600x500?text=Random+Forest+Confusion+Matrix)

---

## 📊 Project Structure

```
assignment/
├── src/
│   ├── train.py                    # Model training & validation
│   ├── preprocessing.py            # Text preprocessing utilities
│   ├── model_training.ipynb        # Full EDA & analysis notebook
│   └── predict.py                  # Generate test predictions
│
├── data/
│   ├── train.csv                   # Training dataset (5,693 × 5)
│   ├── test.csv                    # Test dataset (1,424 × 4)
│   └── sample_submission.csv       # Submission format reference
│
├── models/
│   └── model_random_forest.joblib  # Serialized trained model
│
├── outputs/
│   ├── predictions.csv             # Final predictions (1,424 rows)
│   ├── confusion_matrix_*.png      # Confusion matrices
│   └── model_comparison_tuned.csv  # Performance comparison
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Installation & Execution

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd assignment
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"
```

### Step 4: Train Model & Generate Predictions

#### Option A: Run Jupyter Notebook (Full Analysis)
```bash
cd src
jupyter notebook model_training.ipynb
```
This provides step-by-step analysis with visualizations.

#### Option B: Run Python Scripts (Direct Training)
```bash
# Train model and validate
python src/train.py

# Generate test predictions
python src/predict.py --input data/test.csv --output predictions.csv
```

### Step 5: Verify Output
```bash
# Check predictions file (should have 1,424 rows)
head predictions.csv

# Expected format:
# id,Star Rating
# 1,5
# 2,4
# ...
```

---

## 📝 Task Checklist

✅ **Text Preprocessing**
- Lowercase conversion
- URL & HTML removal
- Special character removal
- Whitespace normalization
- Tokenization
- Stopword removal
- Lemmatization

✅ **Feature Extraction**
- TF-IDF vectorization
- Unigram & bigram features
- Document frequency filtering
- Dimensionality reduction (5000 features)

✅ **Model Training**
- LogisticRegression with regularization
- RandomForest ensemble classifier
- Class weight balancing for imbalanced data

✅ **Internal Validation**
- Nested cross-validation (5-fold outer, 3-fold inner)
- GridSearchCV for hyperparameter tuning
- Stratified folds for consistent distributions

✅ **Hyperparameter Tuning**
- LogisticRegression: C ∈ [0.01, 0.1, 1, 10]
- RandomForest: 8 hyperparameters (n_estimators, max_depth, etc.)

✅ **Test Predictions**
- Generate predictions.csv with proper format
- No modification to test dataset structure

---

## 🔑 Key Insights

### Model Selection Rationale
- **LogisticRegression**: Higher training accuracy (85.63%) but overfits (Outer CV: 0.6606)
- **RandomForest**: Lower training accuracy (75.09%) but better generalization (Outer CV: 0.6745)
- **Decision**: RandomForest selected due to higher outer CV F1-score (most reliable estimate)

### Class Imbalance Impact
- Minority classes (2, 3, 4 stars) have lower precision/recall
- Balanced class weights help but cannot eliminate imbalance effects
- Weighted F1-score properly accounts for this in evaluation

### Why Outer CV F1 is More Reliable
- Training F1 (0.7660) tends to be optimistic
- Outer CV F1 (0.6745) represents realistic test performance
- No data leakage due to strict train-validation-test separation

---

## 📂 Input/Output Specifications

### Training Data (train.csv)
```
Columns: id, App Version Code, App Version Name, Review Text, Star Rating
- Review Text: 89.43% missing Review Title (handled gracefully)
- Star Rating: Target variable (1-5)
```

### Test Data (test.csv)
```
Columns: id, App Version Code, App Version Name, Review Text
- No Star Rating (this is what we predict)
- Structure remains unchanged
```

### Predictions (predictions.csv)
```
Columns: id, Star Rating
Format: One prediction per row, matching test.csv row order
```

---

## 🛠️ Dependencies

See `requirements.txt` for complete list:
- **Data Processing**: pandas, numpy
- **ML/Validation**: scikit-learn, scipy
- **NLP**: nltk
- **Visualization**: matplotlib, seaborn
- **Serialization**: joblib
- **Notebooks**: jupyter, ipython

---

## 📌 Important Notes

1. **Reproducibility**: All models use `random_state=42`
2. **Class Imbalance**: Handled via `class_weight="balanced"` + weighted F1
3. **Nested CV**: Essential for unbiased performance estimation
4. **Missing Values**: Review Title (89.43% missing) filled with empty strings
5. **Test Predictions**: Generated from best-performing model (RandomForest)

---

## 🎯 Evaluation Metric: Weighted F1-Score

**Why Weighted F1?**
- Accounts for class imbalance automatically
- Balances precision and recall
- Industry standard for imbalanced classification

**Formula**: 
```
Weighted F1 = Σ(weight_i × F1_i) where weight_i = support_i / total_samples
```

---

## 📧 Submission Format

**File**: `predictions.csv`
```csv
id,Star Rating
1,5
2,4
3,1
...
1424,5
```

**Requirements**:
- Exactly 1,424 rows (test set size)
- Two columns: id and Star Rating
- Star Rating values: 1, 2, 3, 4, or 5 only
- Evaluated on Weighted F1-Score

---

## 🔄 Reproduction Steps

1. Ensure all dependencies installed: `pip install -r requirements.txt`
2. Run full training pipeline: `python src/train.py`
3. Generate test predictions: `python src/predict.py`
4. Submit `predictions.csv` with 1,424 predictions

---

**Status**: ✅ Complete & Ready for Submission  
**Last Updated**: January 2025
