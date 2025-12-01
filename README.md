# Predicting Overnight Stock Changes from Earnings Call Language

**Course:** Advanced Machine Learning  
**Team:** Alina Hota, Pranit Yadav, Justin Yang, Connor Therrien, Joshua Ringler  
**Tagline:** *Tomorrow’s price signals hidden in today’s language*   

---

## 1. Project Overview

Earnings used to move on numbers alone. Now, language in earnings calls often matters more than the headline beats or misses.

This project turns earnings-call transcripts into numerical features and uses them to predict the **overnight stock return from close-to-next-open**:

\[
\text{close\_to\_open\_return} = \frac{\text{Open}_{t+1} - \text{Close}_t}{\text{Close}_t}
\]

The goal is to forecast:

- **Direction:** Will the stock go up or down overnight?
- **Magnitude:** How large is the move likely to be?
- **Drivers:** Which aspects of management language and guidance are most predictive?

This repository combines **traditional market/financial data** with **finance-specific natural language processing** to build and evaluate several regression and meta-learning models.   

---

## 2. Data Sources

All data comes from Wharton Research Data Services (WRDS) feeds and related platforms:   

- **Capital IQ (CIQ):**  
  Earnings call transcripts and metadata.
- **Compustat:**  
  Fundamental accounting data and reported revenues.
- **CRSP (Center for Research in Security Prices):**  
  Daily stock prices, returns, market capitalization, and Fama–French 12 industry classifications.
- **IBES (Institutional Brokers’ Estimate System):**  
  Analyst revenue guidance (consensus estimates).

### Core Target

- `close_to_open_return`: next-day open minus current close, scaled by current close.  
  This single target embeds both direction and magnitude of the overnight reaction.   

---

## 3. Feature Engineering

### 3.1 Guidance Surprise and Firm Context

From structured data:   

- **Guidance Surprise %**
  - Difference between actual reported revenue and IBES consensus estimates, scaled by expectation.
- **Industry (Fama–French 12)**
  - One-hot encoded sector dummies (e.g., Utilities, Healthcare, Finance).
- **Market Capitalization**
  - Size of the firm at the earnings date.
- **Word Count**
  - Total transcript length as a proxy for information load.

### 3.2 NLP: Finance-Specific Sentiment (Loughran–McDonald)

We use the **Loughran–McDonald financial sentiment dictionary** rather than generic sentiment tools, to handle words like “liability” or “debt” correctly in a financial context.  

Key categories:

- Positive / Negative
- Litigious
- Uncertainty
- Strong modal / Weak modal
- Constraining

From this dictionary we build features such as:   

- `lm_positive_count`, `lm_negative_count`
- `net_sentiment` (positive − negative)
- `sentiment_polarity` (direction and intensity of sentiment)

### 3.3 Management Tone and Communication Style

To capture how management speaks (not just what they say):   

- `emotional_word_ratio` – emotional vs. factual language  
- `uncertainty_ratio` – hedging and doubt  
- `evasion_score` – defensive language patterns  
- `modal_verb_ratio` – frequency of “could,” “might,” “should”  

These are designed to reflect CEO confidence and clarity; higher certainty and less evasion tend to correlate with better market reactions.

### 3.4 Forward-Looking Language

We explicitly engineer forward-looking signals:

- `forward_intensity` – density of future-focused discussion  
- `forward_confidence_score` – optimistic vs. cautious future language  
- `forward_certainty_score` – definiteness of forward statements  

These roll into a composite signal:

\[
\text{forward\_outlook\_score} =
0.5 \times \text{Intensity} +
0.3 \times \text{Confidence} +
0.2 \times \text{Certainty}
\]

We find that **forward-looking confidence has the strongest correlation with overnight returns (r ≈ 0.15–0.20)**, while more complex/readable transcripts correlate negatively with returns (r ≈ −0.12).   

---

## 4. Data Integrity and Leakage Control

To ensure valid modeling:  

- **Entity linkage:**  
  CIQ company identifiers are mapped to Compustat `GVKEY`, then to CRSP `PERMNO`.
- **Time-series consistency:**  
  Each overnight return uses the correct close and next-open prices; only calls reported after the close are kept.
- **Data leakage prevention:**  
  Early versions of the NLP pipeline leaked future information. The final pipeline uses:
  - A strict **70–30 train–test split** at the firm-date level for any learned features.  
  - Feature engineering procedures constrained not to peek at test outcomes.

---

## 5. Modeling Approach

We treat the problem as **regression** on `close_to_open_return`, and later convert predictions to direction (up/down) for trading-relevant metrics.   

### 5.1 Models

- **Ridge Regression (baseline)**
- **XGBoost Regressor**
- **LightGBM Regressor**
- **Stacked Ensemble (meta-learning)**
  - Meta-model learns how to weight Ridge, XGBoost, LightGBM, and an auxiliary XGBoost classifier.

### 5.2 Key Metric: Directional Accuracy

While Mean Absolute Error and Root Mean Squared Error are tracked, the primary business metric is **Directional Accuracy**: the percentage of times the sign of the predicted return matches the sign of the actual return.  

Implementation sketch:

```python
y_true_direction = (actual_returns > 0)
y_pred_direction = (predicted_returns > 0)

directional_accuracy = accuracy_score(y_true_direction, y_pred_direction)
```

## 6. Results (Summary)

### On the held-out test set:

| Model | MAE | RMSE | R² | Directional Accuracy |
|-------|-----|------|----|--------------------|
| Ridge Regression | 0.0599 | 0.0961 | 0.0699 | 0.6923 |
| Stacked Ensemble | 0.0625 | 0.0966 | 0.0599 | 0.6803 |
| LightGBM | 0.0652 | 0.0983 | 0.0268 | 0.6731 |
| XGBoost | 0.0653 | 0.0978 | 0.0367 | 0.6635 |

### Additional insights:

- Forward-looking confidence is the single strongest NLP predictor of overnight return.
- Readability complexity (more complex wording) is mildly negatively associated with returns.
- A composite NLP score improves prediction accuracy by about **7 percent** over financial/quantitative features alone.
- In a simple portfolio simulation, directional accuracy translates to approximately a **74.7 percent trade win rate**.

## 8. Implementation Guide

### Typical Python libraries used:
- pandas, numpy, scikit-learn
- xgboost, lightgbm
- matplotlib / seaborn
- nltk or similar for tokenization
- Any custom WRDS connection utilities you use

You must also have access to:
- **WRDS credentials** for pulling Compustat, CRSP, IBES, and CIQ data.
- A local or remote location to store the extracted tables.

### 8.2 Running the Pipeline

#### 1. Obtain Data
- Use WRDS or your institution's access to download:
  - Earnings call transcripts and metadata (CIQ).
  - Compustat revenues.
  - CRSP prices and returns.
  - IBES consensus revenue estimates.

#### 2. Run Data + NLP Notebook
- Open `Advanced_ML_Pulling_Data_+_NLP.ipynb`.
- Execute cells in order:
  - Connect to WRDS / read in flat files.
  - Clean and merge (CIQ → GVKEY → PERMNO).
  - Filter to after-close earnings events.
  - Engineer guidance surprise and firm-level features.
  - Build NLP features (Loughran-McDonald counts, tone, forward outlook).

#### 3. Train Models
- Use additional modeling:
  - Split into training and test sets (no leakage).
  - Fit Ridge, XGBoost, LightGBM.
  - Train stacked ensemble meta-learner.
  - Evaluate MAE, RMSE, R², and directional accuracy.

#### 4. Visualize and Interpret
- Reproduce charts from the presentation:
  - Correlation bar charts for different NLP feature groups.
  - Distribution of NLP composite score across calls.
  - Regression metrics by model.
  - Stacked ensemble weight breakdown.

## 9. Interpretation and Use Cases

### Practical applications:

- **Earnings-night trading:**
  - Use directional predictions to decide which names to overnight or hedge immediately after calls.

- **Analyst prioritization:**
  - Rank upcoming calls by predicted impact magnitude to decide where to spend human analyst attention.

- **Risk management:**
  - Stress-test portfolios around earnings seasons using predicted volatility from the model.

This project is a proof-of-concept for combining interpretable finance-specific NLP with modern gradient boosting and ensemble techniques.

## 10. Limitations and Future Directions

As highlighted in the presentation:

### 1. Enhanced NLP
- Replace bag-of-words with transformer models (BERT, FinBERT).
- Use live sentiment for prepared remarks versus Q&A.
- Use topic modeling to identify themes (pricing power, guidance, macro risk).

### 2. Real-World Implementation
- Incorporate transaction costs, slippage, and turnover.
- Add risk management (position sizing, stop-loss logic).
- Calibrate thresholds for when to trade based on prediction confidence.

### 3. Robust Validation
- Walk-forward evaluation by year to mimic real-time deployment.
- Compare bull vs. bear market performance.
- Train sector-specific models where language norms differ.

### 4. Multi-Modal Expansion
- Layer in audio features (tone, hesitations).
- Track post-call analyst estimate revisions.
- Incorporate social media sentiment in the immediate aftermath of calls.
