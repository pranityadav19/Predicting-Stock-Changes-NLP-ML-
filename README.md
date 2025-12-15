# Predicting Overnight Stock Changes Post-Earnings
## Tomorrow's Price Hidden in Today's Language

**Project by:** Alina Hota, Pranit Yadav, Joshua Ringler, Connor Therrien, Justin Yang

---

## Overview

Ever notice how stocks sometimes tank even after beating earnings? Or rally despite missing targets? It turns out that **what management says matters as much as what they report**.

We built a machine learning system that predicts overnight stock movements by analyzing earnings call transcripts. Using **domain-specific NLP** and **stacked ensemble models**, we achieved **58.2% directional accuracy**—translating to a **1.67 Sharpe ratio** and **58.9% win rate** in simulated trading.

---

## What Makes This Different

Our approach combines three key elements:

1. **Guidance Surprise Metrics** - The gap between expected and reported revenues
2. **Loughran-McDonald Financial Dictionary** - Finance-specific sentiment analysis that understands words like 'liability' aren't negative in financial contexts
3. **Custom Word Importance Analysis** - Identifying which specific terms historically drive overnight returns

**Key Innovation:** Our stacked ensemble discovered that XGBoost and LightGBM have anti-correlated errors, using a **negative weight (-0.88)** on XGBoost to exploit this and boost performance.

---

## Data & Features

### Data Sources (WRDS)
- **Capital IQ:** 186,000+ earnings call transcripts (2006-2024) → filtered to 2,770 quality events
- **Compustat:** Reported revenues
- **CRSP:** Daily prices, returns, market cap, industry classifications
- **IBES:** Analyst revenue expectations

### Target Variable
`close_to_open_return` - The overnight gap from close (when earnings are released) to next morning's open

### NLP Features Engineered
- **Financial Sentiment:** Positive/negative counts, net sentiment, polarity (via Loughran-McDonald)
- **Management Tone:** Emotional language, uncertainty, evasion, modal verb usage
- **Readability:** Complexity scores (simpler language correlates with positive returns)
- **Custom Dictionary:** Words that historically drove the strongest market reactions

---

## Models & Architecture

### Two-Layer System

**Layer 1: Direction Classification**
- Logistic Regression (baseline): 56.0% accuracy
- XGBoost Classifier: 58.4% accuracy

**Layer 2: Magnitude Regression**
- Ridge Regression: 57.3% directional accuracy
- XGBoost Regressor: 53.5% directional accuracy
- LightGBM: 55.0% directional accuracy
- **Stacked Ensemble: 58.2% directional accuracy** ✓

### Stacked Ensemble Meta-Weights
- Ridge: 0.56 (56%)
- XGBoost: **-0.88** (negative 88%!)
- LightGBM: 1.12 (112%)

The negative XGBoost weight isn't a bug—it's the meta-learner exploiting anti-correlated errors between XGBoost and LightGBM.

---

## Results

| Model | MAE | RMSE | R² | Directional Accuracy |
|-------|-----|------|----|--------------------|
| **Stacked Ensemble** | 0.0601 | 0.0954 | 0.0340 | **58.2%** |
| Ridge | 0.0607 | 0.0959 | 0.0240 | 57.3% |
| LightGBM | 0.0610 | 0.0960 | 0.0217 | 55.0% |
| XGBoost | 0.0610 | 0.0989 | 0.0144 | 53.5% |
| Random Baseline | - | - | - | 50.0% |

**Trading Performance:** 1.67 Sharpe Ratio | 58.9% Win Rate

### The Honest Truth: Data Leakage Cost Us

Early on, we saw **74% accuracy**—amazing, right? Wrong. We had data leakage. After fixing:
- Strict temporal splits (train: 2007-2023, test: 2023-2024)
- No future-peeking features
- Proper cross-validation

Accuracy dropped to **58%**. That 16-point drop? The cost of doing it right. But 58% on truly unseen data is real alpha—8 points above random in one of the hardest prediction problems in finance.

---

## Key Findings

### What Worked
- **Loughran-McDonald dictionary** crushes generic sentiment tools
- **Clear, simple language** correlates with positive returns (r = -0.12 with complexity)
- **Ensemble meta-learning** extracted performance no single model could achieve

### What Didn't Work: Forward-Looking Confidence

We engineered a `forward_outlook_score` combining intensity, confidence, and certainty of future-oriented language. Research suggested this should predict returns.

**The problem:** It showed correlation (r = 0.15-0.20) but **hurt model performance** when included, dropping accuracy to 56%. Why? Our training set was full of performative optimism—everyone sounds confident on earnings calls regardless of actual conviction. The model couldn't distinguish genuine confidence from executive posturing.

**The lesson:** Correlation in exploratory analysis ≠ predictive power on unseen data.

---

## Lessons Learned

1. **Data leakage will destroy you** - We rebuilt our entire pipeline with strict temporal controls
2. **Domain-specific tools matter** - Finance dictionaries > generic NLP
3. **Ensemble methods can extract gains** - Meta-learning found error patterns no single model saw
4. **Directional accuracy > prediction error** - In trading, getting the sign right is what matters
5. **Kill your darlings** - Sometimes promising features hurt performance
6. **Clean data beats clever algorithms** - Database linking and timestamp verification took forever but was crucial

---

## Business Applications

### Trading Firms
58.2% directional accuracy with 1.67 Sharpe ratio offers a genuine statistical edge. Compute NLP features within minutes of transcript release for after-hours positioning.

### Risk Managers
Identify which holdings will gap overnight based on management language. Flag elevated risk even when headline numbers look fine.

### Sell-Side Analysts
Systematically prioritize which earnings calls deserve deep analysis. Flag inconsistencies between beats and hesitant language.

### Investor Relations
Optimize communication strategy. Clear language correlates with positive reactions; jargon-heavy, evasive language correlates with negative reactions.

---

## Future Work

1. **Upgrade to FinBERT** - Transformer models vs. dictionary-based NLP
2. **Split Management Remarks vs. Q&A** - Unscripted responses likely contain better signals
3. **Add Audio Analysis** - Vocal tone, hesitation, speaking pace
4. **Social Media Integration** - Real-time crowd interpretation signals
5. **Sector-Specific Models** - Tech companies talk differently than healthcare
6. **Extend Time Horizon** - Predict multi-day reactions, not just overnight

---

## References

1. Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35-65.

2. Price, S. M., et al. (2012). Earnings conference calls and stock returns: The incremental informativeness of textual tone. *Journal of Banking & Finance*, 36(4), 992-1011.

3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD*, 785-794.

---

*Graduate machine learning capstone project. Trading strategies discussed are for educational purposes. Past performance does not guarantee future results.*
