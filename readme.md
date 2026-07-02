# ContextSarcasmDetector

> **Melampaui Klasifikasi Biner: Analisis Deteksi Sarkasme Sadar Konteks pada Hiperbola dan Pertanyaan Retoris dalam Diskusi Daring**
>
> *Beyond Binary Classification: An Analysis of Context-Aware Sarcasm Detection on Hyperbole and Rhetorical Questions in Online Discourse*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JordanCodeGit/ContextSarcasmDetector/blob/main/FinalProjectAI_2311102139_2311102179.ipynb)
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Model](https://img.shields.io/badge/model-Linear%20SVM-EE4C2C?style=flat)
![Accuracy](https://img.shields.io/badge/accuracy-71.5%25-brightgreen?style=flat)

This repository evaluates the limitations of machine learning in detecting sarcasm in online forums. Rather than treating sarcasm as a single binary class, we ask a sharper question: **does the *linguistic type* of sarcasm change how well an AI can detect it?**

---

## 📌 Abstract

Conventional sentiment analysis fails catastrophically on sarcasm because sarcasm inverts polarity — using positive words to convey a negative message (e.g., *"Great job crashing the server"*). This study evaluates the limits of standard ML on this problem using the **Sarcasm Corpus V2** (Oraby et al.).

We train a **Linear SVM** on **9,386 balanced samples** spanning three categories — General Sarcasm, **Hyperbole**, and **Rhetorical Questions** — and compare it against a Naive Bayes baseline. The model reaches an overall accuracy of **71.5%**, but hides a significant performance gap between categories: it detects **Rhetorical Questions well (73.3%)** yet **struggles with Hyperbole (65.6%)**.

The finding confirms our hypothesis: **structural cues** (like punctuation) are easy for a model to learn, but **semantic sarcasm** built on exaggeration requires *world knowledge* that standard NLP models do not possess.

---

## 🎯 Problem Statement

Sentiment analysis is a standard tool for companies to gauge customer feedback, but sarcasm is a critical failure point because standard algorithms rely on **keywords** rather than **context**.

We hypothesize that model performance is **not uniform** across sarcasm types. Specifically, we investigate whether **Structural Sarcasm** (Rhetorical Questions) is easier to detect than **Semantic Sarcasm** (Hyperbole).

---

## 🗂️ Dataset

**Sarcasm Corpus V2** (Oraby et al., 2016) — forum debate posts, perfectly balanced between sarcastic and non-sarcastic examples.

| File | Category | Sarcasm Device |
|------|----------|----------------|
| `GEN-sarc-notsarc.csv` | General | Mixed / general sarcasm |
| `HYP-sarc-notsarc.csv` | Hyperbole | Exaggeration (semantic) |
| `RQ-sarc-notsarc.csv`  | Rhetorical Question | Question framing (structural) |

- **Total samples:** 9,386 (balanced 50/50 across the `sarc` / `notsarc` label)
- **Split:** 80% train (7,508) / 20% test (1,878), stratified by label

> ⚠️ The three CSV files are **not** included in this repo. Download the [Sarcasm Corpus V2](https://nlds.soe.ucsc.edu/sarcasm2) and place `GEN-sarc-notsarc.csv`, `HYP-sarc-notsarc.csv`, and `RQ-sarc-notsarc.csv` in the notebook's working directory (or upload them to your Colab session).

---

## 🔬 Methodology

Our pipeline is designed to force the model to learn from the *sarcastic speaker* and to guarantee statistical validity.

1. **Surgical Quote Removal** — Forum replies often quote the previous argument with `[quote]...[/quote]` tags (including malformed variants). A dedicated regex routine strips these out so the model trains **only on the user's actual response**, not the text they are replying to.

2. **Strict Train/Test Split *Before* Feature Extraction** — We split the data before fitting the vectorizer and scaler to prevent **data leakage** (TF-IDF and MinMax scaling are `fit` on train, then `transform` on test only).

3. **Hybrid Feature Engineering** — We combine lexical and paralinguistic signals:
   - **TF-IDF** — unigrams + bigrams, top 5,000 features, English stopwords removed.
   - **Punctuation Density** — counts of `!` and `?` as tonal / sarcastic indicators.
   - **Capitalization Ratio** — proxy for "shouting" / emotional intensity.

4. **Model Comparison** — **Naive Bayes** (baseline) vs. **Linear SVM** (`class_weight='balanced'`) to test whether a hyperplane-based approach better separates complex sarcastic nuance.

---

## 📊 Results

### Overall Accuracy

| Model | Accuracy |
|-------|----------|
| Naive Bayes (baseline) | **72.8%** |
| Linear SVM (advanced)  | **71.5%** |

**SVM Classification Report**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Non-Sarcastic | 0.71 | 0.73 | 0.72 | 939 |
| Sarcastic     | 0.72 | 0.70 | 0.71 | 939 |
| **Accuracy**  |      |      | **0.71** | 1,878 |

The confusion matrix shows a good balance between precision and recall — the model is **not biased** toward a single class, even when trained on noisy forum text.

### Accuracy by Linguistic Category — *the key finding*

| Sarcasm Type | Accuracy |
|--------------|----------|
| 🟢 Rhetorical Question (structural) | **73.3%** |
| 🔵 General | 71.9% |
| 🟠 Hyperbole (semantic) | **65.6%** |

Rhetorical Questions score highest thanks to distinctive punctuation patterns. Hyperbole lags by ~8%, revealing the model's difficulty with meaning-based (semantic-heavy) sarcasm.

### The Confidence Gap

| Input Sentence | Type | Model Score |
|----------------|------|-------------|
| *"Do you really think I was born yesterday?"* | Rhetorical Question | **+0.88** (very high) |
| *"This download is taking 500 years."* | Hyperbole | **+0.30** (low / uncertain) |
| *"The service was slow and the food was cold."* | Normal complaint | **−0.25** (correctly not sarcastic) |

The model is highly confident on structural patterns and correctly rejects plain complaints — but on hyperbole its score collapses into the "uncertain zone." Without world knowledge, it cannot validate logic-based sarcasm like the phrase *"500 years."*

---

## 🧭 Conclusion

The core challenge in sarcasm detection is **not vocabulary variation — it is context dependency.**

1. **Structural advantage** — Structural cues (question marks, sentence openers) are reliable indicators, validating our punctuation-based feature engineering.
2. **The context gap** — Detecting exaggerated meaning requires *world knowledge* (e.g., knowing a 500-year download is impossible). Standard NLP cannot bridge this from surface text alone.
3. **Model choice** — Naive Bayes' independence assumption proved surprisingly robust to noisy forum text, slightly edging out the SVM.

**Future work:** integrate **contextual embeddings (e.g., BERT)** to bridge the world-knowledge gap and better handle semantic sarcasm.

---

## 🚀 Getting Started

### Run in the cloud (recommended)
Click the **Open In Colab** badge above, upload the three dataset CSVs, and run all cells.

### Run locally

```bash
git clone https://github.com/JordanCodeGit/ContextSarcasmDetector.git
cd ContextSarcasmDetector

pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# Place GEN/HYP/RQ CSV files in this folder, then:
jupyter notebook FinalProjectAI_2311102139_2311102179.ipynb
```

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`

---

## 📁 Repository Structure

```
ContextSarcasmDetector/
├── FinalProjectAI_2311102139_2311102179.ipynb   # Full analysis notebook
├── Poster.png                                    # Research poster summary
└── readme.md                                     # This file
```

The notebook is organized into 8 segments: imports & data loading → aggressive cleaning → leakage-free feature engineering → model training & comparison → error analysis by type → confusion matrix → real-world manual testing → conclusion.

---

## 🖼️ Research Poster

A one-page visual summary of the study is available in [`Poster.png`](Poster.png).

---

## 👥 Authors

| Name | Student ID |
|------|-----------|
| **Jordan Angkawijaya** | 2311102139 |
| **Axandio Biyanatul Lizan** | 2311102179 |

Final Project — Artificial Intelligence, Telkom University.

---

## 📚 References

1. Oraby, S., et al. (2016). *Creating and Characterizing a Diverse Corpus of Sarcasm in Dialogue.* LREC.
2. Justo, R., et al. (2014). *Extracting Sarcasm from Twitter: A Feature-Based Approach.*
