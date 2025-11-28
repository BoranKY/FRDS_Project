# Fake Review Detection using Fuzzy Logic (FRDS)

This repository contains our course project **“Fuzzy Rule-Based Detection System for Fake Online Reviews (FRDS)”**, developed for the **Social Computing** course.

The goal is to detect potentially fake product reviews on e-commerce platforms using an **interpretable fuzzy logic system** instead of a black-box machine learning model.

---

## Project Overview

Online marketplaces increasingly suffer from fake or misleading reviews that distort user perception and harm platform trust.  
FRDS addresses this problem by mapping a small set of **linguistic features** into a single, human-interpretable **probability of deception**.

Key ideas:

- Use *linguistic cues* extracted from review text (sentiment, subjectivity, readability, length)
- Add a simple *rating–text tone mismatch* signal
- Feed these into a **Mamdani-type fuzzy inference system**
- Output a probability `P(fake) ∈ [0, 1]` plus an interpretable traffic-light label:

  - **Green** (≤ 0.30) – likely genuine  
  - **Amber** (0.30–0.60) – borderline, needs manual check  
  - **Red** (> 0.60) – likely fake

FRDS prioritises **transparency and low data requirements** over raw accuracy.

---

## Dataset

We use the public **Fake Reviews Dataset (Kaggle, CC BY 4.0)**:

- ~40k English Amazon product reviews
- Fields: review text, 1–5 star rating, product category, binary authenticity label
- Labels: computer-generated (fake) vs. genuine

Reviews are loaded into a Pandas DataFrame; basic text normalisation (lower-casing, regex cleaning, whitespace collapse) is applied.

---

## Feature Set

Each review is mapped to a 5-dimensional feature vector:

1. **Sentiment polarity** (scaled to [0, 1])  
2. **Subjectivity score**  
3. **Readability ease** (Flesch-based, normalised to [0, 1])  
4. **Relative review length** (scaled and capped at 300 words)  
5. **Rating–text tone mismatch flag** (0 = consistent, 1 = suspicious mismatch)

These features are intentionally compact and interpretable, making them suitable for fuzzy reasoning.

---

## Fuzzy Logic System

FRDS employs a **Mamdani-type fuzzy inference system** with:

- **Input membership functions**  
  - For each feature: three triangular sets – *low*, *medium*, *high* on [0, 1]
- **Output variable**  
  - `fake_probability` with the same (*low*, *medium*, *high*) triangular family
- **Rule base**  
  - 10 hand-crafted IF–THEN rules, e.g.:

    - IF sentiment is **high** AND subjectivity is **high** AND readability is **low**  
      THEN `fake_probability` is **high**  
    - IF sentiment–rating **mismatch flag** is 1  
      THEN `fake_probability` is **high**

- **Inference & defuzzification**  
  - Min for AND, max for OR  
  - Centroid method to obtain a scalar `P(fake)`

A tuned threshold of **0.20** is used to map `P(fake)` to a binary fake/genuine decision.

---

## Evaluation

We compare FRDS against three traditional baselines trained on TF-IDF vectors:

- **Logistic Regression (LR)** – F1 ≈ 0.87  
- **Linear SVM** – F1 ≈ 0.87  
- **Random Forest (RF)** – F1 ≈ 0.86  
- **FRDS (fuzzy system)** – F1 ≈ 0.64  

Although FRDS underperforms the ML baselines in F1-score, it offers:

- Full **explainability** (rules and membership functions are human-readable)  
- **No training** or hyperparameter tuning  
- Easy deployment as a lightweight, rule-based moderation tool

---



## Installation & Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/frds-fake-review-fuzzy.git
cd frds-fake-review-fuzzy
