## ⚖️ Legal Disclaimer

This repository is an unofficial, non-commercial, and purely educational data analysis project. 
All character names, game titles, and stat profiles are the intellectual property of Bandai Namco 
Entertainment and their respective creators. This project is not affiliated with, endorsed by, 
or associated with any official entity, and its usage falls under Fair Use guidelines for 
academic data science research.

All character data and stats were taken from 'dblegends.net'

# Dragon Ball Legends: Adaptive Powercreep Predictive Engine

An end-to-end, multi-paradigm data science pipeline that analyzes and predicts mobile game powercreep anomalies. The system evaluates character stat configurations against a historical baseline using a hybrid approach: **Supervised Gradient Boosting** for data-rich rarity tiers and **Unsupervised Isolation Forests** for smaller data tranches.

> **Important Scope Limitation**: This engine evaluates powercreep based purely on **raw, baseline numerical character stats**. It does not take dynamic in-game kits, unique abilities, passive traits, or card effects into account. It serves strictly as a tool to measure base statistical inflation over time.

##  Key Architectural Features

### 1. Hybrid Machine Learning Pipeline
Games scale differently across character tiers. The engine dynamically checks the density of each `rarity_tier` and automatically selects the optimal algorithmic path:
* **Supervised Tier Approach ($\ge 50$ units)**: Utilizes a `GradientBoostingClassifier`. It handles gacha dataset imbalance by applying **SMOTE (Synthetic Minority Over-sampling Technique)** to synthetically scale up the true powercrept minority class for robust training.
* **Unsupervised Fallback Tier ($< 50$ units)**: Automatically falls back to an `IsolationForest`, dynamically tuning its `contamination` rate to the exact statistical proportion of the target tier's outliers.

### 2. Domain-Specific Feature Engineering
Instead of passing raw features independently, the engine transforms the data space into 4 synergistic interaction features to capture complex character build profiles:
* **Composite Power Score**: A weighted domain formula evaluating a character's complete overall value across all 7 base stats.
* **Offense/Defense & Offense/HP Ratios**: Captures mathematical stat optimization efficiency.
* **Crit × Offense & Ki × Offense Interactions**: Isolates hyper-aggressive utility threats.

### 3. Three-State Verdict Engine
Unlike basic binary anomaly models, our model maps a multidimensional anomaly signal alongside a strict 1D **Z-Score Directional Cutoff** to predict a precise three-state meta classification:
*  `POWERCREPT` ($Z > +0.5$): Anomalously strong; pushes past ecosystem boundaries and warps the baseline stat ceiling.
*  `BALANCED` (Within Tolerance): Fits cleanly within historical tier parameters; safe to release.
*  `UNDERPOWERED` ($Z < -0.5$): Anomalously weak; fails to make any competitive statistical impact.

---

##  Model Performance & Validation Dashboard

The evaluation system uses a strict **Three-State Verdict Validation Matrix**. Supervised tiers are strictly evaluated on a held-out **20% stratified test split**, ensuring zero data leakage, while underpowered anomalies are isolated so they never trigger false alarms in precision metrics.

```text
=======================================================
     ENGINE ACCURACY & METRICS REPORT    
=======================================================
 Global Classification Accuracy: 93.66%

--- CONFUSION MATRIX (Powercrept vs Everything Else) ---
• True Negatives  (Correctly labelled Balanced)    : 625
• False Positives (Wrong powercreep flags)         : 35
• False Negatives (Missed Powercreep units)        : 37
• True Positives  (Correctly caught Powercreep)    : 34

--- PRECISION, RECALL, AND F1 REPORT ---
                         precision    recall  f1-score   support
Balanced / Underpowered       0.94      0.95      0.95       660
        Powercrept       0.49      0.48      0.49        71
=======================================================
