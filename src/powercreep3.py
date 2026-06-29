import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# LOAD DATA


try:
    df = pd.read_csv('cleaned_dbl_stats.csv')
    df['character_name'] = df['character_name'].str.strip()
except FileNotFoundError:
    print(" csv not found. Make sure it's in the same folder.")
    exit()

BASE_FEATURES = [
    'base_strike_attack', 'base_blast_attack', 'base_hp',
    'base_strike_defence', 'base_blast_defence', 'critical_rate', 'ki_recovery'
]

STAT_LABELS = {
    'base_strike_attack':  'Strike ATK  ',
    'base_blast_attack':   'Blast ATK   ',
    'base_hp':             'HP          ',
    'base_strike_defence': 'Strike DEF  ',
    'base_blast_defence':  'Blast DEF   ',
    'critical_rate':       'Crit Rate   ',
    'ki_recovery':         'Ki Recovery '
}

# Tiers with fewer rows than this use IsolationForest instead of GradientBoosting
MIN_ROWS_FOR_SUPERVISED = 50

#  Thresholds used:
#    GradientBoosting  → powercreep_proba > 0.5 means the model thinks
#                        it's powercrept; then z-score decides direction
#    IsolationForest   → raw -1 prediction means anomaly; z-score decides
#                        whether that anomaly is above or below the mean
#
#  Z-score direction cutoffs:
#    z > +0.5  → unit is genuinely above average  → POWERCREPT
#    z < -0.5  → unit is genuinely below average  → UNDERPOWERED
#    between   → minor outlier, not meaningful     → BALANCED


POWERCREEP_Z_THRESHOLD   =  0.5   # above this z-score = strong enough to flag
UNDERPOWERED_Z_THRESHOLD = -0.5   # below this z-score = weak enough to flag

def get_verdict(model_flags_anomaly, z_score):
    
    if not model_flags_anomaly:
        return 'BALANCED'

    if z_score > POWERCREEP_Z_THRESHOLD:
        return 'POWERCREPT'
    elif z_score < UNDERPOWERED_Z_THRESHOLD:
        return 'UNDERPOWERED'
    else:
        
        return 'BALANCED'

#  COMPOSITE POWER SCORE


def composite_score(row_or_dict):
    """Single number representing a unit's overall power across all stats."""
    return (
        row_or_dict['base_strike_attack'] * 0.25 +
        row_or_dict['base_blast_attack']  * 0.25 +
        row_or_dict['base_hp']            * 0.025 +
        row_or_dict['base_strike_defence']* 0.15  +
        row_or_dict['base_blast_defence'] * 0.15  +
        row_or_dict['critical_rate']      * 50000 +
        row_or_dict['ki_recovery']        * 50000
    )

df['composite_power'] = df.apply(composite_score, axis=1)
df['offensive_power'] = df['base_strike_attack'] + df['base_blast_attack']

#  RATIO & INTERACTION FEATURES

def add_ratio_features(source_df):
    """
    Adds four derived feature columns that capture relationships between stats.
    Works on both a full DataFrame and a single-unit dict.
    """
    d = source_df.copy() if isinstance(source_df, pd.DataFrame) else dict(source_df)

    off  = d['base_strike_attack'] + d['base_blast_attack']
    def_ = d['base_strike_defence'] + d['base_blast_defence']
    hp   = d['base_hp']
    crit = d['critical_rate']
    ki   = d['ki_recovery']

    d['offence_defence_ratio'] = off / (def_  + 1)
    d['offence_hp_ratio']      = off / (hp    + 1)
    d['crit_x_offence']        = crit * off
    d['ki_x_offence']          = ki   * off

    return d

df = add_ratio_features(df)

ALL_FEATURES = BASE_FEATURES + [
    'offence_defence_ratio', 'offence_hp_ratio',
    'crit_x_offence', 'ki_x_offence'
]


df['true_ground_truth'] = 0
for tier, group in df.groupby('rarity_tier'):
    top_threshold    = group['composite_power'].quantile(0.90)
    df.loc[group.index, 'true_ground_truth'] = (
        group['composite_power'] > top_threshold
    ).astype(int)



tier_models = {}
all_preds   = pd.Series(index=df.index, dtype=object)
all_truth   = pd.Series(index=df.index, dtype=object)

for tier, group in df.groupby('rarity_tier'):

    X        = group[ALL_FEATURES].values
    y        = group['true_ground_truth'].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pre-compute per-tier composite stats needed for z-score direction
    tier_mean = group['composite_power'].mean()
    tier_std  = group['composite_power'].std()

    if len(group) >= MIN_ROWS_FOR_SUPERVISED:
        # supervised GradientBoosting

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_scaled, y, group.index,
            test_size=0.2, random_state=42, stratify=y
        )

        n_minority  = y_train.sum()
        k_neighbors = min(5, max(1, n_minority - 1))
        sm          = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=42
        )
        model.fit(X_train_res, y_train_res)

        # Evaluate on test set using the three-state system so the accuracy
        # report reflects real-world behaviour, not just binary classification
        preds_proba = model.predict_proba(X_test)[:, 1]
        test_composites = group.loc[idx_test, 'composite_power'].values
        test_z_scores   = (test_composites - tier_mean) / (tier_std + 1e-9)

        preds_verdict = np.array([
            get_verdict(p > 0.5, z)
            for p, z in zip(preds_proba, test_z_scores)
        ])
        # For confusion matrix: POWERCREPT=1, anything else=0
        all_preds.loc[idx_test] = (preds_verdict == 'POWERCREPT').astype(int)
        all_truth.loc[idx_test] = y_test

        tier_models[tier] = {
            'type':      'supervised',
            'scaler':    scaler,
            'model':     model,
            'df':        group,
            'tier_mean': tier_mean,
            'tier_std':  tier_std,
        }

    else:
        # IsolationForest fallback

        contamination = float(np.clip(y.mean(), 0.01, 0.49))
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X_scaled)

        raw_preds    = model.predict(X_scaled)          # -1 or 1
        composites   = group['composite_power'].values
        z_scores     = (composites - tier_mean) / (tier_std + 1e-9)

        preds_verdict = np.array([
            get_verdict(r == -1, z)
            for r, z in zip(raw_preds, z_scores)
        ])
        all_preds.loc[group.index] = (preds_verdict == 'POWERCREPT').astype(int)
        all_truth.loc[group.index] = y

        tier_models[tier] = {
            'type':      'isolation_forest',
            'scaler':    scaler,
            'model':     model,
            'df':        group,
            'tier_mean': tier_mean,
            'tier_std':  tier_std,
        }

#  ACCURACY REPORT

eval_mask   = all_preds.notna() & all_truth.notna()
y_eval_pred = all_preds[eval_mask].astype(int)
y_eval_true = all_truth[eval_mask].astype(int)

global_accuracy = accuracy_score(y_eval_true, y_eval_pred)
conf_matrix     = confusion_matrix(y_eval_true, y_eval_pred)

print("=======================================================")
print("     ENGINE ACCURACY & METRICS REPORT    ")
print("=======================================================")
print(f" Global Classification Accuracy: {global_accuracy * 100:.2f}%")
print(" (Supervised tiers on held-out 20% | IsolationForest on full tier)")
print(" (Three-state verdicts applied - UNDERPOWERED units never")
print("  counted as false positives for powercreep)\n")

for tier, bucket in sorted(tier_models.items()):
    n     = len(bucket['df'])
    mtype = "GradientBoosting + SMOTE" if bucket['type'] == 'supervised' else "IsolationForest (fallback)"
    print(f"   {tier:<10} ({n:>3} units)  →  {mtype}")

print()
print("--- CONFUSION MATRIX (Powercrept vs Everything Else) ---")
print(f" True Negatives  (Correctly labelled Balanced)    : {conf_matrix[0][0]}")
print(f" False Positives (Wrong powercreep flags)         : {conf_matrix[0][1]}")
print(f" False Negatives (Missed Powercreep units)        : {conf_matrix[1][0]}")
print(f" True Positives  (Correctly caught Powercreep)    : {conf_matrix[1][1]}\n")

print("--- PRECISION, RECALL, F1 ---")
print(classification_report(
    y_eval_true, y_eval_pred,
    target_names=['Balanced / Underpowered', 'Powercrept'],
    zero_division=0
))
print("=======================================================\n")

# HELPERS


def _stat_bar(value, series):
    """Visual █░ progress bar + percentile rank for one stat vs. a group."""
    pct    = (series < value).mean() * 100
    filled = int(pct / 10)
    bar    = '█' * filled + '░' * (10 - filled)
    return bar, pct


def analyze_bulk(stats, tier_df):
    """
    Standalone bulk check - catches tanky outliers the ML model might miss,
    and also flags units that are suspiciously fragile for their tier.
    """
    unit_bulk = stats['base_hp'] + stats['base_strike_defence'] + stats['base_blast_defence']
    tier_bulk = tier_df['base_hp'] + tier_df['base_strike_defence'] + tier_df['base_blast_defence']
    bar, pct  = _stat_bar(unit_bulk, tier_bulk)

    print(f"\n   BULK CHECK (within tier) ─────────────────────────────")
    print(f"  │  Combined Bulk  : {unit_bulk:>12,}                           │")
    print(f"  │  Tier Percentile: {bar}  {pct:5.1f}th            │")
    if pct > 90:
        print("  │    BULK FLAG    - extreme survivability, tank-tier outlier │")
    elif pct > 75:
        print("  │    ABOVE AVERAGE - bulk is high but within tolerance        │")
    elif pct < 10:
        print("  │    FRAGILE FLAG  - unusually low bulk for this tier         │")
    elif pct < 25:
        print("  │    BELOW AVERAGE - bulk is low but within tolerance          │")
    else:
        print("  │     NORMAL        - bulk sits within tier norms               │")
    print("  ────────────────────────────────────────────────────────────")

#  VERDICT DISPLAY
#  Centralised so the same formatting is used in both
#  single predictions and bulk CSV output printing.

VERDICT_LINES = {
    'POWERCREPT': [
        "  STATUS:   SEVERE POWERCREEP RISK DETECTED",
        "  This unit is anomalously strong for its tier.",
        "  Releasing it will displace current top units and force a meta shift.",
    ],
    'UNDERPOWERED': [
        "  STATUS:   UNDERPOWERED - WILL NOT IMPACT META",
        "  This unit is anomalously weak for its tier.",
        "  It will be immediately outclassed and won't change the competitive landscape.",
    ],
    'BALANCED': [
        "  STATUS:  BALANCED - SUSTAINABLE PROFILE",
        "  Stats fit comfortably within the tier's current range.",
        "  This unit can coexist with the meta without warping it.",
    ],
}

# PREDICTION FUNCTION

def predict_powercreep(name, theoretical_stats, tier='LEGEND'):
    """
    Predicts the meta impact of a hypothetical unit using a three-state verdict:
      POWERCREPT   — too strong, will break the meta
      UNDERPOWERED — too weak, won't affect the meta
      BALANCED     — fits the current tier range

    Parameters
    ----------
    name              : str   — character name for the report header
    theoretical_stats : dict  — must contain all 7 keys in BASE_FEATURES
    tier              : str   — LEGEND / ULTRA / SPARKING / EXTREME / HERO
    """

    tier = tier.upper()
    if tier not in tier_models:
        print(f"[ERROR] Unknown tier '{tier}'. Choose from: {list(tier_models.keys())}")
        return

    bucket    = tier_models[tier]
    tier_df   = bucket['df']
    tier_mean = bucket['tier_mean']
    tier_std  = bucket['tier_std']

    enriched_stats = add_ratio_features(theoretical_stats)
    new_row        = pd.DataFrame([enriched_stats])[ALL_FEATURES]
    new_scaled     = bucket['scaler'].transform(new_row.values)

    # Composite power & z-score 
    new_composite = composite_score(enriched_stats)
    new_offense   = theoretical_stats['base_strike_attack'] + theoretical_stats['base_blast_attack']
    z_score       = (new_composite - tier_mean) / (tier_std + 1e-9)

    # ML signal + direction → three-state verdict 
    if bucket['type'] == 'supervised':
        powercreep_proba = bucket['model'].predict_proba(new_scaled)[0][1]
        is_anomaly       = powercreep_proba > 0.5
        severity_score   = powercreep_proba
        severity_label   = "Higher = more likely powercrept"
    else:
        raw_pred         = bucket['model'].predict(new_scaled)[0]
        is_anomaly       = (raw_pred == -1)
        severity_score   = bucket['model'].score_samples(new_scaled)[0]
        severity_label   = "Lower = more disruptive"

    verdict   = get_verdict(is_anomaly, z_score)
    model_tag = "GradientBoosting" if bucket['type'] == 'supervised' else "IsolationForest"

    #  Print report
    print("=" * 64)
    print(f"  POWERCREEP SIMULATION: {name.upper()}")
    print(f"  Tier: {tier}   |   Engine: {model_tag}")
    print("=" * 64)

    print(f"\n  Combined Offence   : {new_offense:>12,}")
    print(f"  Composite Power    : {new_composite:>12,.1f}")
    print(f"  Tier Displacement  : {z_score:>+.2f} σ from {tier} mean")
    print(f"  (positive = stronger than average, negative = weaker)\n")

    print("  --- STAT PERCENTILE BREAKDOWN (vs tier) ---")
    for feat, label in STAT_LABELS.items():
        bar, pct = _stat_bar(theoretical_stats[feat], tier_df[feat])
        if pct > 90:
            flag = "   HIGH"
        elif pct < 10:
            flag = "   LOW"
        else:
            flag = ""
        print(f"   {label}: {bar}  {pct:5.1f}th{flag}")

    ratio_cols = {
        'offence_defence_ratio': 'Off/Def Ratio',
        'offence_hp_ratio':      'Off/HP  Ratio',
        'crit_x_offence':        'Crit×Offence ',
        'ki_x_offence':          'Ki×Offence   '
    }
    print("\n  --- INTERACTION FEATURE PERCENTILES (vs tier) ---")
    for feat, label in ratio_cols.items():
        bar, pct = _stat_bar(enriched_stats[feat], tier_df[feat])
        if pct > 90:
            flag = "   HIGH"
        elif pct < 10:
            flag = "   LOW"
        else:
            flag = ""
        print(f"   {label}: {bar}  {pct:5.1f}th{flag}")

    analyze_bulk(theoretical_stats, tier_df)

    print("\n  --- MACHINE LEARNING DIAGNOSIS ---")
    for line in VERDICT_LINES[verdict]:
        print(line)
    print(f"  Confidence Score   : {severity_score:.4f}  ({severity_label})")
    print("=" * 64 + "\n")

#BULK CSV PREDICTION
#only for use with a CSV of theoretical units, not for single-unit predictions

def predict_bulk_from_csv(filepath, output_path='bulk_powercreep_results.csv'):
    """
    Reads a CSV of theoretical units and evaluates every row.
    Required columns: character_name, rarity_tier, + all 7 base stat columns.

    Output CSV columns:
      verdict          — POWERCREPT / UNDERPOWERED / BALANCED
      confidence       — model confidence score
      composite_power  — overall power number
      tier_z_score     — std deviations from tier mean (sign shows direction)
      bulk_percentile  — combined HP+DEF percentile within tier
      bulk_flag        — HIGH / LOW / NORMAL
      pct_<stat>       — per-stat percentile for each of the 7 base stats
    """
    try:
        candidates = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[ERROR] Could not find '{filepath}'.")
        return

    required_cols = ['character_name', 'rarity_tier'] + BASE_FEATURES
    missing = [c for c in required_cols if c not in candidates.columns]
    if missing:
        print(f"[ERROR] Missing columns in input CSV: {missing}")
        return

    results = []

    for _, row in candidates.iterrows():
        tier = str(row['rarity_tier']).upper().strip()
        name = str(row['character_name']).strip()

        if tier not in tier_models:
            results.append({'character_name': name, 'rarity_tier': tier,
                            'verdict': 'UNKNOWN_TIER'})
            continue

        stats          = {f: row[f] for f in BASE_FEATURES}
        enriched_stats = add_ratio_features(stats)
        bucket         = tier_models[tier]
        tier_df        = bucket['df']

        new_row    = pd.DataFrame([enriched_stats])[ALL_FEATURES]
        new_scaled = bucket['scaler'].transform(new_row.values)

        new_composite = composite_score(enriched_stats)
        z_score       = (new_composite - bucket['tier_mean']) / (bucket['tier_std'] + 1e-9)

        if bucket['type'] == 'supervised':
            powercreep_proba = bucket['model'].predict_proba(new_scaled)[0][1]
            is_anomaly       = powercreep_proba > 0.5
            confidence       = powercreep_proba
        else:
            raw        = bucket['model'].predict(new_scaled)[0]
            is_anomaly = (raw == -1)
            confidence = bucket['model'].score_samples(new_scaled)[0]

        verdict = get_verdict(is_anomaly, z_score)

        unit_bulk = stats['base_hp'] + stats['base_strike_defence'] + stats['base_blast_defence']
        tier_bulk = tier_df['base_hp'] + tier_df['base_strike_defence'] + tier_df['base_blast_defence']
        _, bulk_pct = _stat_bar(unit_bulk, tier_bulk)

        if bulk_pct > 90:
            bulk_flag = 'HIGH'
        elif bulk_pct < 10:
            bulk_flag = 'LOW'
        else:
            bulk_flag = 'NORMAL'

        entry = {
            'character_name':  name,
            'rarity_tier':     tier,
            'verdict':         verdict,
            'confidence':      round(confidence, 4),
            'composite_power': round(new_composite, 1),
            'tier_z_score':    round(z_score, 3),
            'bulk_percentile': round(bulk_pct, 1),
            'bulk_flag':       bulk_flag,
        }
        for feat in STAT_LABELS:
            _, pct = _stat_bar(stats[feat], tier_df[feat])
            entry[f'pct_{feat}'] = round(pct, 1)

        results.append(entry)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)

    n_powercrept   = (out_df['verdict'] == 'POWERCREPT').sum()
    n_underpowered = (out_df['verdict'] == 'UNDERPOWERED').sum()
    n_balanced     = (out_df['verdict'] == 'BALANCED').sum()

    print(f"Bulk evaluation complete: {len(out_df)} units processed.")
    print(f"  Powercrept   : {n_powercrept}")
    print(f"  Underpowered : {n_underpowered}")
    print(f"  Balanced     : {n_balanced}")
    print(f"  Results saved to '{output_path}'\n")

#  TEST CASES

# Case 1: Reasonably balanced LEGEND — should be BALANCED
balanced_concept = {
    'base_strike_attack':  255000,
    'base_blast_attack':   248000,
    'base_hp':            2400000,
    'base_strike_defence': 175000,
    'base_blast_defence':  176000,
    'critical_rate':        0.1450,
    'ki_recovery':          0.2200
}
predict_powercreep("Hypothetical Balanced Super Vegito", balanced_concept, tier='LEGEND')

# Case 2: Intentionally broken ULTRA — should be POWERCREPT
broken_concept = {
    'base_strike_attack':  315000,
    'base_blast_attack':   310000,
    'base_hp':            2950000,
    'base_strike_defence': 198000,
    'base_blast_defence':  199000,
    'critical_rate':        0.2200,
    'ki_recovery':          0.3500
}
predict_powercreep("Ultra Ultra Instinct Goku", broken_concept, tier='ULTRA')

# Case 3: Deliberately weak LEGEND — should be UNDERPOWERED
weak_concept = {
    'base_strike_attack':  190000,
    'base_blast_attack':   185000,
    'base_hp':            1900000,
    'base_strike_defence': 140000,
    'base_blast_defence':  140000,
    'critical_rate':        0.0900,
    'ki_recovery':          0.1500
}
predict_powercreep("Hypothetical Weak LEGEND Gohan", weak_concept, tier='LEGEND')

# Case 4: Strong SPARKING — judged vs SPARKING baseline only
sparking_concept = {
    'base_strike_attack':  280000,
    'base_blast_attack':   275000,
    'base_hp':            2600000,
    'base_strike_defence': 185000,
    'base_blast_defence':  186000,
    'critical_rate':        0.1550,
    'ki_recovery':          0.2600
}
predict_powercreep("Hypothetical Strong SPARKING Broly", sparking_concept, tier='SPARKING')

# Case 5: Glass cannon SPARKING — ratio features should flag Off/Def ratio
glass_cannon = {
    'base_strike_attack':  305000,
    'base_blast_attack':   300000,
    'base_hp':            2100000,
    'base_strike_defence': 140000,
    'base_blast_defence':  141000,
    'critical_rate':        0.1900,
    'ki_recovery':          0.3100
}
predict_powercreep("Hypothetical Glassy Cannon Vegeta", glass_cannon, tier='SPARKING')
