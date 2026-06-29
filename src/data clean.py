import pandas as pd

try:
    df = pd.read_csv('detailed_characters_data.csv')

except FileNotFoundError:
    print("Error: csv not found. Ensure it's in the same directory.")
    exit()


# Filter out any rows where the script had to use fallback 0 values due to server errors
df_clean = df[df['base_hp'] > 0].copy()

# Clean percentage strings (e.g., "13.64%") and convert them to numeric decimals (0.1364)
for col in ['critical_rate', 'ki_recovery']:
    if col in df_clean.columns:
        # Convert to string first to safely apply string manipulation methods
        df_clean[col] = df_clean[col].astype(str).str.replace('%', '')
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce') / 100.0


print("=======================================================")
print("              DATA AUDIT            ")
print("=======================================================")

# 1. Print structural column information and data types
print("\n--- 1. DATA TYPES AND NULL VALUE CHECK ---")
print(df_clean.info())

# 2. Generate core statistical metrics (Mean, Min, Max, Standard Deviation)
print("\n--- 2. CORE STATISTICAL PROFILE ---")
# Select our main numeric balance features
numeric_features = [
    'base_strike_attack', 'base_blast_attack', 'base_hp', 
    'base_strike_defence', 'base_blast_defence', 'critical_rate', 'ki_recovery'
]
print(df_clean[numeric_features].describe().T)

# 3. Check for obvious correlation pairs (Strike vs Blast emphasis)
print("\n--- 3. HIGHEST OFFENSIVE PEAKS IN THE CURRENT DATASET ---")
# Calculate a simple combined offensive index to see who sits at the top right now
df_clean['combined_offence'] = df_clean['base_strike_attack'] + df_clean['base_blast_attack']
top_offensive = df_clean.sort_values(by='combined_offence', ascending=False).head(5)

for idx, row in top_offensive.iterrows():
    print(f"• {row['character_name']} ({row['rarity_tier']}) -> Combined Offence: {row['combined_offence']:,}")

df_clean.to_csv('cleaned_dbl_stats.csv', index=False)
print("\n-------------------------------------------------------")
print("Cleaned data baseline saved to 'cleaned_dbl_stats.csv'!")
print("-------------------------------------------------------")