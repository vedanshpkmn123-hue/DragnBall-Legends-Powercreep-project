import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import re  # 1. NEW: Import Regular Expressions to handle partial text matching

try:
    df = pd.read_csv('dbl_meta_dataset.csv')
except FileNotFoundError:
    print("File not found. Please ensure 'dbl_meta_dataset.csv' is in the current directory.")
    exit()

detailed_characters_data = []

# Testing on the first 5 characters
for index, row in df.iterrows():

    character_name = row['character_name']
    link_slug = row['url_slug']

    full_url = 'the url used in data scrape.py' + link_slug 

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    time.sleep(5) # Polite 5-second delay

    sub_response = requests.get(full_url, headers=headers)

    if sub_response.status_code == 200:
        sub_soup = BeautifulSoup(sub_response.text, 'html.parser')

        # Target ONLY the hidden container that holds maximum level PvP stats
        max_stats_container = sub_soup.find("div", id="MaxStats")

        if max_stats_container is not None:
            try:
                # 2. FIXED: Using .find(string=re.compile(...)) to match text containing our keywords safely
                
                # Strike Attack
                strike_label = max_stats_container.find(string=re.compile('Strike ATK'))
                strike_attack = strike_label.find_next().text.strip()
                strike_attack = int(strike_attack.replace(',', ''))

                # Blast Attack
                blast_label = max_stats_container.find(string=re.compile('Blast ATK'))
                blast_attack = blast_label.find_next().text.strip()
                blast_attack = int(blast_attack.replace(',', ''))
                
                # HP
                hp_label = max_stats_container.find(string=re.compile('HP'))
                hp = hp_label.find_next().text.strip()
                hp = int(hp.replace(',', ''))
                
                # Strike Defence
                strike_defence_label = max_stats_container.find(string=re.compile('Strike DEF'))
                strike_defence = strike_defence_label.find_next().text.strip()
                strike_defence = int(strike_defence.replace(',', ''))
                
                # Blast Defence
                blast_defence_label = max_stats_container.find(string=re.compile('Blast DEF'))
                blast_defence = blast_defence_label.find_next().text.strip()
                blast_defence = int(blast_defence.replace(',', ''))
                
                # Critical Rate
                critical_rate_label = max_stats_container.find(string=re.compile('Critical Rate'))
                critical_rate = critical_rate_label.find_next().text.strip()
                
                # Ki Recovery
                ki_recovery_label = max_stats_container.find(string=re.compile('Ki Recover'))
                ki_recovery = ki_recovery_label.find_next().text.strip()
                
            except Exception as e:
                print(f"Parsing error inside try block for {character_name}: {e}")
                strike_attack = blast_attack = hp = strike_defence = blast_defence = 0
                critical_rate = ki_recovery = "0%"
        else:
            print(f"MaxStats box not found for {character_name}. Using fallback zeros.")
            strike_attack = blast_attack = hp = strike_defence = blast_defence = 0
            critical_rate = ki_recovery = "0%"
        
        # 3. NOTE: Double check your column headers ('element' vs 'element_color' etc.)
        complete_profile = {
            "character_name": character_name,
            "element_color": row.get("element", row.get("character_element", "Unknown")),
            "rarity_tier": row.get("rarity", row.get("character_rarity", "Unknown")),
            "base_strike_attack": strike_attack,
            "base_blast_attack": blast_attack,
            "base_hp": hp,
            "base_strike_defence": strike_defence,
            "base_blast_defence": blast_defence,
            "critical_rate": critical_rate,
            "ki_recovery": ki_recovery
        }

        detailed_characters_data.append(complete_profile)
    else:
        print(f"Failed to retrieve data for {character_name}. Status code: {sub_response.status_code}")

# Export final data
final_df = pd.DataFrame(detailed_characters_data)
final_df.to_csv('detailed_characters_data.csv', index=False)
print("Done! Check 'detailed_characters_data.csv' for the test records.")
