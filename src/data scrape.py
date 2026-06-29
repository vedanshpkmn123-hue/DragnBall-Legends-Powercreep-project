import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

#load the url

url = 'insert url here'

#tell you are a standard user

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

#set delay

time.sleep(5)

#download page content

response = requests.get(url, headers=headers)
raw_html = response.text

#organize the data

soup = BeautifulSoup(raw_html, 'html.parser')

# Grab every single character anchor on this page

character_tags = soup.find_all('a', class_='chara-list chara-listing zoom')

#empty list

character_data =[]

#loop through all elements

for tag in character_tags:

    name = tag.get('data-charaname')
    form = tag.get('data-charaformname')
    element = tag.get('data-element')
    rarity = tag.get('data-rarity')


    #other info such as lf, zenkai, tags
    is_lf = int(tag.get('data-lf', 0))
    zenkai = int(tag.get('data-zenkai', -1))
    synergy_tags = tag.get('data-tags', '')

    #extract partial string
    link_slug = tag.get('href')

    #dictionary

    character_info ={
        'character_name': name,
        'character_form': form,
        'character_element': element,
        'character_rarity': rarity,
        
        'is_legends_limited': is_lf,
        'zenkai_status': zenkai,
        'synergy_tags': synergy_tags,
        'url_slug': link_slug
    }

    #append 
    character_data.append(character_info)

#Turn your master list of dictionaries into a structured data table
df = pd.DataFrame(character_data)

#Save it to a clean CSV file inside your local directory
df.to_csv("dbl_meta_dataset.csv", index=False)

print(f"Extraction successful! Saved {len(df)} character records to dbl_meta_dataset.csv.")
