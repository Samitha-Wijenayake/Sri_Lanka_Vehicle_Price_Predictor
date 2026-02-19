import requests
import json
import csv
import time
import sys
from bs4 import BeautifulSoup

# Configuration
BASE_URL = "https://ikman.lk/en/ads/sri-lanka/vehicles"
OUTPUT_FILE = "vehicle_data_large.csv"
TARGET_COUNT = 20000
DELAY_SECONDS = 0.3

# CSV Header
CSV_HEADER = ["Title", "Price", "Mileage", "Location", "Description", "PublishedDate", "Link", "ImageURL", "Brand", "Model", "Condition", "Transmission", "FuelType", "EngineCapacity", "Year"]

def get_page(page_number):
    """Fetches a single page of results."""
    url = f"{BASE_URL}?page={page_number}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {page_number}: {e}")
        return None

def extract_data_from_html(html_content):
    """Extracts the initialData JSON from the script tag."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the script containing window.initialData
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and 'window.initialData' in script.string:
            # Extract the JSON part
            try:
                # The script content looks like: window.initialData = {...}
                json_str = script.string.split('window.initialData = ')[1].strip()
                # Remove trailing semicolon if present (though split might handle it if at end)
                # Usually it's just the object or followed by newline/semicolon
                # Let's try to be robust
                # Sometimes it might be followed by other vars
                # Using a safer extraction method if split is too fragile
                # But initialData is usually at the start of a line. 
                # Let's try parsing up to the last closing brace if json loads fails?
                # Actually, the previous view_file showed it quite cleanly on one line or block
                # window.initialData = {...}
                # let's assume it ends with } or };
                
                # Simple extraction: find the start brace and match braces? or just load it
                # Logic: Find the first { and attempt to load.
                # Or just split by 'window.initialData = ' and take the rest, verify it's valid json
                
                # Re-examining the snippet from view_file:
                # window.initialData = {"locale":"en",...}
                # window.chatConfig = ...
                # It seems each is in its own script tag or separated. 
                # Let's clean up potential trailing stuff.
                
                # The safest way is to find the object start and end. 
                # However, since we saw the source, it looked like:
                # <script type="text/javascript">
                # window.initialData = {...}
                # </script>
                
                # So taking everything after initialData = should work if we strip properly.
                
                content = json_str
                # Try to find the end of the JSON object
                # It's safer to just rely on the fact that execution usually puts one big object there.
                
                data = json.loads(content)
                return data
            except (IndexError, json.JSONDecodeError) as e:
                # Fallback for complex script content
                # print(f"JSON extraction error details: {e}")
                pass
                
    return None

def parse_ad_data(ad):
    """Parses a single ad dictionary into our desired format."""
    
    # Extract basic fields
    title = ad.get('title', '')
    price = ad.get('price', '').replace('Rs', '').replace(',', '').strip()
    
    # Location usually "City, Category" e.g., "Colombo, Cars"
    description = ad.get('description', '')
    location = ad.get('location', '')
    
    # The 'details' field often has mileage e.g. "101,223 km" or year
    details = ad.get('details', '')
    mileage = ''
    if 'km' in details:
        mileage = details
    
    published_date = ad.get('timeStamp', '')
    slug = ad.get('slug', '')
    link = f"https://ikman.lk/en/ad/{slug}" if slug else ''
    image_url = ad.get('imgUrl', '')
    
    # Some fields might be in 'attributes' if available, but the initialData snippet showed mostly summary data.
    # The summary data (serp.ads.data.ads) is limited. 
    # It has: id, slug, title, description, details, subtitle, imgUrl, price, isMember, etc.
    # Deep details like Brand, Model, Year, Fuel might require parsing the title or details string 
    # OR requesting the individual ad page (which is too slow for 5000 ads).
    # IKMAN usually puts Year and Mileage in the title or details.
    
    # Try to extract Year from title or details
    # Example Title: "Toyota Premio G Superior 2018" -> Year 2018
    # Example Details: "101,223 km" 
    
    year = ''
    # Simple heuristic for year (4 digits starting with 19 or 20)
    import re
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    if year_match:
        year = year_match.group(0)
    elif details and re.search(r'\b(19|20)\d{2}\b', details):
        # Sometimes details has year if mileage isn't there?
         year = re.search(r'\b(19|20)\d{2}\b', details).group(0)

    return {
        "Title": title,
        "Price": price,
        "Mileage": mileage,
        "Location": location,
        "Description": description,
        "PublishedDate": published_date,
        "Link": link,
        "ImageURL": image_url,
        "Year": year,
        # Placeholder for other fields not in summary
        "Brand": "", 
        "Model": "",
        "Condition": "",
        "Transmission": "",
        "FuelType": "",
        "EngineCapacity": ""
    }

def main():
    # Default range
    start_page = 1
    end_page = 800
    
    # Parse arguments
    if "--start_page" in sys.argv:
        try:
            idx = sys.argv.index("--start_page")
            start_page = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            print("Invalid start_page. Using default.")

    if "--end_page" in sys.argv:
        try:
            idx = sys.argv.index("--end_page")
            end_page = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            print("Invalid end_page. Using default.")
            
    # Allow simple limit override if only limit is provided (for backward compatibility or simple testing)
    if "--limit" in sys.argv and "--end_page" not in sys.argv:
        try:
            limit_idx = sys.argv.index("--limit")
            limit = int(sys.argv[limit_idx + 1])
            end_page = start_page + limit - 1
        except (ValueError, IndexError):
            pass

    # Dynamic output file to avoid collisions
    output_filename = f"vehicle_data_{start_page}_{end_page}.csv"
    print(f"Scraping pages {start_page} to {end_page} into {output_filename}...")

    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
        writer.writeheader()
        
        total_scraped = 0
        
        for i in range(start_page, end_page + 1):
            print(f"Scraping page {i}...")
            html = get_page(i)
            
            if not html:
                continue
                
            data = extract_data_from_html(html)
            
            if not data:
                print(f"Failed to extract data from page {i}")
                continue
            
            # Navigate to the ads list
            try:
                # Based on observed structure: window.initialData.serp.ads.data.ads
                ads_list = data.get('serp', {}).get('ads', {}).get('data', {}).get('ads', [])
                
                if not ads_list:
                    print(f"No ads found on page {i}. Stopping.")
                    break
                
                for ad in ads_list:
                    parsed_ad = parse_ad_data(ad)
                    writer.writerow(parsed_ad)
                    total_scraped += 1
                
                print(f"  Saved {len(ads_list)} ads. Total: {total_scraped} (in this batch)")
                
            except AttributeError as e:
                print(f"Error parsing JSON structure on page {i}: {e}")
            
            time.sleep(DELAY_SECONDS)

    print(f"Scraping complete. {total_scraped} ads saved to {output_filename}")

if __name__ == "__main__":
    main()
