import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import argparse
import os

# -------------------------------------------------------------------------
# ETHICAL SCRAPING NOTE:
# This script is for educational purposes as part of a Machine Learning assignment.
# It includes delays to respect the server's load and adhere to general 
# web scraping etiquette. Always check a website's robots.txt before scraping.
# -------------------------------------------------------------------------

def scrape_ikman_vehicles(max_pages=5, output_path="data/data.csv"):
    """
    Scrapes vehicle advertisements from ikman.lk using requests and BeautifulSoup.
    """
    base_url = "https://ikman.lk/en/ads/sri-lanka/vehicles"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    all_data = []
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Starting to scrape {max_pages} pages from ikman.lk...")

    for page in range(1, max_pages + 1):
        # The pagination parameter might vary, often it's "page=" or similar.
        # Assuming '?page=' for illustration based on common patterns.
        url = f"{base_url}?page={page}"
        print(f"Scraping page {page}...")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch page {page}: {e}")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        
        # NOTE: The exact selectors here depend heavily on Ikman's current HTML structure.
        # The following are generic placeholder selectors that typically represent listing cards.
        # You may need to inspect the live website and update these classes.
        ads = soup.find_all("li", class_="normal--2QYVk") # Example class name, verify on actual site
        
        if not ads:
            # Fallback if class name changed or trying another common wrapper
            ads = soup.find_all("div", class_="list-item")
            
        if not ads:
             print("Warning: Could not find ad list items with current selectors. Check HTML structure.")

        for ad in ads:
            try:
                # Extract details based on typical structure.
                title_elem = ad.find("h2")
                title = title_elem.text.strip() if title_elem else "Unknown"
                
                price_elem = ad.find("div", class_="price--3SnqI") # Example
                price = price_elem.text.strip() if price_elem else "0"
                
                # Usually subtext contains location, condition, etc.
                desc_elem = ad.find("div", class_="description--2-ez3") # Example
                desc = desc_elem.text.strip() if desc_elem else "Unknown"
                
                # Further extraction (Brand, Model, Year, Mileage) usually requires visiting the ad page
                # or parsing the title/description string. Here we do simple parsing for assignment sake.
                # Assuming title often has: "Brand Model Year"
                
                parts = title.split()
                brand = parts[0] if len(parts) > 0 else "Unknown"
                model = " ".join(parts[1:-1]) if len(parts) > 2 else "Unknown"
                year = parts[-1] if len(parts) > 1 and parts[-1].isdigit() else "Unknown"
                
                all_data.append({
                    "title": title,
                    "brand": brand,
                    "model": model,
                    "year": year,
                    "price": price,
                    "description": desc,
                    "location": "Unknown", # Needs deeper scraping or specific regex
                    "mileage": "Unknown",  # Needs deeper scraping or specific regex
                    "transmission": "Unknown", # Needs deeper scraping
                    "fuel_type": "Unknown"  # Needs deeper scraping
                })
            except Exception as e:
                print(f"Error parsing an ad: {e}")
                continue

        # Ethical delay between requests
        time.sleep(random.uniform(2, 5))

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_path, index=False)
        print(f"\nSuccessfully scraped {len(df)} vehicles.")
        print(f"Saved to {output_path}")
    else:
        print("\nNo data scraped. Please check your CSS selectors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape vehicle ads from ikman.lk")
    parser.add_argument("--pages", type=int, default=5, help="Maximum number of pages to scrape")
    parser.add_argument("--output", type=str, default="data/scraped_data.csv", help="Output CSV path")
    args = parser.parse_args()

    scrape_ikman_vehicles(max_pages=args.pages, output_path=args.output)
