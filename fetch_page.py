import requests
import sys

try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get('https://ikman.lk/en/ads/sri-lanka/vehicles', headers=headers)
    response.raise_for_status()
    
    with open('page_source.html', 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("Successfully saved page_source.html")

except ImportError:
    print("Error: Requests library not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error fetching page: {e}")
    sys.exit(1)
