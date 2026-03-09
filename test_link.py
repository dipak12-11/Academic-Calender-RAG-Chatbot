import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

def get_latest_calendar_url() -> str:
    """Scrapes IIT ISM academics page to find the latest academic calendar PDF."""
    
    base_url = "https://people.iitism.ac.in/~academics/"
    response = requests.get(base_url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find the "Academic Calendar" dropdown link
    # then grab the FIRST <a> tag inside it (top option = latest year)
    all_links = soup.find_all("a", href=True)
    
    for link in all_links:
        if "Academic" in link.text and ".pdf" in link["href"]:
            pdf_url = link["href"]
            # Handle relative URLs
            if not pdf_url.startswith("http"):
                pdf_url = base_url + pdf_url.lstrip("/")
            parts = pdf_url.split("iitism.ac.in/")
            encoded_path = quote(parts[1])  # encodes spaces → %20 etc.
            pdf_url = f"https://people.iitism.ac.in/{encoded_path}"
            print(f"📄 Found calendar: {pdf_url}")
            return pdf_url
    
    raise ValueError("Could not find Academic Calendar PDF on the page!")


# Then in your loader, replace the hardcoded url with:
url = get_latest_calendar_url()
# load_pdf_from_url(url)