from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup
import os
import urllib.parse

# Initialize random user agent
ua = UserAgent()

# Set headers
headers = {
    "User-Agent": ua.random,
}

# Target website
url = (
    "https://www.princeedwardisland.ca/en/service/apply-for-a-tourism-pei-hosting-grant"
)

# Request the page
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Create a folder for images
os.makedirs("images", exist_ok=True)

# Find all image tags
for img_tag in soup.find_all("img"):
    img_url = img_tag.get("src")
    if img_url:
        img_url = urllib.parse.urljoin(url, img_url)  # Handle relative URLs
        img_name = os.path.join("images", os.path.basename(img_url))

        # Download image
        img_data = requests.get(img_url, headers=headers).content
        with open(img_name, "wb") as img_file:
            img_file.write(img_data)
