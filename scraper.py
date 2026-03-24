import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import csv
import pandas as pd

BASE_URL = "https://ekantipur.com"

CATEGORY_URLS = [
    "https://ekantipur.com/news",
    "https://ekantipur.com/business",
    "https://ekantipur.com/opinion",
    "https://ekantipur.com/world",
    "https://ekantipur.com/sports",
]

# ✅ HEADERS (VERY IMPORTANT)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://google.com",
}

session = requests.Session()
session.headers.update(HEADERS)


def delay(ms):
    time.sleep(ms / 1000)


# 🔍 ARTICLE DETAIL
def scrape_article_detail(url):
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # 📰 CONTENT
        content = ""
        content_div = soup.find("div", class_="description")

        if content_div:
            paragraphs = content_div.find_all("p")
            content = "\n".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

        # 📅 DATE
        published_date = ""
        updated_date = ""

        date_tag = soup.find("span", class_="published-at")
        if date_tag:
            published_date = date_tag.get_text(strip=True)

        updated_tag = soup.find("span", class_="updated-at")
        if updated_tag:
            updated_date = updated_tag.get_text(strip=True)

        # ✍️ AUTHOR
        author = ""
        author_tag = soup.find("a", class_="author")
        if author_tag:
            author = author_tag.get_text(strip=True)

        # 📍 LOCATION
        location = ""
        loc_tag = soup.find("span", class_="location")
        if loc_tag:
            location = loc_tag.get_text(strip=True)

        # 🏷 CATEGORY
        segments = [seg for seg in urlparse(url).path.split("/") if seg]
        category = segments[0] if segments else "general"

        return {
            "author": author,
            "publishedDate": published_date,
            "updatedDate": updated_date,
            "location": location,
            "category": category,
            "content": content,
        }

    except Exception as e:
        print(f"❌ Detail error: {url} → {e}")
        return None


# 🔍 CATEGORY PAGE
def scrape_category_page(category_url):
    articles_data = []

    try:
        print(f"🔎 Scraping: {category_url}")

        response = session.get(category_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        articles = soup.find_all("article")

        for art in articles:
            a_tag = art.find("a", href=True)

            if not a_tag:
                continue

            title = a_tag.get_text(strip=True)
            link = urljoin(BASE_URL, a_tag["href"])

            detail = scrape_article_detail(link)

            if detail:
                articles_data.append({
                    "title": title,
                    "link": link,
                    **detail
                })

                delay(800)

    except Exception as e:
        print(f"❌ Category error: {category_url} → {e}")

    return articles_data


# 🚀 MAIN
if __name__ == "__main__":
    all_articles = []

    for url in CATEGORY_URLS:
        data = scrape_category_page(url)
        all_articles.extend(data)

    print(f"✅ Total scraped: {len(all_articles)}")

    # 💾 SAVE CSV
    filename = "ekantipur_news.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "title", "link", "author",
            "publishedDate", "updatedDate",
            "location", "category", "content"
        ])
        writer.writeheader()
        writer.writerows(all_articles)

    print(f"📁 Saved: {filename}")

    # 📊 DataFrame
    df = pd.DataFrame(all_articles)
    print(df.head())