import requests
from bs4 import BeautifulSoup
import json

# Yoxlama funksiyası


def check_response(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(url, headers=headers)
    print(f"URL: {url} | Status Code: {response.status_code}")
    return response


def fetch_cnn_news():
    url = "https://edition.cnn.com/world"
    response = check_response(url)
    soup = BeautifulSoup(response.content, "html.parser")
    articles = soup.find_all(
        "span", class_="cd__headline-text")  # Alternativ selektor
    news = []
    for article in articles[:10]:
        title = article.get_text(strip=True)
        link = "https://edition.cnn.com" + article.parent['href']
        news.append({"source": "CNN", "title": title, "link": link})
    return news


def fetch_bbc_news():
    url = "https://www.bbc.com/news"
    response = check_response(url)
    soup = BeautifulSoup(response.content, "html.parser")
    articles = soup.find_all(
        "a", class_="gs-c-promo-heading")  # Alternativ selektor
    news = []
    for article in articles[:10]:
        title = article.get_text(strip=True)
        link = article['href']
        if not link.startswith("http"):
            link = "https://www.bbc.com" + link
        news.append({"source": "BBC", "title": title, "link": link})
    return news


def collect_news():
    try:
        print("Xəbərlər yığılır...")
        cnn_news = fetch_cnn_news()
        bbc_news = fetch_bbc_news()
        all_news = cnn_news + bbc_news
        with open("news.json", "w", encoding="utf-8") as file:
            json.dump(all_news, file, indent=4, ensure_ascii=False)
        print(f"{len(all_news)} xəbər toplandı və 'news.json' faylına yazıldı.")
    except Exception as e:
        print(f"Xəta baş verdi: {e}")


if __name__ == "__main__":
    collect_news()
