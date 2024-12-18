import requests
from bs4 import BeautifulSoup
import json
import time
import schedule
import threading

# Xəbər yığmaq üçün funksiyalar


def fetch_cnn_news():
    url = "https://edition.cnn.com/world"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    articles = soup.select("h3 a")  # CNN-də başlıqlar üçün uyğun selektor
    news = []
    for article in articles[:10]:  # İlk 10 xəbəri götürürük
        title = article.get_text(strip=True)
        link = "https://edition.cnn.com" + article['href']
        news.append({"source": "CNN", "title": title, "link": link})
    return news


def fetch_bbc_news():
    url = "https://www.bbc.com/news"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # BBC-də başlıqlar üçün uyğun selektor
    articles = soup.select(".gs-c-promo-heading")
    news = []
    for article in articles[:10]:
        title = article.get_text(strip=True)
        link = article['href']
        if not link.startswith("http"):  # Nisbətən linkləri tam linkə çeviririk
            link = "https://www.bbc.com" + link
        news.append({"source": "BBC", "title": title, "link": link})
    return news

# Xəbərləri JSON fayla yazır


def save_news_to_json(news, filename="news.json"):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(news, file, indent=4, ensure_ascii=False)

# Xəbərləri toplayan əsas funksiya


def collect_news():
    try:
        print("Xəbərlər yığılır...")
        cnn_news = fetch_cnn_news()
        bbc_news = fetch_bbc_news()
        all_news = cnn_news + bbc_news
        save_news_to_json(all_news)
        print(f"{len(all_news)} xəbər toplandı və 'news.json' faylına yazıldı.")
    except Exception as e:
        print(f"Xəta baş verdi: {e}")

# İş planlaşdırıcısı


def start_scheduler():
    schedule.every(24).hours.do(collect_news)  # Hər 24 saatda bir dəfə çalışır
    while True:
        schedule.run_pending()
        time.sleep(1)


# Multi-threading ilə həm botu çalışdırır, həm də digər işləri edə bilərsiniz
if __name__ == "__main__":
    collect_news()  # İlk dəfə xəbərləri yığır
    threading.Thread(target=start_scheduler).start()
