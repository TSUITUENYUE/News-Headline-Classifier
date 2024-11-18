import csv
import requests
import bs4
import os

def read_news_url(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def open_url(url):
    response = requests.get(url)
    return response

def get_news(url):
    if "foxnews" in url:
        return "fox"
    elif "nbcnews" in url:
        return "nbc"
    # else:
    #     return None

def process_res(response):
    if response:
        soup = bs4.BeautifulSoup(response.content, 'html.parser')
        return soup
    # else:
    #     return None

def get_title(soup, news):
    if news == "fox":
        if soup.find('h1'):
            title = soup.find('h1').get_text()
        elif soup.find('h2'):
            title = soup.find('h2').get_text()
    elif news == "nbc":
        if soup.find("h1"):
            title = soup.find("h1").get_text()
        elif soup.find("h2"):
            title = soup.find("h2").get_text()
    # else:
    # title = None
    return title

# def save_csv(output, news_list, title_list, url_list):
#     with open(output, 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(["title", "news", "url"])
#         for i in range(len(news_list)):
#             writer.writerow([news_list[i], title_list[i], url_list[i]])

def init_output_csv(output):
    with open(output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["title", "news", "index", "url"])

def append_to_csv(output, news, title, url, index):
    with open(output, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([title, news, index, url])

if __name__ == '__main__':
    input = "./data/fox_nbc/url_only_data.csv"
    output = "./data/fox_nbc/data.csv"
    data = read_news_url(input)
    url_list = [data[i][0] for i in range(1, len(data))]
    title_list = []
    news_list = []
    if not os.path.exists(output):
        init_output_csv(output)

    start = 0
    for i in range(start, len(url_list)):
        url = url_list[i]
        response = open_url(url)
        news = get_news(url)
        soup = process_res(response)
        title = get_title(soup, news)
        title_list.append(title)
        news_list.append(news)
        # print(f"Progress {i+1}/{len(url_list)} | News: {news} | Title: {title}" + " "*50, end="\r")
        print(f"Progress {i+1}/{len(url_list)} | News: {news} | Title: {title[:15]}... | Link: {url}")
        append_to_csv(output, news, title, url, i)
    print("\n")
    # save_csv(output, news_list, title_list, url_list)
    # print(f"Data saved to {output}")
