from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
import pandas as pd

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.get('https://finance.yahoo.com')
driver.maximize_window()

yf_input = driver.find_element(By.XPATH, "/html/body/div[1]/header/div/div/div/div[2]/div/div[1]/div[3]/form/input[1]")
yf_search = driver.find_element(By.XPATH, '/html/body/div[1]/header/div/div/div/div[2]/div/div[1]/div[3]/form/div[3]/button')

yf_input.send_keys("GOOG")
sleep(1)
yf_search.click()

yf_news = driver.find_element(By.XPATH, '/html/body/div[1]/main/section/section/aside/section/nav/ul/li[2]/a')
yf_news.click()

yf_goog_news =  driver.find_element(By.XPATH, '/html/body/div[1]/main/section/section/section/article/section[2]/div[1]/button[2]')
yf_goog_news.click()

def scroll_to_bottom():
    body = driver.find_element(By.TAG_NAME, 'body')
    scroll_iterations = 3
    for _ in range(scroll_iterations):
        body.send_keys(Keys.END)
        sleep(3)

scroll_to_bottom()

headlines = []
summaries = []

for i in range(1, 15):
    try:
        headline_elements = driver.find_elements(By.XPATH, f'/html/body/div[1]/main/section/section/section/article/section[2]/div[2]/div/div/ul/li[{i}]/section/div/a/h3')
        for headline_element in headline_elements:
            headline_text = headline_element.text
            headlines.append(headline_text)
    except:
        headlines.append("")

    try:
        summary_elements = driver.find_elements(By.XPATH, f'/html/body/div[1]/main/section/section/section/article/section[2]/div[2]/div/div/ul/li[{i}]/section/div/a/p')
        for summary_element in summary_elements:
            summary_text = summary_element.text
            summaries.append(summary_text)
    except:
        summaries.append("")

print("Headlines:")
print(headlines)
print(f"Number of headlines: {len(headlines)}")

print("\nSummaries:")
print(summaries)
print(f"Number of summaries: {len(summaries)}")

df = pd.DataFrame({
    'Headline': headlines,
    'Summary': summaries
})

df.to_csv('goog_news.csv', index=False)

print("SUCCESS")
driver.quit()
