import time
from selenium import webdriver
from datetime import datetime
import pandas as pd
import os

today = str(datetime.today().date())

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument("--start-maximized")

scrape_list = pd.read_excel('./scrape_list/AI Agents -Scraping List.xlsx')
scrape_list['Name'] = scrape_list['Name'].apply(lambda x: x.replace(' ','_'))
scrape_list['Operator'] = scrape_list['Operator'].apply(lambda x: x.replace(' ','_'))
scrape_list['Operator'] = scrape_list['Operator'].apply(lambda x: x.lower())
scrape_dict = {}
reference_dict = {}

for i,r in scrape_list.iterrows():
    reference_dict[r['Name']] = r['Operator']

for i,r in scrape_list.iterrows():
    scrape_dict[r['Name']] = r['Site']

scrape_list.to_csv("./scrape_list/scrape_list_formatted.csv",index=False)

for k in scrape_dict.keys():
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(scrape_dict[k])
    #driver.maximize_window()
    time.sleep(10)

    total_height = driver.execute_script("return document.body.parentNode.scrollHeight") + 1000

    driver.set_window_size(1920, total_height)

    filename = k+'_'+today+'.png'
    driver.save_screenshot(filename)
    os.system(f"mv {filename} ./{reference_dict[k]}/screenshot/")

    driver.quit()