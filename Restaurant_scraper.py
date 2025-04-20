import time
import csv
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up Chrome with undetected_chromedriver
options = uc.ChromeOptions()
options.page_load_strategy = 'eager'
driver = uc.Chrome(options=options)

# Open Zomato Kanpur page
url = "https://www.zomato.com/kanpur"
driver.get(url)

try:
    # Wait for restaurant cards to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.sc-evWYkj.cRThYq'))
    )

    # Get the first restaurant card link
    divs = driver.find_elements(By.CSS_SELECTOR, '.sc-evWYkj.cRThYq')[:10]
    links = []

    for div in divs:
        try:
            element = div.find_element(By.CSS_SELECTOR, "a.sc-hqGPoI.kCiEKB")
            link = element.get_attribute("href").strip()
            links.append(link)
        except:
            print("Link not found")

    # Write to CSV
    with open('restaurants.csv', 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Restaurant Name', 'Location', 'Cuisine', 'Price', 'Ratings(Dining, Delivery)', 'Operating Hours',
             'Contact', 'More Info'])

        for link in links:
            driver.get(link)
            time.sleep(1)

            try:
                name = driver.find_element(By.CSS_SELECTOR, 'h1.sc-7kepeu-0.sc-iSDuPN.fwzNdh').text
            except:
                name = 'N/A'

            try:
                address = driver.find_element(By.CSS_SELECTOR, 'div.sc-clNaTc.ckqoPM').text
            except:
                address = 'N/A'

            try:
                phone = driver.find_element(By.CSS_SELECTOR, 'a.sc-bFADNz.leEVAg').get_attribute("textContent").strip()
            except:
                phone = 'N/A'

            try:
                price = driver.find_element(By.CSS_SELECTOR, 'div.sc-bEjcJn.ePRRqr').text
            except:
                price = 'N/A'

            try:
                rate_elements = driver.find_elements(By.CSS_SELECTOR, 'div.sc-1q7bklc-1.cILgox')[:2]
                rating = ', '.join([r.text for r in rate_elements if r.text.strip()])
            except:
                rating = 'N/A'

            try:
                hours = driver.find_element(By.CSS_SELECTOR, 'span.sc-kasBVs.dfwCXs').text
            except:
                hours = 'N/A'

            try:
                cuisine_elements = driver.find_elements(By.CSS_SELECTOR, 'div.sc-bFADNz.grcEwO')
                cuisines = ', '.join([c.text for c in cuisine_elements if c.text.strip()])
            except:
                cuisines = 'N/A'

            try:
                info_elements = WebDriverWait(driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'p.sc-1hez2tp-0.cunMUz')))
                info = ', '.join([i.text for i in info_elements if i.text.strip()])
            except:
                info = 'N/A'

            writer.writerow([name, address, cuisines, price, rating, hours, phone, info])

except Exception as e:
    print(f"An error occurred: {e}")

driver.quit()
