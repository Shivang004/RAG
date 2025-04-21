import time
import csv
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up Chrome with undetected_chromedriver
options = uc.ChromeOptions()
options.page_load_strategy = 'eager'
options.add_argument('incognito')
driver = uc.Chrome(options=options)

# Open Zomato Kanpur page
url = "https://www.swiggy.com/city/kanpur"
driver.get(url)

max_loading_attempt = 0
count = 0

while True:
    try:
        Load_Button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//div[text()='Show more']")))
        driver.execute_script("arguments[0].scrollIntoView(true);", Load_Button)
        driver.execute_script("arguments[0].click();", Load_Button)
        print("clicked")
        count += 1
        time.sleep(1.5)
    except Exception:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        max_loading_attempt += 1

    if count == 5 or max_loading_attempt >= 1:
        print("No more new content")
        break

    # Wait for restaurant cards to load
WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a.sc-hhyLtv.evCvfO'))
)

# Get the first restaurant card link
a_tags = driver.find_elements(By.CSS_SELECTOR, 'a.sc-hhyLtv.evCvfO')
links = []

for a_tag in a_tags:
    try:
        link = a_tag.get_attribute("href").strip()
        links.append(link)
        print(link)
    except:
        print("Link not found")

# Write to CSV
with open('restaurants.csv', 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Restaurant Name', 'Location', 'Cuisine', 'Price', 'Ratings(Dining, Delivery)', 'Operating Hours',
         'Menu', 'More Info'])

    for link in links:
        driver.get(link)
        time.sleep(1)

        try:
            name = driver.find_element(By.CSS_SELECTOR, 'h1.sc-beySbM.iHVupX').text
        except:
            name = 'N/A'

        try:
            address = driver.find_element(By.CSS_SELECTOR, 'div.sc-beySbM.izMqWl.LocationWidget__TextContainer-sc-5o5o50-6.bSqMPE').text
        except:
            address = 'N/A'

        try:
            price = driver.find_elements(By.CSS_SELECTOR, 'div.sc-beySbM.eLCOgH.RestaurantShortInfo__InfoText-sc-1py2i0v-17.dwYYtL')[1].text
        except:
            price = 'N/A'

        try:
            rating = driver.find_elements(By.CSS_SELECTOR, 'div.sc-beySbM.eLCOgH.RestaurantShortInfo__InfoText-sc-1py2i0v-17.dwYYtL')[0].text
        except:
            rating = 'N/A'

        try:
            hours = driver.find_element(By.CSS_SELECTOR, 'div.sc-beySbM.izMqWl.RestaurantTimings__TextContainer-sc-1a1nnl6-5.egyKAO').text
        except:
            hours = 'N/A'

        try:
            cuisines = driver.find_element(By.CSS_SELECTOR, 'div.sc-beySbM.bWPKRg.CuisineListWidget__CuisineText-sc-15vhfs3-3.hDNqYo').text
            # cuisines = ', '.join([c.text for c in cuisine_elements if c.text.strip()])
        except:
            cuisines = 'N/A'

        try:
            info_elements = WebDriverWait(driver, 5).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'span.HighlightWidget__TextStyle-sc-1ouo46e-5.lfvwst')))
            info = ', '.join([i.text for i in info_elements if i.text.strip()])
        except:
            info = 'N/A'

        try:
            menu_items = driver.find_elements(By.CSS_SELECTOR, 'img.sc-guDLRT.iUZcNW')
            menu_links = [item.get_attribute('src') for item in menu_items]
            menu = ', '.join(menu_links)

        except Exception as e:
            Menu = 'N/A'
        writer.writerow([name, address, cuisines, price, rating, hours, menu, info])
driver.quit()
