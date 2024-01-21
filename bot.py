import time
from selenium import webdriver

for i in range(10000):
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    driver.get("http://127.0.0.1:3000/index.html")
    time.sleep(3)
    driver.quit()
    print('Registered a visit', i)
