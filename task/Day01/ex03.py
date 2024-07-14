from selenium import webdriver
from selenium.webdriver.common.by import By

# URL 설정
URL = "https://ko.wikipedia.org/wiki/위키백과:대문"

# Chrome 옵션 설정
options = webdriver.ChromeOptions()
options.add_argument("--headless") # 브라우저 창을 띄우지 않음
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--no-sandbox")

# 웹 드라이버 설정
driver = webdriver.Chrome(options=options)

try:
    # 위키백과 대문 페이지 열기
    driver.get(URL)

    # "우리 모두가 만들어가는 자유 백과사전"과 "문서 이하 내용" 추출
    main_content = driver.find_element(By.CSS_SELECTOR, "#mw-content-text > div.mw-content-ltr.mw-parser-output > div.main-box.main-top > div > p:nth-child(2)").text
    print("Main Content:", main_content)
finally:
    # 웹 드라이버 종료
    driver.quit()