import time
import os
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from PIL import Image
from io import BytesIO

# Set the path to chromedriver and create a Service object
chromedriver_path = r"C:\Users\angel\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"
service = Service(executable_path=chromedriver_path)

# Set Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Optional: run Chrome in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialize the ChromeDriver
driver = webdriver.Chrome(service=service, options=chrome_options)


# Function to download images from Google search
def download_images(query, num_images, output_dir):
    print(f"Downloading {num_images} images of {query} to {output_dir}...")
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open Google Images
    print("Opening Google Images...")
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    driver.get(search_url)

    # Scroll the page to load more images
    print("Scrolling to load more images...")
    for _ in range(5):
        print("Scrolling...")
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        time.sleep(2)

    # Find image elements
    images = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
    print(f"Found {len(images)} images")

    downloaded = 0

    for i, img in enumerate(images[:num_images]):
        try:
            print(f"Downloading image {i + 1}...")
            # Click the image to open a larger version
            ActionChains(driver).move_to_element(img).click().perform()
            time.sleep(2)  # wait for the image to load

            # Find the large image URL
            large_image = driver.find_element(By.CSS_SELECTOR, "img.n3VNCb")
            src = large_image.get_attribute("src")

            # Skip if the image is a base64 or a link to another page
            if "http" not in src:
                continue

            # Download the image
            img_response = requests.get(src)
            img_data = Image.open(BytesIO(img_response.content))

            # Save the image
            img_path = os.path.join(output_dir, f"{query}_{i + 1}.jpg")
            img_data.save(img_path)
            print(f"Downloaded image {i + 1}: {img_path}")

            downloaded += 1
            if downloaded >= num_images:
                break

        except Exception as e:
            print(f"Failed to download image {i + 1}: {e}")
            continue


# Download blue images
download_images("blue objects", 500, "random blue images")

# Download yellow images
download_images("yellow objects", 500, "random yellow images")

# Close the browser
driver.quit()
