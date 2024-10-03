import os
import cv2
import time
import random
import requests
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def download_images_with_variations(base_query, variations, total_images, download_path):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get("https://www.google.com/imghp")

    images = set()
    downloaded = 0
    batch_size = 100

    while downloaded < total_images:
        query = f"{base_query} {random.choice(variations)}"
        search_box = driver.find_element_by_name("q")
        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)

        while len(images) < total_images:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            thumbnails = driver.find_elements_by_css_selector("img.Q4LuWd")
            for img in thumbnails:
                try:
                    img.click()
                    time.sleep(2)
                    images.add(driver.find_element_by_css_selector("img.n3VNCb").get_attribute("src"))
                except:
                    continue
                if len(images) >= total_images:
                    break

        downloaded += batch_size
        print(f"Progress: Downloaded {min(downloaded, total_images)} / {total_images} images")

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for i, img_url in enumerate(images):
        try:
            img_data = requests.get(img_url).content
            with open(os.path.join(download_path, f"image_{i}.jpg"), 'wb') as img_file:
                img_file.write(img_data)
        except:
            continue

    driver.quit()
    return download_path


def load_images(image_path):
    data = []
    labels = []
    image_paths = [os.path.join(image_path, file) for file in os.listdir(image_path)]
    for image_path in image_paths:
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (64, 64))
            image = img_to_array(image)
            data.append(image)
            labels.append(0)  # Assuming all images are of the same object
        except:
            continue
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=2)  # Adjust num_classes based on your data
    return train_test_split(data, labels, test_size=0.2, random_state=42)


def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_model(trainX, trainY, testX, testY):
    model = build_model((64, 64, 3))
    history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=32)
    return model


def track_object(video_path, model):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (64, 64))
        frame_resized = img_to_array(frame_resized)
        frame_resized = np.expand_dims(frame_resized, axis=0) / 255.0
        prediction = model.predict(frame_resized)
        if np.argmax(prediction) == 1:
            cv2.putText(frame, "Object Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    base_query = "single cat"
    variations = ["photo", "image", "picture", "snapshot", "shot"]
    total_images = 1000
    download_path = "scraped_images"

    print("Starting image download...")
    download_images_with_variations(base_query, variations, total_images, download_path)
    print("Image download completed.")

    print("Loading images...")
    trainX, testX, trainY, testY = load_images(download_path)
    print(f"Loaded {len(trainX)} training images and {len(testX)} testing images.")

    print("Training model...")
    model = train_model(trainX, trainY, testX, testY)
    print("Model training completed.")

    # Example usage for tracking object
    # Replace 'video.mp4' with the path to your video file
    video_path = "video.mp4"
    print("Starting object tracking...")
    track_object(video_path, model)
    print("Object tracking completed.")
