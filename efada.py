from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import datetime
import os
import re
def web_name(url):
  match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
  if match:
    return match.group(1)
  else:
    return ""

# Function to load an image from the computer
def load_image_from_file(image_path):
  try:
    image = cv.imread(image_path)
    if image is None:
      print(f"Error: Could not load image from '{image_path}'.")
      return None
    return image
  except Exception as e:
    print(f"Error loading image: {e}")
    return None
def save_result(folder_name,object_found,image_path):
  # Open the text file for saving results
  results_file_path = os.path.join(folder_name, "detection_results.txt")
  with open(results_file_path, "a") as results_file:
        image = cv.imread(image_path)
        if object_found:
          results_file.write(f"{image_path} Object 'efada' found!\n")
        else:
          results_file.write(f"{image_path} Object 'efada' not found.\n")
def detaction_model(folder_name , image_path):
    # Load the image
    image = load_image_from_file(image_path)

    # Check if image loading was successful
    if image is None:
        exit()

    model = YOLO("best.pt")

    names = ["efada"]  # Order

    # Perform emotion detection on the loaded image
    results = model(image)

    # Process the detection results
    object_found = False  # Flag to track object detection
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(image, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            current_class = names[cls]

            # Check if detected class matches desired object
            if current_class == "efada":
                object_found = True
                # You can also break the loop here if you only care about one detection

                break

            cvzone.putTextRect(image, f"{conf} {current_class}", (x1, y1 + 20))

    # Display the processed image
    cv.imshow("logo Detection", image)
    save_result(folder_name, object_found, image_path)
    cv.destroyAllWindows()



def screenshot(url):
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)


    driver.get(url)

    driver.maximize_window()
    time.sleep(5)
    # Create folder name with current date
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder_name = f"Screenshots_{web_name(url)}_{current_datetime}"
    # Check if folder exists, otherwise create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    screenshot_path1 = os.path.join(folder_name, f"{web_name(url)}1_{current_datetime}.png")
    screenshot = driver.save_screenshot(screenshot_path1)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    time.sleep(2)

    screenshot_path2 = os.path.join(folder_name, f"{web_name(url)}2_{current_datetime}.png")
    screenshot = driver.save_screenshot(screenshot_path2)

    driver.quit()
    return folder_name,screenshot_path1,screenshot_path2


url = "https://www.focallurearabia.com/"
folder_name ,screenshot_path1 ,screenshot_path2=screenshot(url)
detaction_model(folder_name , screenshot_path1)
detaction_model(folder_name , screenshot_path2)