import shutil
import config
import utils
import os
import re
import time
import shutil
import requests
from typing import List, Dict, Tuple
from random import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


driver = config.DRIVER

def get_categories()-> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Get the list of all categories of minifigures.
    """

    # Find the "Category" button and click on it
    button = driver.find_element(By.XPATH, "//span[@class='selectboxit-text' and text()='Category']")
    button.click()

    # Find all the <li> elements within the dropdown menu
    category_elements = driver.find_elements(By.XPATH, "//ul[@class='selectboxit-options selectboxit-list']//li")

    # Scrape the links and category names and store them in a dictionary
    category_to_link = {} # Maps category name to relative link
    category_to_count = {} # Maps category name to the number of minifigures in that category
    for element in category_elements:
        # Extract the link
        link = element.get_attribute("data-val")
        category = clean_category_name(element.text)
        category_to_link[category] = link
        
        # Extract the category count
        counts = re.findall(r'\d+', element.text)
        if len(counts) >= 1:
            category_to_count[category] = int(counts[0])
    
    try:
        del category_to_link['']
    except KeyError:
        pass
    
    return category_to_link, category_to_count

def scrape(category: str) -> None:
    """
    Function for scraping a single page. It scrapes the minifigures images and names.
    """
    print(f"Scraping images...")

    # Find things on page
    # all_img = driver.find_elements(By.XPATH, "//img")
    all_img = driver.find_elements(By.XPATH, "//article[@class='set']//img")
    all_minifigure_names = driver.find_elements(By.XPATH, "//article[@class='set']/div[@class='meta']/h1/a")


    # Check if the number of images and captions is equal
    assert len(all_img) == len(all_minifigure_names), f"all_img ({len(all_img)}) and all_minifigure_names ({len(all_minifigure_names)}) should lenghts be equal!"

    # Loop trough all images on page
    for idx, image in enumerate(all_img):
        src = image.get_attribute('src')
        response = requests.get(src, stream=True)

        # Get the original name (it can be usefull for Condiditonal GANs)
        minifigure_name = utils.clean_name(all_minifigure_names[idx].get_attribute('text'))

        # Save image as .png
        save_path = f'./dataset/{category}/{minifigure_name}.png'
        with open(save_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
            print(f"Image saved to: {save_path}")

def scrape_all_pages(url: str, category:str) -> None:
    """
    Gets a url to a specific minifigures category and scrapes all the images (from sub page 1, 2, 3, ... n).
    """

    # Visit the first page
    driver.get(url)

    while True:

        # Call the scrape() function on the current page
        scrape(category=category)

        # Find the 'next' button and check if it is disabled (last page)
        try:
            next_button = driver.find_element(By.CSS_SELECTOR, "li.next a")
            next_disabled = next_button.get_attribute("aria-disabled")
        except NoSuchElementException:
            print("Next button not found.")
            break
            

        # If 'next' button is disabled, it means we reached the last page, so break the loop
        if next_disabled == "true":
            break

        # Otherwise, click the 'next' button to go to the next page
        driver.execute_script("arguments[0].click();", next_button)

        # Wait for a few seconds (you can adjust the time if needed)
        time.sleep(5)
    
    # Exiting driver after scraping all pages from given category
    #driver.quit()


def run_main_scraping_loop():
    # Scrape category names, links and number of minifigures per category
    category_to_link, category_to_number = utils.get_categories()

    # Print out information about the number of categories and images to scrape
    number_of_all_images = sum([x for x in category_to_number.values()])
    print(f"There are {number_of_all_images} images to scrape divided into {len(category_to_link.values())} categories.")


    ### Main Scraping Loop
    print(f"Scraping lego minifigure images started.")
    for category, link in category_to_link.items():
        # Create folder to store images
        path = create_folder(folder_name=category, root="./dataset")
        print(path, category)
        # Check if the category was scraped before if not scrape if
        desired_count = category_to_number[category]
        real_count = count_images_in_directory(path)
        print(f"For now there are {real_count} scraped images from {desired_count} images.")
        if desired_count != real_count:
            # Access the full link
            full_link = "https://brickset.com" + link

            print(f"Current scraped category: {category.upper()} from: {full_link}")

            # Scrape all images from given category and close the page
            scrape_all_pages(full_link, category)

            # Open the main page again
            driver.get(base_path)
            driver.maximize_window() # Fit the window to your screen
        else:
            print(f"All images ({real_count}) from category {category} have already been scraped.\
                Skippin scraping {category}. Scraiping the next category...")
        
        # After scraping the whole category go to the next one..