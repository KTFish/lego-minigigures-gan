# This script aims to scrape Lego Minifigures Images using Selenium
# It uses helpfunctions stored in `/scraping_scripts` directory. In this directory you will find the follwoing files:
# - config.py - stores all constants and usefull variables.
# - configure_webdriver.py
# - scraping.py - all the logic of scraping a page. Helper functions and the main loop can be found there.
# - utils.py - other helper functions.

from scraping_scripts import utils, config, scraping
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

os.environ['PATH'] += '.'  # Add webdriver to PATH

# Create a folder for dataset    
utils.create_folder(folder_name='dataset', root="./")

# Initialize and access webdriver
# driver = get_webdriver()

# Run the scraiping script
scraping.run()
