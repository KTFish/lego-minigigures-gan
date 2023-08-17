from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
import config
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

def get_webdriver(window_size:str="--window-size=1920, 1080") -> webdriver:
    """Initializes and returns a webdriver. Functions main purpose is to simplify the main scrip.
    """
    # Configure the size of the window
    options = Options()
    options.add_argument(window_size)

    # Intitialize the driver
    driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))
    driver.get(config.base_path)
    driver.maximize_window() # Fit the window to your screen
    assert type(driver) == webdriver

    return driver
