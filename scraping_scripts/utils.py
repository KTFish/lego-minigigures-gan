import os
import re

def create_folder(folder_name: str, root: str) -> str:
    full_path = os.path.join(root, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True, )
    return full_path

def clean_name(minifigure_name: str) -> str:
    invalid_chars = r'.[\\/:\*\?"<>\|]\& \''
    for c in invalid_chars:
        minifigure_name = minifigure_name.replace(c, '-')
    return re.sub(r'-+', '-', minifigure_name)
    
def clean_category_name(category: str) -> str:
    """
    Cleans the category name
    """

    # Replace all numbers with empty characters
    result = re.sub(r'\d+', '', category)

    # Replace parentheses '()' with empty characters
    result = result.replace('(', '').replace(')', '')

    # Replace all spaces with dash
    result = result.strip().replace(' ', '-')
    result = result.replace('.','')
    
    return result.lower()

def count_images_in_directory(dir_path: str) -> int:
    count = 0
    for path in os.listdir(dir_path):
        # Check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return count
