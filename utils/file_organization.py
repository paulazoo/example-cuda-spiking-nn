import re
import os

def extract_number(string_with_trailing_number):
    string_with_number_no_extension = os.path.splitext(string_with_trailing_number)[0]
    match = re.search(r'(\d+)$', string_with_number_no_extension)
    return int(match.group(1)) if match else float('inf')  # Use inf to push invalid files to the end



def get_image_filenames(image_folder, filename_prefix):
    unsorted_image_filenames = [
        img for img in os.listdir(image_folder)
        if img.endswith('.bin') and img.startswith(filename_prefix)
        ]
    image_filenames = sorted(unsorted_image_filenames, key=extract_number)
    return image_filenames
