import xml.etree.ElementTree as ET
from os import listdir, mkdir
from matplotlib.pyplot import figure, plot, subplot, imread, imshow
from numpy import random
from itertools import chain
from multiprocessing import Pool

data_dir = "../data/"
breed_list = listdir(f"{data_dir}images/")


def crop_image(breed, dog):
    tree = ET.parse(f"{data_dir}annotations/{breed}/{dog}")
    bounding_box = tree.getroot().findall("object")[0].find("bndbox")
    xmin = int(bounding_box.find("xmin").text)
    xmax = int(bounding_box.find("xmax").text)
    ymin = int(bounding_box.find("ymin").text)
    ymax = int(bounding_box.find("ymax").text)
    return imread(f"{data_dir}images/{breed}/{dog}.jpg")[ymin:ymax, xmin:xmax, :]


def plot_sample_image_vs_crop():
    figure(figsize=(20, 20))
    for i in range(4):
        subplot(421 + (i * 2))
        breed = random.choice(breed_list)
        f"{data_dir}images/{breed}/{dog}.jpg"
        dog = random.choice(listdir(f"{data_dir}annotations/{breed}"))
        img = imread(f"{data_dir}images/{breed}/{dog}.jpg")
        imshow(img)
        tree = ET.parse(f"{data_dir}annotations/{breed}/{dog}")
        boundingBox = tree.getroot().findall("object")[0].find("bndbox")
        xmin = int(boundingBox.find("xmin").text)
        xmax = int(boundingBox.find("xmax").text)
        ymin = int(boundingBox.find("ymin").text)
        ymax = int(boundingBox.find("ymax").text)
        plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
        crop_img = crop_image(breed, dog)
        subplot(422 + (i * 2))
        imshow(crop_img)


def save_image(breed_and_file):
    global data_dir
    img = open(f"{data_dir}images/{breed_and_file}.jpg")
    tree = ET.parse(f"{data_dir}annotation/{breed_and_file}")
    boundingBox = tree.getroot().findall("object")[0].find("bndbox")
    xmin = int(boundingBox.find("xmin").text)
    xmax = int(boundingBox.find("xmax").text)
    ymin = int(boundingBox.find("ymin").text)
    ymax = int(boundingBox.find("ymax").text)
    img = img.crop((xmin, ymin, xmax, ymax))
    img = img.convert("RGB")
    img.save(f"{data_dir}cropped_images/{breed_and_file}.jpg")


def parallel_save_cropped_images():
    with Pool() as pool:
        pool.map(
            func=save_image,
            iterable=list(
                chain.from_iterable(
                    [
                        [
                            f"{breed}/{file}"
                            for file in listdir(f"{data_dir}annotation/{breed}")
                        ]
                        for breed in breed_list
                    ]
                )
            ),
            chunksize=50,
        )


def count_all_files_in_folder(folder_rel_path):
    original_files_counter = 0
    for breed in listdir(data_dir + folder_rel_path):
        original_files_counter += len(listdir(f"{data_dir}images/{breed}/"))
    return original_files_counter


def crop_data_if_missing_crops():
    if "cropped_images" not in listdir(data_dir):
        mkdir(data_dir + "cropped_images/")
        for breed in breed_list:
            mkdir(data_dir + "cropped_images/" + breed)
        data_dir_size = len(listdir(data_dir))
        print(
            f"Created {data_dir_size} folders to store cropped images./n"
            "Starting images save..."
        )
        parallel_save_cropped_images()
        print("Saved cropped images.")
    else:
        if count_all_files_in_folder("images/") == count_all_files_in_folder(
            "cropped_images/"
        ):
            print("Images are already cropped.")
        else:
            from shutil import rmtree

            print("Cropped-images' folder partially filled, overwriting...")
            rmtree(data_dir + "cropped_images/")
            parallel_save_cropped_images()
            print("Saved cropped images.")
