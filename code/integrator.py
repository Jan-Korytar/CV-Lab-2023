# -*- coding: utf-8 -*-
"""data_preprocessor.ipynb

# Data preprocessor

## Usage:

- Put this file in the **AOSTD/LFR/python** folder
- Change the input and output folders in the cell below (if needed)
- Run all cells

## Description

This notebook will process all the images, ground truths and metadata files in the imageset_path folder (defined in the cell below). It will separate all the different sets of images, get rid of incomplete or corrupt sets and then integrate them. The focal planes of the integration is defined in the last cell of this notebook. **We will need to discuss and change this.**

The output is all the integral images as well as an image_sets.json file that contains all the data about the image sets. All of this gets saved in the integral_path folder (defined in the cell below). The format of an image set in the image_sets.json file is the following:

### Image set format in image_sets.json

- GT: filename, coordinates
- Images: filenames and coordinates
- Metadata:
    - Numbers of trees per ha
    - person shape (standing (idle), sitting, laying, no person)
    - person pose (doesn't exist if there is no person)
    - person rotation (z) in radian (doesn't exist if there is no person)
    - person rotation (z) in degree (doesn't exist if there is no person)
    - ambient light
    - azimuth angle of sun light in degrees
    - compass direction of sunlight in degrees
    - ground surface temp in Kelvin
    - tree top temp in Kelvin
- Integrals: filenames and their corresponding focal planes
"""

# The input folder
# imageset_path = "test_images"
imageset_path = "../dataset/"

# The output folder
integral_path = "../integrals"

"""## Dataclass definition

In this section we just define a class that will store all the information above about a set of images (11 images + GT + metadata)
"""

from typing import List, Dict, Tuple

# Just returns an empty dict with all the keys that we need in the metadata.
def empty_metadata_dict():
    return {
        "trees_per_ha": None,
        "person_shape": None,
        "person_pose": None,
        "person_rotation_radian": None,
        "person_rotation_degree": None,
        "ambient_light": None,
        "azimuth_angle_sunlight_degrees": None,
        "compass_direction_sunlight_degrees": None,
        "ground_surface_temp_kelvin": None,
        "tree_top_temp_kelvin": None
    }

class ImageSet:
    def __init__(self, id: str = None, GT: Dict = None, images: List[Dict] = None, metadata: Dict = None):
        self.id = id
        self.GT = GT if GT is not None else {"filename": None, "coordinates": None}
        self.images = images if images is not None else [{"filename": None, "coordinates": None} for _ in range(11)]
        self.metadata = metadata if metadata is not None else empty_metadata_dict()

    def check_data(self):
        if not self.id:
            raise ValueError("ImageSet id is missing.")

        if not self.GT or not self.GT["filename"]:
            raise ValueError("Ground truth image file is missing.")

        # Check each image in the list
        for i, image in enumerate(self.images):
            if not image["filename"] or not image["coordinates"]:
                raise ValueError(f"Image {i+1} in the set is incomplete: missing filename or coordinates.")

        if not self.metadata:
            raise ValueError("Metadata is missing.")

        if self.metadata["trees_per_ha"] is None:
            raise ValueError("Trees per hectare data is missing in metadata.")

        valid_shapes = {"idle", "sitting", "laying", "no person"}
        if self.metadata["person_shape"] not in valid_shapes:
            raise ValueError(f"Invalid person shape: {self.metadata['person_shape']}. Must be one of {valid_shapes}.")

        if self.metadata["person_shape"] != "no person":
            if self.metadata["person_pose"] is None:
                raise ValueError("Person pose data is missing in metadata.")

            if self.metadata["person_rotation_radian"] is None:
                raise ValueError("Person rotation (radian) data is missing in metadata.")

            if self.metadata["person_rotation_degree"] is None:
                raise ValueError("Person rotation (degree) data is missing in metadata.")

        if self.metadata["ambient_light"] is None:
            raise ValueError("Ambient light data is missing in metadata.")

        if self.metadata["azimuth_angle_sunlight_degrees"] is None:
            raise ValueError("Azimuth angle of sunlight in degrees is missing in metadata.")

        if self.metadata["compass_direction_sunlight_degrees"] is None:
            raise ValueError("Compass direction of sunlight in degrees is missing in metadata.")

        if self.metadata["ground_surface_temp_kelvin"] is None:
            raise ValueError("Ground surface temperature in Kelvin is missing in metadata.")

        if self.metadata["tree_top_temp_kelvin"] is None:
            raise ValueError("Tree top temperature in Kelvin is missing in metadata.")

        return True

    # Ignore this it's just for debug printing
    def __str__(self):
        images_str = ',\n  '.join(str(img) for img in self.images)
        return (
            f"ImageSet(\n"
            f"  GT={self.GT},\n"
            f"  images=[\n  {images_str}\n  ],\n"
            f"  metadata={self.metadata}\n)"
        )

"""## Function definitions

Here we define a function for parsing metadata files (.txt). The metadata_key_dict dictionary converts the text in the .txt into metadata dict key names.
"""

import os
import re

metadata_key_dict = {
    "numbers of tree per ha": "trees_per_ha",
    "person shape": "person_shape",
    "person pose (x,y,z,rot x, rot y, rot z)": "person_pose",
    # jesus fucking christ
    # ok so for some godforsaken reason if there is no person then the "person pose" part of the .txt is different
    # and it makes way more sense to put "no person" in person_shape than person_pose and idk why it wasn't done that way??
    # but so anyways that's why this part is so ugly
    # this will surely not bite us in the ass later
    "person pose": "person_shape",
    "person rotation (z) in radian": "person_rotation_radian",
    "person rotation (z) in degree": "person_rotation_degree",
    "ambient light": "ambient_light",
    "azimuth angle of sun light in degrees": "azimuth_angle_sunlight_degrees",
    "compass direction of sunlight in degrees": "compass_direction_sunlight_degrees",
    "ground surface temperature in kelvin": "ground_surface_temp_kelvin",
    "tree top temperature in kelvin": "tree_top_temp_kelvin",
}

# Function to parse metadata file
def parse_metadata(metadata_path):
    metadata = empty_metadata_dict()

    GT_coords = {}
    image_coords = [None] * 11

    with open(metadata_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('img'):
                img_name, coords_str = line.split(' (', 1)
                img_index = img_name.split('_')[1]
                coords = tuple(map(float, coords_str.rstrip(')').split(', ')))
                if img_index == 'GT':
                    GT_coords = coords
                else:
                    image_coords[int(img_index) - 1] = coords
            elif '=' in line:
                text_key, value = line.split('=', 1)
                value = value.strip()
                text_key = text_key.strip()

                if text_key not in metadata_key_dict:
                    raise ValueError(f"Invalid key {text_key} in file {metadata_path}")

                key = metadata_key_dict[text_key]

                # Handle special cases and set the attribute
                if key == 'person_pose':
                    metadata[key] = tuple(map(float, value.split()))
                elif key in ['person_shape', 'trees_per_ha']:
                    metadata[key] = value
                else:
                    metadata[key] = float(value)

    return metadata, GT_coords, image_coords

"""Here we define the main function for processing our images. It goes through the files in our folder, finds all files that belong to one set, separates the images, the GT and the metadata files and saves their data as an ImageSet. It also does some error checking and gets rid of every image set that is incomplete before returning a list of image sets that are usable."""

from collections import defaultdict
import json

# Main processing function
def process_image_sets(folder_path):
    image_sets = defaultdict(ImageSet)
    processed_image_sets = []

    for filename in os.listdir(folder_path):
        # We need this disgusting godawful check bc random files and folders can mess this up
        # They can still mess this up but at least it's less likely
        if not os.path.isfile(os.path.join(folder_path, filename)) and not (filename.endswith('.png') or filename.endswith('.txt')):
            continue

        # First we get or create the corresponding ImageSet
        split_filename = filename.split('_')
        set_id = split_filename[0] + "_" + split_filename[1]

        image_set = image_sets[set_id]
        image_set.id = set_id

        # If the file we're processing is a GT or an image
        if filename.endswith('.png'):
            if 'GT' in filename:
                image_set.GT["filename"] = filename
            else:
                img_index = int(split_filename[3])
                image_set.images[img_index]["filename"] = filename
        # If the file we're processing is the metadata
        elif filename.endswith('.txt'):
            metadata_path = os.path.join(folder_path, filename)
            metadata, GT_coords, image_coords = parse_metadata(metadata_path)

            # We set the metadata and the coordinates
            image_set.metadata = metadata
            image_set.GT["coordinates"] = GT_coords
            for i, coords in enumerate(image_coords):
                if coords:
                    image_set.images[i]["coordinates"] = coords

            # I would think this is not needed but the code doesn't work without this line
            image_sets[set_id] = image_set

    for image_set in image_sets.values():
        try:
            if image_set.check_data():
                # __dict__ to convert the class into a dict (disgusting)
                processed_image_sets.append(image_set.__dict__)
        except ValueError as e:
            print(f"Skipped {image_set.id} as there was an error during processing: {e}")

    return processed_image_sets

"""## Process the image sets"""

# This variable stores all the image sets
image_sets = process_image_sets(imageset_path)

# Uncomment this to see what an image_set looks like
# print(json.dumps(image_sets[0], indent = 4))

"""# Integrating the dataset

Now that we have processed and filtered the data, it's time to create integral images.

### Predefined functions
**Do not touch these**, below are certain functions required to convert the poses to a certain format to be compatabile with the AOS Renderer.
"""

def eul2rotm(theta) :
    s_1 = math.sin(theta[0])
    c_1 = math.cos(theta[0])
    s_2 = math.sin(theta[1])
    c_2 = math.cos(theta[1])
    s_3 = math.sin(theta[2])
    c_3 = math.cos(theta[2])
    rotm = np.identity(3)
    rotm[0,0] =  c_1*c_2
    rotm[0,1] =  c_1*s_2*s_3 - s_1*c_3
    rotm[0,2] =  c_1*s_2*c_3 + s_1*s_3

    rotm[1,0] =  s_1*c_2
    rotm[1,1] =  s_1*s_2*s_3 + c_1*c_3
    rotm[1,2] =  s_1*s_2*c_3 - c_1*s_3

    rotm[2,0] = -s_2
    rotm[2,1] =  c_2*s_3
    rotm[2,2] =  c_2*c_3

    return rotm

def createviewmateuler(eulerang, camLocation):

    rotationmat = eul2rotm(eulerang)
    translVec =  np.reshape((-camLocation @ rotationmat),(3,1))
    conjoinedmat = (np.append(np.transpose(rotationmat), translVec, axis=1))
    return conjoinedmat

def divide_by_alpha(rimg2):
        a = np.stack((rimg2[:,:,3],rimg2[:,:,3],rimg2[:,:,3]),axis=-1)
        return rimg2[:,:,:3]/a

def pose_to_virtualcamera(vpose ):
    vp = glm.mat4(*np.array(vpose).transpose().flatten())
    #vp = vpose.copy()
    ivp = glm.inverse(glm.transpose(vp))
    #ivp = glm.inverse(vpose)
    Posvec = glm.vec3(ivp[3])
    Upvec = glm.vec3(ivp[1])
    FrontVec = glm.vec3(ivp[2])
    lookAt = glm.lookAt(Posvec, Posvec + FrontVec, Upvec)
    cameraviewarr = np.asarray(lookAt)
    #print(cameraviewarr)
    return cameraviewarr

"""### Starting the AOS renderer
Probably also shouldn't touch these
"""

import pyaos

if not os.path.exists(integral_path):
    os.mkdir(integral_path)

# Start the AOS Renderer

# Resolution and field of view. This should not be changed.
w, h, fovDegrees = 512, 512, 50
render_fov = 50

# idk what any of this does, it opens a random useless frozen window
if 'window' not in locals() or window == None:
    window = pyaos.PyGlfwWindow(w, h, 'AOS')

aos = pyaos.PyAOS(w, h, fovDegrees)

set_folder = './'
aos.loadDEM( os.path.join(set_folder, 'zero_plane.obj'))

"""### Configuring the integrator

Just a bunch of stuff that you should also probably not touch
"""

import re
import numpy as np
import math

number_of_images = 11

# These are the x and y positions of the images. It's in the form of [[x_positions],[y_positions]]
ref_loc = [[5,4,3,2,1,0,-1,-2,-3,-4,-5], [0,0,0,0,0,0,0,0,0,0,0]]

# Z values of the images (which is the height the drone was flying at)
altitude_list = [35,35,35,35,35,35,35,35,35,35,35]

# The index of which image we should integrate to. 5 means that we integrate to the image in the center
center_index = 5

site_poses = []
for i in range(number_of_images):
    EastCentered = (ref_loc[0][i] - 0.0) # get MeanEast and Set MeanEast
    NorthCentered = (0.0 - ref_loc[1][i]) # get MeanNorth and Set MeanNorth
    M = createviewmateuler(np.array([0.0, 0.0, 0.0]),np.array( [ref_loc[0][i], ref_loc[1][i], - altitude_list[i]] ))
    ViewMatrix = np.vstack((M, np.array([0.0,0.0,0.0,1.0],dtype=np.float32)))
    camerapose = np.asarray(ViewMatrix.transpose(),dtype=np.float32)
    site_poses.append(camerapose)  # site_poses is a list now containing all the poses of all the images in a certain format that is accecpted by the renderer.

numbers = re.compile(r'(\d+)')
# This is for later when we sort the image filenames so they're given to the integrator in order
def numericalSort(value):
    try:
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
    except TypeError as e:
        print(f'TypeError for value "{value}", {e}')
    return parts

"""### The actual integration part"""

import cv2
import glm

# Average person heights (circa):
# Standing: 170cm
# Sitting: 90cm
# Laying: 42cm


# For now these are just random numbers but we need to refine this
# Also we probably want to take into account the person's position (standing, sitting, laying or no person)
focal_planes = [-0.1, -0.8, -1.6]
# focal_planes = [0]

# This is just for printing the progress of where we're at
image_set_count = len(image_sets)
print("Integrating...")

for image_set_i, image_set in enumerate(image_sets):
    image_filenames = [image['filename'] for image in image_set['images']]

    image_filenames.sort(key=numericalSort)
    image_filenames = [os.path.join(imageset_path, filename) for filename in image_filenames]

    image_list = []

    # We read the images as pixels
    for img in image_filenames:
        n = cv2.imread(img)
        image_list.append(n)

    # We need to call this every time to clear the previous views
    aos.clearViews()

    for i, image in enumerate(image_list):
        aos.addView(image, site_poses[i], "idk what to put here")
        
    # We generate an integral image for every focal plane
    for focal_plane in focal_planes:
        # No clue what this does
        aos.setDEMTransform([0, 0, focal_plane])

        integral_file_name = f'{image_set["id"]}_integral_f{focal_plane}.png'

        # Also no idea about this but yay we're finished
        proj_RGBimg = aos.render(pose_to_virtualcamera(site_poses[center_index]), render_fov)
        tmp_RGB = divide_by_alpha(proj_RGBimg)
        cv2.imwrite(os.path.join(integral_path, integral_file_name), tmp_RGB)

        # Check if 'integral' key exists in image_set
        if 'integrals' not in image_set:
            image_set['integrals'] = []

        # New dict with filename and focal plane to potentially add to the 'integral' list
        new_dict = {'filename': integral_file_name, 'focal_plane': focal_plane}

        # Check if the new_dict is not already in the list (just bc if I run this cell twice, without this the image_sets gets fucked up
        if not any(d['filename'] == integral_file_name and d['focal_plane'] == focal_plane for d in image_set['integrals']):
            image_set['integrals'].append(new_dict)

    # Just a little progress print so we know where we're at
    progress_percentage = (image_set_i + 1) / image_set_count * 100
    if image_set_i == 0 or progress_percentage >= 10 * (image_set_i // (0.1 * image_set_count) + 1):
        print(f"{progress_percentage:.0f}% done")

print('Integrating done!')

# Finally let's also save the dat#a about the image sets in a json format
#output_file_path = os.path.join(integral_path, "image_sets.json")

#with open(output_file_path, 'w') as file:
#    json.dump(image_sets, file, indent=4)

