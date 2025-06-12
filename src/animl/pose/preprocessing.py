"""
Pose estimation dataset preprocessing
"""
import pandas as pd
import json
from typing import Optional
from animl.split import train_val_test

def process_dataset(file_path: dict, dataset_name: str):
    '''
        Processes pose estimation datasets in COCO format for machine learning.

    Args:
        - file_path (str): path to dataset
        - dataset_name (str): name of dataset

    Returns:
        - final_df
    '''
    image_list = []
    with open(file_path, 'r') as file:
        dataset = json.load(file)
    if dataset_name == "stanford-dogs":
        for image in enumerate(dataset):
            full_image = {}
            full_image["image_id"] = image[0]
            full_image["file_name"] = f'/mnt/machinelearning/Viewpoint/stanford_dogs/Images/{image[1]["img_path"]}'
            full_image["width"] = image[1]["img_width"]
            full_image["height"] = image[1]["img_height"]
            full_image["dataset_name"] = dataset_name
            full_image["bbox1"] = image[1]["img_bbox"][0]
            full_image["bbox2"] = image[1]["img_bbox"][1]
            full_image["bbox3"] = image[1]["img_bbox"][2]
            full_image["bbox4"] = image[1]["img_bbox"][3]
            full_image["species"] = "dog"

            front_indices = [0, 1, 2, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            back_indices = [3, 4, 5, 9, 10, 11, 12, 13]
            front_half = [image[1]["joints"][i][0:3] for i in range(0, len(image[1]["joints"])) if i in front_indices]
            back_half = [image[1]["joints"][i][0:3] for i in range(0, len(image[1]["joints"])) if i in back_indices]

            try:
                front_average = sum(x[0] for x in front_half if x[2] == 1) / sum(x[2] for x in front_half if x[2] == 1)
            except ZeroDivisionError as e:
                    front_average = 0
            try:
                back_average = sum(x[0] for x in back_half if x[2] == 1) / sum(x[2] for x in back_half if x[2] == 1)
            except ZeroDivisionError:
                    back_average = 0

            full_image["front_average"] = front_average
            full_image["back_average"] = back_average

            if front_average < back_average:
                    full_image["viewpoint"] = "left"
            elif front_average > back_average:
                    full_image["viewpoint"] = "right"
            else:
                    full_image["viewpoint"] = "undefined"
            image_list.append(full_image)
        final_df = pd.DataFrame.from_dict(image_list)

        return final_df
    if dataset_name == "animal-pose":
        ann_list = []
        for id, file in dataset.get("images", []).items():
            full_image = {}
            full_image["id"] = int(id)
            full_image["file_name"] = f'/mnt/machinelearning/Viewpoint/animalpose_image_part2/images/{file}'
            full_image["width"] = "None"
            full_image["height"] = "None"
            full_image["dataset_name"] = dataset_name
            image_list.append(full_image)
        for annotation in dataset.get("annotations", []):
            annotation["bbox1"] = annotation["bbox"][0]
            annotation["bbox2"] = annotation["bbox"][1]
            annotation["bbox3"] = annotation["bbox"][2]
            annotation["bbox4"] = annotation["bbox"][3]
        
            front_indices = [0, 1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 17, 18]
            back_indices = [7, 8, 11, 12, 15, 16, 19]
            front_half = [annotation["keypoints"][i][0:3] for i in range(0, len(annotation["keypoints"])) if i in front_indices]
            back_half = [annotation["keypoints"][i][0:3] for i in range(0, len(annotation["keypoints"])) if i in back_indices]
            try:
                front_average = sum(x[0] for x in front_half if x[2] == 1) / sum(x[2] for x in front_half if x[2] == 1)
            except ZeroDivisionError as e:
                    front_average = 0
            try:
                back_average = sum(x[0] for x in back_half if x[2] == 1) / sum(x[2] for x in back_half if x[2] == 1)
            except ZeroDivisionError:
                    back_average = 0

            annotation["front_average"] = front_average
            annotation["back_average"] = back_average
            if front_average < back_average:
                    annotation["viewpoint"] = "left"
            elif front_average > back_average:
                    annotation["viewpoint"] = "right"
            else:
                    annotation["viewpoint"] = "undefined"
            ann_list.append(annotation)
    
        category_list = []
        for category in dataset.get("categories", []):
            new_category = {}
            new_category["species"] = category["name"]
            new_category["id"] = category["id"]    
            category_list.append(new_category)
        
        image_df = pd.DataFrame.from_dict(image_list)
        ann_df = pd.DataFrame.from_dict(ann_list)
        result = image_df.merge(ann_df, left_on='id', right_on='image_id')
        category_df = pd.DataFrame.from_dict(category_list)
        merge = result.merge(category_df, left_on='category_id', right_on='id')
        final_df = merge.drop(['bbox', 'num_keypoints', 'keypoints', 'category_id', 'id_x', 'id_y'], axis=1)

        return final_df

    for i, image in enumerate(dataset.get("images", [])):
        image["dataset_name"] = dataset_name
        if dataset_name == "ap-10k":
            image["file_name"] = f'/mnt/machinelearning/Viewpoint/ap-10k/data/{image["file_name"]}'
        elif dataset_name == "ATRW":
            image["filename"] = f'/mnt/machinelearning/Viewpoint/ATRW/train/{image["filename"]}'
        image_list.append(image)
    
    ann_list = []
    categories = dataset.get("categories")
    
    keypoints = categories[0].get("keypoints")

    for annotation in dataset.get("annotations", []):
        annotation["bbox1"] = annotation["bbox"][0]
        annotation["bbox2"] = annotation["bbox"][1]
        annotation["bbox3"] = annotation["bbox"][2]
        annotation["bbox4"] = annotation["bbox"][3]    
        ann_keypoints = annotation["keypoints"]

        xyv = [ann_keypoints[i:i+3] for i in range(0, len(ann_keypoints), 3)]
        keypoints_xyv = list(zip(keypoints, xyv))
        if dataset_name == "ap-10k": 
            front_half = keypoints_xyv[0:10]
            back_half = keypoints_xyv[11:16]
        else:
            front_half = keypoints_xyv[0:6]
            back_half = keypoints_xyv[7:14]
        try:
            front_average = sum(x[1][0] for x in front_half if x[1][2] == 2) / sum(x[1][2] for x in front_half if x[1][2] == 2)
        except ZeroDivisionError as e:
                front_average = 0

        try:
            back_average = sum(x[1][0] for x in back_half if x[1][2] == 2) / sum(x[1][2] for x in back_half if x[1][2] == 2)
        except ZeroDivisionError:
                back_average = 0
        annotation["front_average"] = front_average
        annotation["back_average"] = back_average

        if front_average < back_average:
                annotation["viewpoint"] = "left"
        elif front_average > back_average:
                annotation["viewpoint"] = "right"
        else:
             annotation["viewpoint"] = "undefined"
        if dataset_name == "ATRW":
            annotation["species"] = "tiger"
        ann_list.append(annotation)

    category_list = []
    if dataset_name == "ap-10k":
        for category in dataset.get("categories", []):
            new_category = {}
            new_category["species"] = category["name"]
            new_category["id"] = category["id"]    
            category_list.append(new_category)
    
    image_df = pd.DataFrame.from_dict(image_list)
    ann_df = pd.DataFrame.from_dict(ann_list)
    result = image_df.merge(ann_df, left_on='id', right_on='image_id')
    if dataset_name == "ap-10k":
        category_df = pd.DataFrame.from_dict(category_list)
        result = result.merge(category_df, left_on='category_id' ,right_on='id')
        final_df = result.drop(['license', 'background', 'iscrowd', 'area', 'bbox', 'num_keypoints', 'keypoints', 'id_y', 'id_x', 'id', 'category_id'], axis=1)
    elif dataset_name == "ATRW":
            final_df = result.drop(['iscrowd', 'area', 'bbox', 'num_keypoints', 'keypoints', 'id_y', 'id_x', 'category_id'], axis=1)
            final_df = final_df.rename(columns={'filename': 'file_name'})
    return final_df

def merge_and_split(df_list, species_to_remove: Optional[list[str]] = None, imbalanced_class: Optional[str] = None): #concatenate datasets 
    '''
        Merges separate dataframes into one and splits the combined dataset into training and validation datasets.

    Args:
        - df_list (str): list of dataframes to merge and split
        - species_to_remove (str): names of species to be filtered out of the dataset.
        - imbalanced_class (str): class to be balanced in the dataset
    '''
    df = pd.concat(df_list)[pd.concat(df_list)['viewpoint'] != 'undefined']
    df.drop_duplicates(keep='first')

    if species_to_remove:
        df = df[~df['species'].isin(species_to_remove)]

    if imbalanced_class is not None:
        imbalanced = df.groupby(imbalanced_class).size()
        max = imbalanced.idxmax()
        min = imbalanced.idxmin()
        undersampled_result = df[df[imbalanced_class] == max].sample(df[imbalanced_class].value_counts()[min])
        manifest = pd.concat([undersampled_result, df[df[imbalanced_class] == min]])
    
    train_val_test(manifest=manifest,out_dir='/mnt/machinelearning/Viewpoint/test/',label_col='dataset_name',percentage=(0.8,0.2,0.0), repeat_column='file_name')

if __name__ == "__main__":
    file_paths = {"/mnt/machinelearning/Viewpoint/annotations/StanfordExtra_v12.json": "stanford-dogs", "/mnt/machinelearning/Viewpoint/annotations/merged_ap10k.json": "ap-10k",
                   "/mnt/machinelearning/Viewpoint/annotations/merged_ATRW.json": "ATRW", "/mnt/machinelearning/Viewpoint/annotations/animal-pose.json": "animal-pose"}
    df_list = [process_dataset(file, dataset) for file, dataset in file_paths.items()]
    remove = ['hippo', 'otter', 'uakari', 'monkey', 'chimpanzee', 'noisy night monkey', 'spider monkey', 'alouatta']
    merge_and_split(df_list, remove, 'viewpoint')
    