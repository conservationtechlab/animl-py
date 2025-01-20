# -*- coding: utf-8 -*-
"""
Rename and resize images, create a subject set and upload new subjects to it
Created on Mon Jul 11 15:34:53 2016

@author: Mathias Tobler

Copyright (C) 2017 Mathias Tobler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Based on code by:
________________________________________________
Stijn Calders
Space Physics - Space Weather

Royal Belgian Institute for Space Aeronomy (BIRA-IASB)
Ringlaan 3
B-1180 Brussels
BELGIUM

phone  : +32 (0)2 373.04.19
e-mail : stijn.calders@aeronomie.be
web    : www.aeronomie.be
________________________________________________
"""

from panoptes_client import SubjectSet, Subject, Project, Panoptes
import os
import time
import sys
import pandas as pd
from datetime import datetime
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def copy_image(image, out_dir, image_height):
    """
        Copy and resize an image for upload

        Args:
            - image (DataFrame): image to upload, row from list
            - out_dir (str): location to save temporary resized images to
            - image_height (int): height in px to resize images before upload
        Returns:
            - out_file: the path of the copied image for upload
    """
    if not isinstance(image, pd.Series):
        raise TypeError(f"Expected Pandas Series for image, got {type(image)}")
    if not isinstance(out_dir, str):
        raise TypeError(f"Expected string for out_dir, got {type(out_dir)}")
    if not isinstance(image_height, int):
        raise TypeError(f"Expected int for image_height, got {type(image_height)}")
    img = Image.open(image.FilePath)
    try:
        exif = img.info['exif']
    except KeyError:
        return None

    wpercent = (image_height / float(img.size[1]))
    imagewidth = int((float(img.size[0]) * float(wpercent)))
    img = img.resize((imagewidth, image_height), Image.ANTIALIAS)
    out_file = out_dir + image.FileName
    img.save(out_file, "JPEG", exif=exif)
    return out_file


def create_SubjectSet(project, subject_set_name):
    """
        Create a new subject set under a given project

        Args:
            - project (int): Zooniverse project number
            - subject_set_name (str): name of new subject set

        Return:
            - subject_set: the number associated with the new subject set
    """
    subject_set = SubjectSet()
    subject_set.links.project = project
    subject_set.display_name = subject_set_name
    subject_set.save()
    return subject_set


def connect_to_Panoptes(usr, pw):
    """
        Login to Zooniverse through panoptes

        Args:
            - usr (str): Zooniverse account username
            - pw (str): Zooniverse account password
    """
    Panoptes.connect(username=usr, password=pw)
    print("Connected to Zooniverse")


def upload_to_Zooniverse_Simple(project_name, subject_set_name, images,
                                temp_dir, image_height=700):
    """
        Upload single images to Zooniverse

        Args:
            - project_name (str): name of project on zooniverse
            - subject_set_name (str): name of subject set to upoad into
            - images (DataFrame): list of images to upload
            - temp_dir (str): location to save temporary resized images to
            - image_height (int): height in px to resize images before upload
    """
    sys.stdout.flush()
    # Create a new subject set or append the subjects to an existing one
    project = Project.find(project_name)
    print("Connected to ", project)
    subject_set = SubjectSet.find(int(subject_set_name))
    print(subject_set)
    print("Images to upload: ", images)
    print("Uploading images...")
    for _, infile in images.iterrows():
        if not os.path.exists(infile.FileName):
            print(infile.FileName)
        subject = Subject()  # create new subject
        subject.links.project = project
        outfile = copy_image(infile, temp_dir, image_height)
        if outfile is None:
            continue
        subject.add_location(outfile)
        subject.metadata['OrigName'] = infile.FileName
        subject.metadata['!OrigFolder'] = infile.FilePath
        subject.metadata['DateTime'] = str(infile.FileModifyDate)
        subject.metadata['#machine_prediction'] = infile.ZooniverseCode  # map to zooniverse name
        subject.metadata['#machine_confidence'] = infile.confidence
        subject.save()
        subject_set.add(subject)


def upload_to_Zooniverse(project_name, subject_set_name, images, temp_dir,
                         max_seq=1, max_time=0, image_height=700):
    """
        Upload sets of images to Zooniverse

        Args:
            - project_name (str): name of project on zooniverse
            - subject_set_name (str): name of subject set to upoad into
            - images (DataFrame): list of images to upload
            - temp_dir (str): location to save temporary resized images to
            - max_seq (int): maximum number of images within a set
            - max_time (int): maximum duration in seconds to consider a set
            - image_height (int): height in px to resize images before upload
    """
    sys.stdout.flush()
    # Create a new subject set or append the subjects to an existing one
    project = Project.find(project_name)
    print("Connected to ", project)
    subject_set = SubjectSet.find(int(subject_set_name))

    print(subject_set)
    print("Uploading images...")

    # print(images.columns.values)
    images['DateTime'] = pd.to_datetime(images['DateTime'], format="%Y:%m:%d %H:%M:%S")

    subject = None
    dDateTime = 0
    si = 0  # set sequence counter to 0
    for _, infile in images.iterrows():
        bProc = False
        try:
            currentDateTime = time.mktime(time.strptime(infile.DateTime, "%Y-%m-%d %H:%M:%S"))
        except TypeError:
            currentDateTime = datetime.strptime(str(infile.DateTime), "%Y-%m-%d %H:%M:%S")

        # add image to current sequence
        if si > 0 and si < max_seq and (currentDateTime - dDateTime) <= max_time and subject.metadata[
                '#machine_prediction'] == infile.ZooniverseCode:
            # print("Continue")
            outfile = copy_image(infile, temp_dir, image_height)
            if outfile is None:
                continue
            subject.add_location(outfile)
            subject.metadata['OrigName'] = subject.metadata['OrigName'] + ";" + infile.FileName
            subject.metadata['!OrigFolder'] = subject.metadata['!OrigFolder'] + ";" + infile.FilePath
            subject.metadata['DateTime'] = subject.metadata['DateTime'] + ";" + infile.DateTime.strftime(
                "%Y:%m:%d %H:%M:%S")
            # subject.metadata['NewName'] = subject.metadata['NewName']+";"+ infile.NewName
            dDateTime = currentDateTime
            si = si + 1
            bProc = True
        #  print(subject.metadata)

        # sequence has ended, current image is in a new sequence
        elif 0 < si < max_seq and ((currentDateTime - dDateTime) > max_time or subject.metadata[
                '#machine_prediction'] != infile.ZooniverseCode):
            si = max_seq
            bProc = False

            # need to start a new sequence OR maxSeq = 1
        if si == max_seq:
            # print("Upload")
            subject.save()
            subject_set.add(subject)
            si = 0

        # first image within a sequence
        if si == 0 and bProc is False:
            subject = Subject()  # create new subject
            subject.links.project = project
            dDateTime = currentDateTime
            outfile = copy_image(infile, temp_dir, image_height)
            if outfile is None:
                continue
            subject.add_location(outfile)
            # You can set whatever metadata you want, or none at all
            subject.metadata['OrigName'] = infile.FileName
            subject.metadata['!OrigFolder'] = infile.FilePath
            subject.metadata['DateTime'] = infile.DateTime.strftime("%Y:%m:%d %H:%M:%S")
            subject.metadata['#machine_prediction'] = infile.ZooniverseCode  # map to zooniverse name
            subject.metadata['#machine_confidence'] = infile.confidence
            # subject.metadata['NewName'] = infile.NewName
            si = 1
