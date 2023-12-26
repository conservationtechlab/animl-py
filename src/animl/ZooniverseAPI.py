# -*- coding: utf-8 -*-
"""
Rename and resize images, create a subject set and upload new subjects to it
Created on Mon Jul 11 15:34:53 2016

@author: Mathias Tobler

Copyright (C) 2017 Mathias Tobler

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


def copy_image(img_file, out_dir, img_height):
    """
        Copy image to temporary directory, resize to correct height

        Args
            - img_file (str): file path of given image
            - out_dir (str): (temporary) directory to copy uploads to
            - img_height (int): resize height

        Returns
            - out_file: new file path
    """
    img = Image.open(img_file)
    try:
        exif = img.info['exif']
    except KeyError:
        return None

    w_percent = (img_height / float(img.size[1]))
    img_width = int((float(img.size[0]) * float(w_percent)))
    img = img.resize((img_width, img_height), Image.ANTIALIAS)
    out_file = out_dir + os.path.basename(img_file)
    img.save(out_file, "JPEG", exif=exif)
    return out_file


def create_SubjectSet(project, subject_set_name):
    """
        Create new subject set for the dataset

        Args
            - project: (int) number associated with zooniverse project
            - subject_set_name: (str) name to give subject set

        Returns
            - subject_set: new subject set class
    """
    subject_set = SubjectSet()
    subject_set.links.project = project
    subject_set.display_name = subject_set_name
    subject_set.save()
    return subject_set


def connect_to_Panoptes(usr, pw):
    """
        Connect to Panoptes client using given name and password

        Args
            - usr (str): zooniverse login username
            - pw: (str): zooniverse login password
    """
    Panoptes.connect(username=usr, password=pw)
    print("Connected to Zooniverse")


def upload_to_Zooniverse_Simple(project_name, subject_set_name, images,
                                out_dir, img_height=700):
    """
        Upload single image captures to subject set

        Args
            - project_name: (str) Zooniverse Project Name
            - subject_set_name: (str) name of dataset upload
            - images: dataframe of images to upload
            - out_dir: (str) temporary directory to copy uploads to
            - img_height: (int), maximum img height for resize, defaults to 700px
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
        outfile = copy_image(infile.FileName, out_dir, img_height)
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


def upload_to_Zooniverse(project_name, subject_set_name, images, outdir,
                         maxSeq=1, maxTime=0, imageheight=700):
    """
        Upload single image captures to subject set

        Args
            - project_name: (str) Zooniverse Project Name
            - subject_set_name: (str) name of dataset upload
            - images: dataframe of images to upload
            - out_dir: (str) temporary directory to copy uploads to
            - img_height: (int), maximum img height for resize, defaults to 700px
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

    si = 0  # set sequence counter to 0
    subject = None
    lastDateTime = 0
    for _, infile in images.iterrows():
        bProc = False
        try:
            currentDateTime = time.mktime(time.strptime(infile.DateTime, "%Y-%m-%d %H:%M:%S"))
        except TypeError:
            currentDateTime = datetime.strptime(str(infile.DateTime), "%Y-%m-%d %H:%M:%S")

        # add image to current sequence
        if si > 0 and si < maxSeq and (currentDateTime - lastDateTime) <= maxTime and subject.metadata[
                '#machine_prediction'] == infile.ZooniverseCode:
            # print("Continue")
            outfile = copy_image(infile, outdir, imageheight)
            if outfile is None:
                continue
            subject.add_location(outfile)
            subject.metadata['OrigName'] = subject.metadata['OrigName'] + ";" + infile.FileName
            subject.metadata['!OrigFolder'] = subject.metadata['!OrigFolder'] + ";" + infile.FilePath
            subject.metadata['DateTime'] = subject.metadata['DateTime'] + ";" + infile.DateTime.strftime(
                "%Y:%m:%d %H:%M:%S")
            # subject.metadata['NewName'] = subject.metadata['NewName']+";"+ infile.NewName
            lastDateTime = currentDateTime
            si = si + 1
            bProc = True
        #  print(subject.metadata)

        # sequence has ended, current image is in a new sequence
        elif 0 < si < maxSeq and ((currentDateTime - lastDateTime) > maxTime or subject.metadata[
                '#machine_prediction'] != infile.ZooniverseCode):
            si = maxSeq
            bProc = False

        # need to start a new sequence OR maxSeq = 1
        if si == maxSeq:
            # print("Upload")
            subject.save()
            subject_set.add(subject)
            si = 0

        # first image within a sequence
        if si == 0 and bProc is False:
            subject = Subject()  # create new subject
            subject.links.project = project
            lastDateTime = currentDateTime
            outfile = copy_image(infile, outdir, imageheight)
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
