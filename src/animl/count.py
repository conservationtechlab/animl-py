"""
Count Module

Functions for counting animal detections across image sequences.

@ Nikita Sharma 2026
"""
import pandas as pd
from animl import file_management
from animl.utils.general import get_iou


# ==============================================================================
# DEDUPLICATION
# ==============================================================================

def deduplicate(image_detections: pd.DataFrame,
                iou_threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove duplicate detections within a single image using IOU.
    When two detections overlap above the threshold, keep the one
    with higher confidence.

    Args:
        image_detections (pd.DataFrame): detections for a single image
        iou_threshold (float): IOU threshold above which two detections
                               are considered the same animal

    Returns:
        pd.DataFrame: deduplicated detections
    """
    kept = []

    for _, row in image_detections.iterrows():
        bbox_a = [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]]
        is_duplicate = False

        for i, kept_row in enumerate(kept):
            bbox_b = [kept_row["bbox_x"], kept_row["bbox_y"],
                      kept_row["bbox_w"], kept_row["bbox_h"]]
            iou = get_iou(bbox_a, bbox_b)

            if iou >= iou_threshold:
                is_duplicate = True
                if row["conf"] > kept_row["conf"]:
                    kept[i] = row
                break

        if not is_duplicate:
            kept.append(row)

    return pd.DataFrame(kept)


# ==============================================================================
# COUNTING
# ==============================================================================

def count_detections(detections: pd.DataFrame,
                     station_col: str = "station",
                     confidence_threshold: float = 0.5,
                     iou_threshold: float = 0.9,
                     maxdiff: int = 60,
                     max_n: int = None,
                     classes: list = None) -> pd.DataFrame:
    """
    Count detections per species across image sequences.

    Args:
        detections (pd.DataFrame): output of parse_detections()
        station_col (str): column name representing the station or camera
        confidence_threshold (float): minimum confidence to consider a detection
        iou_threshold (float): IOU threshold for duplicate detection removal
        maxdiff (int): max time difference in seconds between images in a sequence
        max_n (int): optional max number of images per sequence to consider
        classes (list): list of category labels to count

    Returns:
        pd.DataFrame: one row per sequence with averaged counts per species
    """

    # Step 1: assign sequence IDs using station and datetime already in detections
    detections = file_management.sequence_calculation(
        detections, station_col=station_col, maxdiff=maxdiff
    )

    # Step 2: get unique classes from detections if not specified
    if classes is None:
        classes = detections['category_label'].dropna().unique().tolist()
        classes = [c for c in classes if c != 'empty']
    
    # Step 3: filter out null and low confidence detections
    detections = detections[detections["conf"].notna()]
    detections = detections[detections["conf"] >= confidence_threshold]

    # Step 4: filter to target classes only
    detections = detections[detections["category_label"].isin(classes)]

    # Step 5: process each sequence
    results = []

    for seq_id, seq_group in detections.groupby("sequence"):

        image_counts = []

        for i, (filepath, image_group) in enumerate(seq_group.groupby("filepath")):

            # optionally cap number of images per sequence
            if max_n is not None and i >= max_n:
                break

            # deduplicate detections within this image
            kept = deduplicate(image_group, iou_threshold)

            # count per class
            counts = {cls: 0 for cls in classes}
            for _, det in kept.iterrows():
                label = det["category_label"]
                if label in counts:
                    counts[label] += 1

            image_counts.append(counts)

        # Step 6: average counts across images in the sequence
        if image_counts:
            avg_counts = {cls: sum(c[cls] for c in image_counts) / len(image_counts)
                          for cls in classes}
            avg_counts["sequence"] = seq_id
            results.append(avg_counts)

    return pd.DataFrame(results)