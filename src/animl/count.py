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


# ==============================================================================
# EMERGENCE DETECTION
# ==============================================================================

def detect_emergence(counts: pd.DataFrame,
                     detections: pd.DataFrame,
                     target_class: str,
                     station_col: str = "station",
                     window_size: int = 20,
                     step_size: int = 5,
                     density_metric: str = "fraction",
                     density_threshold: float = 0.3,
                     sustained_windows: int = 3) -> pd.DataFrame:
    """
    Detect the first sustained emergence of a target class across image
    sequences, using a sliding window to distinguish real, sustained
    presence from isolated false-positive detections.

    Slides a window of `window_size` sequences across the counts
    DataFrame in steps of `step_size`. At each window position, computes
    a density metric (fraction of sequences with target_class present,
    or summed count) for that window. Emergence is declared at the first
    window where density crosses `density_threshold` and the threshold
    remains crossed for `sustained_windows` consecutive window positions.

    Detection is performed independently per station, since different
    camera stations/burrows have independent emergence timelines.

    Args:
        counts (pd.DataFrame): output of count_detections(), one row per
            sequence with a `sequence` column and one column per class
        detections (pd.DataFrame): output of parse_detections() /
            sequence_calculation(), used to look up timestamp and
            filename for the sequence where emergence is detected. Must
            contain `sequence`, `datetime`, `filename`, `filepath`,
            and `station_col` columns.
        target_class (str): the class column in `counts` to track
            (e.g. "owl_juvenile")
        station_col (str): column name in `detections` representing the
            station or camera. Used to look up station per sequence and
            to run detection independently per station.
        window_size (int): number of consecutive sequences per window
        step_size (int): number of sequences to slide the window forward
            at each step
        density_metric (str): "fraction" = fraction of sequences in the
            window with target_class > 0; "sum" = total target_class
            count summed across the window (weights multi-individual
            detections more heavily)
        density_threshold (float): density value that must be crossed to
            count as a candidate emergence window. For "fraction", this
            is a 0-1 proportion. For "sum", this is an absolute count and
            should be scaled to window_size.
        sustained_windows (int): number of consecutive window positions
            that must stay above density_threshold for emergence to be
            declared (guards against a single fluke window)

    Returns:
        pd.DataFrame: one row per station with columns:
            station, emerged (bool), emergence_sequence, emergence_datetime,
            emergence_filename, emergence_filepath, window_density,
            notes
    """
    if target_class not in counts.columns:
        raise ValueError(
            f"target_class '{target_class}' not found in counts columns: "
            f"{list(counts.columns)}"
        )

    # Map sequence -> station/datetime/filename via detections, since
    # counts has no station or timestamp info of its own
    seq_lookup = (
        detections
        .sort_values("datetime")
        .groupby("sequence")
        .agg(
            station=(station_col, "first"),
            datetime=("datetime", "first"),
            filename=("filename", "first"),
            filepath=("filepath", "first"),
        )
        .reset_index()
    )

    merged = counts.merge(seq_lookup, on="sequence", how="left")
    merged = merged.sort_values("sequence").reset_index(drop=True)

    results = []

    for station, station_group in merged.groupby("station"):
        station_group = station_group.sort_values("sequence").reset_index(drop=True)
        n = len(station_group)

        emerged = False
        emergence_idx = None
        window_density_at_emergence = None

        # Slide the window across this station's sequences
        window_starts = list(range(0, max(n - window_size + 1, 1), step_size))
        if not window_starts:
            window_starts = [0]

        densities = []
        for start in window_starts:
            end = min(start + window_size, n)
            window = station_group.iloc[start:end]

            if density_metric == "fraction":
                density = (window[target_class] > 0).mean()
            elif density_metric == "sum":
                density = window[target_class].sum()
            else:
                raise ValueError(
                    f"density_metric must be 'fraction' or 'sum', got '{density_metric}'"
                )

            densities.append({"start": start, "end": end, "density": density})

        # Find first run of `sustained_windows` consecutive windows above threshold
        for i in range(len(densities) - sustained_windows + 1):
            run = densities[i:i + sustained_windows]
            if all(d["density"] >= density_threshold for d in run):
                emerged = True
                emergence_idx = run[0]["start"]
                window_density_at_emergence = run[0]["density"]
                break

        if emerged:
            emergence_row = station_group.iloc[emergence_idx]
            results.append({
                "station": station,
                "emerged": True,
                "emergence_sequence": emergence_row["sequence"],
                "emergence_datetime": emergence_row["datetime"],
                "emergence_filename": emergence_row["filename"],
                "emergence_filepath": emergence_row["filepath"],
                "window_density": round(window_density_at_emergence, 3),
                "notes": (
                    f"First window of {window_size} sequences starting at "
                    f"sequence {emergence_row['sequence']} crossed density "
                    f"{density_threshold} ({density_metric}) and stayed "
                    f"crossed for {sustained_windows} consecutive windows."
                ),
            })
        else:
            max_density = max((d["density"] for d in densities), default=0)
            results.append({
                "station": station,
                "emerged": False,
                "emergence_sequence": None,
                "emergence_datetime": None,
                "emergence_filename": None,
                "emergence_filepath": None,
                "window_density": round(max_density, 3),
                "notes": (
                    f"No sustained emergence found. Max window density "
                    f"observed was {round(max_density, 3)}, below threshold "
                    f"of {density_threshold}."
                ),
            })

    return pd.DataFrame(results)


def emergence_density_curve(counts: pd.DataFrame,
                            detections: pd.DataFrame,
                            target_class: str,
                            station_col: str = "station",
                            window_size: int = 20,
                            step_size: int = 5,
                            density_metric: str = "fraction") -> pd.DataFrame:
    """
    Compute the full sliding-window density curve per station, without
    applying a threshold. Useful for plotting and visually choosing
    threshold/window parameters before committing to detect_emergence().

    Args:
        Same as detect_emergence(), minus density_threshold and
        sustained_windows.

    Returns:
        pd.DataFrame: one row per (station, window) with columns
            station, window_start_sequence, window_end_sequence,
            window_start_datetime, density
    """
    if target_class not in counts.columns:
        raise ValueError(
            f"target_class '{target_class}' not found in counts columns: "
            f"{list(counts.columns)}"
        )

    seq_lookup = (
        detections
        .sort_values("datetime")
        .groupby("sequence")
        .agg(
            station=(station_col, "first"),
            datetime=("datetime", "first"),
        )
        .reset_index()
    )

    merged = counts.merge(seq_lookup, on="sequence", how="left")
    merged = merged.sort_values("sequence").reset_index(drop=True)

    rows = []
    for station, station_group in merged.groupby("station"):
        station_group = station_group.sort_values("sequence").reset_index(drop=True)
        n = len(station_group)

        window_starts = list(range(0, max(n - window_size + 1, 1), step_size))
        if not window_starts:
            window_starts = [0]

        for start in window_starts:
            end = min(start + window_size, n)
            window = station_group.iloc[start:end]

            if density_metric == "fraction":
                density = (window[target_class] > 0).mean()
            elif density_metric == "sum":
                density = window[target_class].sum()
            else:
                raise ValueError(
                    f"density_metric must be 'fraction' or 'sum', got '{density_metric}'"
                )

            rows.append({
                "station": station,
                "window_start_sequence": window.iloc[0]["sequence"],
                "window_end_sequence": window.iloc[-1]["sequence"],
                "window_start_datetime": window.iloc[0]["datetime"],
                "density": density,
            })

    return pd.DataFrame(rows)
