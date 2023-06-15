import os
import glob
from PIL import Image

DEFAULT_COLORS = [
    'AliceBlue', 'Red', 'RoyalBlue', 'Gold', 'Chartreuse', 'Aqua', 'Azure',
    'Beige', 'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
    'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson',
    'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'RosyBrown', 'Aquamarine', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def load_image(input_file):
    """
    from CameraTraps/visualization/visualization_utils.py
    """
    image = Image.open(input_file)
    if image.mode not in ('RGBA', 'RGB', 'L'):
        raise AttributeError('Input image {} uses unsupported mode {}'.format(input_file, image.mode))
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')
    image.load()
    return image


def is_image(s):
    """
    from CameraTraps/MegaDetector
    Check a file's extension against a hard-coded set of image file extensions    '
    """
    image_extensions = ['.jpg', '.jpeg', '.gif', '.png', '.mp4']
    ext = os.path.splitext(s)[1].lower()
    return ext in image_extensions


def crop_image(detections, image, confidence_threshold=0.15, expansion=0):
    """
    Crops detections above *confidence_threshold* from the PIL image *image*,
    returning a list of PIL images.

    *detections* should be a list of dictionaries with keys 'conf' and 'bbox';
    see bbox format description below.  Normalized, [x,y,w,h], upper-left-origin.

    *expansion* specifies a number of pixels to include on each side of the box.
    """

    ret_images = []

    for detection in detections:

        score = float(detection['conf'])

        if score >= confidence_threshold:

            x1, y1, w_box, h_box = detection['bbox']
            ymin,xmin,ymax,xmax = y1, x1, y1 + h_box, x1 + w_box

            # Convert to pixels so we can use the PIL crop() function
            im_width, im_height = image.size
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            if expansion > 0:
                left -= expansion
                right += expansion
                top -= expansion
                bottom += expansion

            # PIL's crop() does surprising things if you provide values outside of
            # the image, clip inputs
            left = max(left,0); right = max(right,0)
            top = max(top,0); bottom = max(bottom,0)

            left = min(left,im_width-1); right = min(right,im_width-1)
            top = min(top,im_height-1); bottom = min(bottom,im_height-1)

            ret_images.append(image.crop((left, top, right, bottom)))

        # ...if this detection is above threshold

    # ...for each detection

    return ret_images

    
def render_detection_bounding_boxes(detections, image,
                                    label_map={}, 
                                    classification_label_map=None, 
                                    confidence_threshold=0.15, thickness=4, expansion=0,
                                    classification_confidence_threshold=0.3,
                                    max_classifications=3,
                                    colormap=DEFAULT_COLORS,
                                    textalign=0):
    """
    Renders bounding boxes, label, and confidence on an image if confidence is above the threshold.

    Boxes are in the format that's output from the batch processing API.

    Renders classification labels if present.

    Args:

        detections: detections on the image 
        image: PIL.Image object

        label_map: optional, mapping the numerical label to a string name. The type of the numerical label
            (default string) needs to be consistent with the keys in label_map; no casting is carried out.
            If this is None, no labels are shown.

        classification_label_map: optional, mapping of the string class labels to the actual class names.
            The type of the numerical label (default string) needs to be consistent with the keys in
            label_map; no casting is carried out.  If this is None, no classification labels are shown.

        confidence_threshold: optional, threshold above which the bounding box is rendered.
        
        thickness: line thickness in pixels. Default value is 4.
        
        expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
        
        classification_confidence_threshold: confidence above which classification result is retained.
        
        max_classifications: maximum number of classification results retained for one image.

    image is modified in place.
    """

    display_boxes = []
    
    # list of lists, one list of strings for each bounding box (to accommodate multiple labels)
    display_strs = []  
    
    # for color selection
    classes = []  

    for detection in detections:

        score = detection['conf']
        
        # Always render objects with a confidence of "None", this is typically used
        # for ground truth data.        
        if score is None or score >= confidence_threshold:
            
            x1, y1, w_box, h_box = detection['bbox']
            display_boxes.append([y1, x1, y1 + h_box, x1 + w_box])
            clss = detection['category']
            
            # {} is the default, which means "show labels with no mapping", so don't use "if label_map" here
            # if label_map:
            if label_map is not None:
                label = label_map[clss] if clss in label_map else clss
                if score is not None:
                    displayed_label = ['{}: {}%'.format(label, round(100 * score))]
                else:
                    displayed_label = ['{}'.format(label)]
            else:
                displayed_label = ''

            if 'classifications' in detection:

                # To avoid duplicate colors with detection-only visualization, offset
                # the classification class index by the number of detection classes
                clss = int(detection['classifications'][0][0]) + 3
                classifications = detection['classifications']
                if len(classifications) > max_classifications:
                    classifications = classifications[0:max_classifications]
                    
                for classification in classifications:
                    
                    classification_conf = classification[1]
                    if classification_conf is not None and \
                        classification_conf < classification_confidence_threshold:
                        continue
                    class_key = classification[0]
                    if (classification_label_map is not None) and (class_key in classification_label_map):
                        class_name = classification_label_map[class_key]
                    else:
                        class_name = class_key
                    if classification_conf is not None:
                        displayed_label += ['{}: {:5.1%}'.format(class_name.lower(), classification_conf)]
                    else:
                        displayed_label += ['{}'.format(class_name.lower())]
                    
                # ...for each classification

            # ...if we have classification results
                        
            display_strs.append(displayed_label)
            classes.append(clss)

        # ...if the confidence of this detection is above threshold

    # ...for each detection
    
    display_boxes = np.array(display_boxes)

    draw_bounding_boxes_on_image(image, display_boxes, classes,
                                 display_strs=display_strs, thickness=thickness, 
                                 expansion=expansion, colormap=colormap, textalign=textalign)