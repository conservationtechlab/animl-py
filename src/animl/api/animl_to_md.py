'''
#
# animl_results_to_md_results.py
#
# Convert a .csv file produced by the Animl package:
#
# https://github.com/conservationtechlab/animl-py
#
# ...to a MD results file suitable for import into Timelapse.
#
# Columns are expected to be:
#
# file
# category (MD category identifies: 0==empty, 1==animal, 2==person, 3==vehicle)
# detection_conf
# bbox1,bbox2,bbox3,bbox4
# class
# classification_conf
#
# Credit goes to Dan Morris https://github.com/agentmorris/MegaDetector/tree/main
# Adding a modified script to animl-py repo
'''
import pandas as pd
import json
import sys
import argparse


detection_category_id_to_name = {'0': 'empty', '1': 'animal', '2': 'person', '3': 'vehicle'}


def animl_results_to_md_results(input_file, output_file=None):
    """
    Converts the .csv file [input_file] to the MD-formatted .json file [output_file].

    If [output_file] is None, '.json' will be appended to the input file.
    """
    if output_file is None:
        output_file = input_file + '.json'

    df = pd.read_csv(input_file)

    expected_columns = ('file', 'category', 'detection_conf',
                        'bbox1', 'bbox2', 'bbox3', 'bbox4',
                        'class', 'classification_conf')

    for s in expected_columns:
        assert s in df.columns, \
            'Expected column {} not found'.format(s)

    classification_category_name_to_id = {}
    filename_to_results = {}

    for i_row, row in df.iterrows():

        if str(row['category']) == '0':
            continue

        # Is this the first detection we've seen for this file?
        if row['file'] in filename_to_results:
            im = filename_to_results[row['file']]
        else:
            im = {}
            im['detections'] = []
            im['file'] = row['file']
            filename_to_results[im['file']] = im

        assert isinstance(row['category'], int), 'Invalid category identifier in row {}'.format(im['file'])
        detection_category_id = str(row['category'])
        assert detection_category_id in detection_category_id_to_name, \
            'Unrecognized detection category ID {}'.format(detection_category_id)

        detection = {}
        detection['category'] = detection_category_id
        detection['conf'] = row['detection_conf']
        bbox = [row['bbox1'], row['bbox2'], row['bbox3'], row['bbox4']]
        detection['bbox'] = bbox
        classification_category_name = row['class']

        # Have we seen this classification category before?
        if classification_category_name in classification_category_name_to_id:
            classification_category_id = \
                classification_category_name_to_id[classification_category_name]
        else:
            classification_category_id = str(len(classification_category_name_to_id))
            classification_category_name_to_id[classification_category_name] = \
                classification_category_id

        classifications = [[classification_category_id, row['classification_conf']]]
        detection['classifications'] = classifications

        im['detections'].append(detection)

    # ...for each row

    info = {}
    info['format_version'] = '1.3'
    info['detector'] = 'Animl'
    info['classifier'] = 'Animl'

    results = {}
    results['info'] = info
    results['detection_categories'] = detection_category_id_to_name
    results['classification_categories'] = \
        {v: k for k, v in classification_category_name_to_id.items()}
    results['images'] = list(filename_to_results.values())

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=1)


def main():

    parser = argparse.ArgumentParser(
        description='Convert an Animl-formatted .csv results file to MD-formatted .json results file')

    parser.add_argument(
        'input_file',
        type=str,
        help='input .csv file')

    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='output .json file (defaults to input file appened with ".json")')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    animl_results_to_md_results(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
