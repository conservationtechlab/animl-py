import argparse
import sys
import pandas as pd

from animl.export import export_megadetector


def convert_animl_to_md():
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

    manifest = pd.read_csv(args.input_file)

    export_megadetector(manifest, args.output_file)


if __name__ == '__main__':
    convert_animl_to_md()
