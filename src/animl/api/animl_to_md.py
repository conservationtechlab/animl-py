"""
Export Animl results to MD json format

"""
import argparse
import os
import sys
import pandas as pd

from animl.export import export_megadetector


def convert_animl_to_md(input_file, output_file=None):
    """Convert an Animl-formatted .csv results file to MD-formatted .json results file.

    Args:
        input_file: Path to the input Animl .csv file.
        output_file: Path to the output MD .json file. If None, defaults to input file with '.json' extension.

    Returns:
        None
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.json'

    manifest = pd.read_csv(input_file)

    export_megadetector(manifest, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert an Animl-formatted .csv results file to MD-formatted .json results file')

    parser.add_argument('input_file', type=str, help='input .csv file')

    parser.add_argument('--output_file',
                        type=str,
                        default=None,
                        help='output .json file (defaults to input file appened with ".json")')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    convert_animl_to_md(args.input_file, args.output_file)
