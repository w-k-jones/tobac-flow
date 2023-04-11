from tobac_flow.linking import File_Linker
import argparse

parser = argparse.ArgumentParser(
    description="Combine multiple files of detected DCCs in GOES-16 ABI data"
)
parser.add_argument("-sd", help="Directory to save output files", default=None)
parser.add_argument("--file_suffix", help="Suffix to save files under", default="")
parser.add_argument("files", help="List of files to combine", nargs="+", type=str)

args = parser.parse_args()
files = args.files
output_path = args.sd


def output_func(ds):
    pass


linker = File_Linker(
    files, output_func, output_path=output_path, output_file_suffix=args.file_suffix
)
linker.process_files()
