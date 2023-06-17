from tobac_flow.linking import Label_Linker
import argparse

parser = argparse.ArgumentParser(
    description="Combine multiple files of detected DCCs in GOES-16 ABI data"
)
parser.add_argument(
    "-sd", help="Directory to save output files", default="../data/linked"
)
parser.add_argument("--file_suffix", help="Suffix to save files under", default="")
parser.add_argument("files", help="List of files to combine", nargs="+", type=str)

args = parser.parse_args()
files = args.files
output_path = args.sd

linker = Label_Linker(files)
linker.run_all()
