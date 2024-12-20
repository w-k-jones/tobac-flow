import argparse
import pathlib
from datetime import datetime

import xarray as xr

from tobac_flow.linking import process_file
from tobac_flow.utils.datetime_utils import get_dates_from_filename
from tobac_flow.utils.xarray_utils import add_compression_encoding

parser = argparse.ArgumentParser(
    description="Combine multiple files of detected DCCs in GOES-16 ABI data"
)
parser.add_argument("file", help="List of files to combine", type=str)
parser.add_argument("links_file", help="Link file containing new labels for each file", type=str)
parser.add_argument(
    "-sd", help="Directory to save output files", default="../data/linked"
)
parser.add_argument(
    "-sdf", help="Date formatting string for subdirectories", default=""
)

if __name__ == "__main__":
    args = parser.parse_args()
    
    filename = pathlib.Path(args.file)
    assert filename.exists(), f'File {filename} not found'

    file_date = get_dates_from_filename(filename)[0]
    
    save_path = pathlib.Path(args.sd)
    if parser.sdf:
        save_path = save_path / file_date.strftime(args.sdf)
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    with xr.open_dataset(args.links_file) as links_ds:
        save_ds = process_file(args.file, links_ds)
        print(datetime.now(), "Adding compression encoding", flush=True)
        save_ds = add_compression_encoding(save_ds, compression="zstd", complevel=5, shuffle=True)

        new_filename = save_path / filename.name
        print(datetime.now(), "Saving to %s" % (new_filename), flush=True)
        save_ds.to_netcdf(new_filename)
        save_ds.close()