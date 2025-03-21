import argparse
import pathlib
from datetime import datetime

import numpy as np
import xarray as xr

from tobac_flow.analysis import get_label_stats, weighted_statistics_on_labels
from tobac_flow.dataset import calculate_label_properties
from tobac_flow.linking import process_file
from tobac_flow.utils.datetime_utils import get_dates_from_filename
from tobac_flow.utils.xarray_utils import add_compression_encoding, add_dataarray_to_ds

parser = argparse.ArgumentParser(
    description="""Postprocess detected DCCs using GOES-16 data"""
)
parser.add_argument("file", help="File to validate", type=str)
parser.add_argument("links_file", help="Link file containing new labels for each file", type=str)

parser.add_argument("-sd", help="Directory to save preprocess files", default="")
parser.add_argument(
    "-sdf", help="Date formatting string for subdirectories", default=""
)
parser.add_argument(
    "--save_spatial_props",
    help="Save statistics of label spatial properties to output file",
    action="store_true",
)
if __name__ == "__main__":
    args = parser.parse_args()
    
    filename = pathlib.Path(args.file)
    assert filename.exists(), f'File {filename} not found'

    start_date, end_date = get_dates_from_filename(filename)
    
    save_path = pathlib.Path(args.sd)
    if args.sdf:
        save_path = save_path / start_date.strftime(args.sdf)
    
    save_path.mkdir(parents=True, exist_ok=True)

    save_path = save_path / filename.name
    
    with xr.open_dataset(args.links_file) as links_ds:
        dataset = process_file(args.file, links_ds)
    
    print(datetime.now(), "Calculating label properties", flush=True)
    calculate_label_properties(dataset)

    if args.save_spatial_props:
        print(datetime.now(), "Calculating spatial properties", flush=True)
        get_label_stats(dataset.core_label, dataset)
        get_label_stats(dataset.thick_anvil_label, dataset)
        get_label_stats(dataset.thin_anvil_label, dataset)

    weights = np.repeat(dataset.area.data[np.newaxis, ...], dataset.t.size, 0)

    print(datetime.now(), "Calculating statistics", flush=True)
    for field in (dataset.bt,):
        [
            add_dataarray_to_ds(da[dataset.core_step.data - 1], dataset)
            for da in weighted_statistics_on_labels(
                dataset.core_step_label,
                field.compute(),
                weights,
                name="core_step",
                dim="core_step",
                dtype=np.float32,
            )
        ]

        [
            add_dataarray_to_ds(da[dataset.thick_anvil_step.data - 1], dataset)
            for da in weighted_statistics_on_labels(
                dataset.thick_anvil_step_label,
                field.compute(),
                weights,
                name="thick_anvil_step",
                dim="thick_anvil_step",
                dtype=np.float32,
            )
        ]

        [
            add_dataarray_to_ds(da[dataset.thin_anvil_step.data - 1], dataset)
            for da in weighted_statistics_on_labels(
                dataset.thin_anvil_step_label,
                field.compute(),
                weights,
                name="thin_anvil_step",
                dim="thin_anvil_step",
                dtype=np.float32,
            )
        ]

    # Remove BT to reduce file size
    dataset = dataset.drop_vars("bt")

    # Add compression encoding
    print(datetime.now(), "Adding compression encoding", flush=True)
    dataset = add_compression_encoding(dataset, compression="zstd", complevel=5, shuffle=True)

    print(datetime.now(), "Saving to %s" % (save_path), flush=True)
    dataset.to_netcdf(save_path)

    dataset.close()
