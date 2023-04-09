import numpy as np
import xarray as xr
import argparse
import pathlib
from datetime import datetime
from dateutil.parser import parse as parse_date
from tobac_flow.analysis import get_label_stats, weighted_statistics_on_labels
from tobac_flow.dataset import (
    flag_nan_adjacent_labels,
    add_dataarray_to_ds,
    calculate_label_properties,
)

parser = argparse.ArgumentParser(
    description="""Postprocess detected DCCs using GOES-16 data"""
)
parser.add_argument("file", help="File to validate", type=str)
parser.add_argument("-sd", help="Directory to save preprocess files", default=None)
parser.add_argument(
    "--save_spatial_props",
    help="Save statistics of label spatial properties to output file",
    action="store_true",
)
args = parser.parse_args()

fname = pathlib.Path(args.file)

if args.sd is None:
    save_dir = fname.parent
else:
    save_dir = pathlib.Path(args.sd)
if not save_dir.exists():
    save_dir.mkdir()

save_name = fname.stem + "_processed.nc"

save_path = save_dir / save_name

print("Saving to:", save_path)

dataset = xr.open_dataset(fname)

start_date = parse_date((str(fname)).split("_S")[-1].split("_E")[0], fuzzy=True)
end_date = parse_date((str(fname)).split("_E")[-1].split("_X")[0], fuzzy=True)

if "BT" in dataset.data_vars:
    flag_nan_adjacent_labels(dataset, dataset.BT)

calculate_label_properties(dataset)

if args.save_spatial_props:
    get_label_stats(dataset.core_label, dataset)
    get_label_stats(dataset.thick_anvil_label, dataset)
    get_label_stats(dataset.thin_anvil_label, dataset)

weights = np.repeat(dataset.area.data[np.newaxis, ...], dataset.t.size, 0)

for field in (dataset.BT,):

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

print(datetime.now(), "Saving to %s" % (save_path), flush=True)
# Add compression encoding
comp = dict(zlib=True, complevel=5, shuffle=True)
for var in dataset.data_vars:
    dataset[var].encoding.update(comp)

dataset.to_netcdf(save_path)

dataset.close()
