import numpy as np
import xarray as xr
import argparse
import pathlib
from datetime import datetime
from dateutil.parser import parse as parse_date
from tobac_flow.analysis import get_label_stats, weighted_statistics_on_labels
from tobac_flow.dataset import calculate_label_properties
from tobac_flow.utils.xarray_utils import add_compression_encoding, add_dataarray_to_ds

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

if __name__ == "__main__":
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

    # Remove BT to reduce file size
    dataset = dataset.drop_vars("BT")

    # Add compression encoding
    print(datetime.now(), "Adding compression encoding", flush=True)
    dataset = add_compression_encoding(dataset, compression="zstd", complevel=5, shuffle=True)

    print(datetime.now(), "Saving to %s" % (save_path), flush=True)
    dataset.to_netcdf(save_path)

    dataset.close()
