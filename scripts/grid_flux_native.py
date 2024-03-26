#!/home/users/wkjones/miniconda3/envs/tobac_flow/bin/python
import numpy as np
import pandas as pd
import xarray as xr
import argparse
import pathlib
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from scipy.stats import binned_statistic_2d
from tobac_flow.postprocess import add_cre_to_dataset
from tobac_flow.utils import add_area_to_dataset

parser = argparse.ArgumentParser(description="""Grid SEVIRI fluxes to a fixed grid""")
parser.add_argument("date", help="Date on which to start process", type=str)
parser.add_argument("-sd", help="Directory to save gridded flux files", default=None)

args = parser.parse_args()

start_date = parse_date(args.date, fuzzy=True)
end_date = start_date + timedelta(hours=24)

if args.sd is None:
    save_dir = pathlib.Path("./")
else:
    save_dir = pathlib.Path(args.sd)
if not save_dir.exists():
    save_dir.mkdir()

save_name = "flux_regrid_SEVIRI_S%s.nc" % (start_date.strftime("%Y%m%d_%H0000"))

save_path = save_dir / save_name

print("Saving to:", save_path)

lon_bins = np.arange(-180, 181)
lat_bins = np.arange(-90, 91)

lons = lon_bins[1:] - 0.5
lats = lat_bins[1:] - 0.5


print(datetime.now(), "Loading flux properties", flush=True)

flx_files = sorted(
    sum(
        [
            list(
                (
                    pathlib.Path("/gws/nopw/j04/eo_shared_data_vol1/satellite/seviri-orac/Data/")
                    / date.strftime("%Y")
                    / date.strftime("%m")
                    / date.strftime("%d")
                    / date.strftime("%H")
                ).glob(
                    f"{date.strftime('%Y%m%d%H')}[0-9][0-9]00-ESACCI-TOA-SEVIRI-MSG[1-4]-fv3.0.nc"
                )
            )
            for date in pd.date_range(start_date, end_date, freq="h", inclusive="left")
        ],
        [],
    )
)

flx_ds = xr.open_mfdataset(
    flx_files, combine="nested", concat_dim="t", decode_times=False
)

# Add time coord to flx_ds
flx_dates = [
    datetime.strptime(f.name[:14], "%Y%m%d%H%M%S")
    + timedelta(minutes=12, seconds=42)
    for f in flx_files
]
flx_ds.coords["t"] = flx_dates

print(datetime.now(), "Processing flux properties", flush=True)
flx_ds = add_cre_to_dataset(flx_ds)

grid_ds = xr.Dataset(coords={"lat": lats, "lon": lons})

grid_ds["n_times"] = flx_ds.t.size


def weighted_binned_mean_2d(x, y, data, weights, bins=None):
    wh = np.isfinite(data)
    binned_data = binned_statistic_2d(
        x[wh], y[wh], data[wh] * weights[wh], bins=bins, statistic="sum"
    )[0]
    binned_data /= binned_statistic_2d(
        x[wh], y[wh], weights[wh], bins=bins, statistic="sum"
    )[0]
    return binned_data


for var in (
    "toa_swup",
    "toa_swup_clr",
    "toa_swup_cre",
    "toa_lwup",
    "toa_lwup_clr",
    "toa_lwup_cre",
    "toa_net",
    "toa_net_cre",
    "boa_swdn",
    "boa_swdn_clr",
    "boa_swdn_cre",
    "boa_swup",
    "boa_swup_clr",
    "boa_swup_cre",
    "boa_lwdn",
    "boa_lwdn_clr",
    "boa_lwdn_cre",
    "boa_lwup",
    "boa_lwup_clr",
    "boa_lwup_cre",
    "boa_net",
    "boa_net_cre",
):
    print(f"Processing {var}", flush=True)
    grid_values = weighted_binned_mean_2d(
        flx_ds.lat.to_numpy(),
        flx_ds.lon.to_numpy(),
        flx_ds[var].to_numpy(),
        flx_ds.area.to_numpy(),
        bins=(lat_bins, lon_bins),
    )
    grid_ds[var] = xr.DataArray(
        grid_values,
        coords={"lat": lats, "lon": lons},
        dims=("lat", "lon"),
    )

print(datetime.now(), "Saving to %s" % (save_path), flush=True)
# Add compression encoding
comp = dict(zlib=True, complevel=5, shuffle=True)
for var in grid_ds.data_vars:
    grid_ds[var].encoding.update(comp)

grid_ds.to_netcdf(save_path)

print(datetime.now(), "Saving complete, closing datasets", flush=True)

grid_ds.close()
flx_ds.close()
