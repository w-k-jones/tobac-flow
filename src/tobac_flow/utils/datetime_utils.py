import pathlib
import numpy as np
import xarray as xr
from datetime import datetime
from dateutil.parser import parse as parse_date


def get_dates_from_filename(filename: str | pathlib.Path) -> tuple[datetime, datetime]:
    if isinstance(filename, str):
        start_date = parse_date(
            filename.split("/")[-1].split("_S")[-1][:15], fuzzy=True
        )
        end_date = parse_date(filename.split("/")[-1].split("_E")[-1][:15], fuzzy=True)
    elif isinstance(filename, pathlib.Path):
        start_date = parse_date(filename.name.split("_S")[-1][:15], fuzzy=True)
        end_date = parse_date(filename.name.split("_E")[-1][:15], fuzzy=True)
    else:
        raise ValueError("filename parameter must be either a string or a Path object")

    return start_date, end_date


def trim_file_start(dataset: xr.Dataset, filename: str | pathlib.Path) -> xr.Dataset:
    return dataset.sel(t=slice(get_dates_from_filename(filename)[0], None))


def trim_file_end(dataset: xr.Dataset, filename: str | pathlib.Path) -> xr.Dataset:
    return dataset.sel(t=slice(None, get_dates_from_filename(filename)[1]))


def get_datetime_from_coord(coord: xr.DataArray) -> list[datetime]:
    return [parse_date(t.item()) for t in coord.astype("datetime64[s]").astype(str)]


def time_diff(datetime_list: list[datetime]) -> list[float]:
    return (
        [(datetime_list[1] - datetime_list[0]).total_seconds() / 60]
        + [
            (datetime_list[i + 2] - datetime_list[i]).total_seconds() / 120
            for i in range(len(datetime_list) - 2)
        ]
        + [(datetime_list[-1] - datetime_list[-2]).total_seconds() / 60]
    )


def get_time_diff_from_coord(coord: xr.DataArray) -> list[float]:
    return np.array(time_diff(get_datetime_from_coord(coord)))
