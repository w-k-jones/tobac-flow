import pathlib
from dateutil.parser import parse as parse_date

def get_dates_from_filename(filename):
    if isinstance(filename, str):
        start_date = parse_date(filename.split("/")[-1].split("_S")[-1][:15], fuzzy=True)
        end_date = parse_date(filename.split("/")[-1].split("_E")[-1][:15], fuzzy=True)
    elif isinstance(filename, pathlib.Path):
        start_date = parse_date(filename.name.split("_S")[-1][:15], fuzzy=True)
        end_date = parse_date(filename.name.split("_E")[-1][:15], fuzzy=True)
    else:
        raise ValueError("filename parameter must be either a string or a Path object")
    
    return start_date, end_date

def trim_file_start(dataset, filename):
    return dataset.sel(t=slice(get_dates_from_filename(filename)[0], None))

def trim_file_end(dataset, filename):
    return dataset.sel(t=slice(None, get_dates_from_filename(filename)[1]))
    