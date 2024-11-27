import functools
from typing import Any, Callable, Optional
import xarray as xr

def handle_output(arg, output, name, drop_attrs, attributes):
    if not isinstance(output, xr.DataArray):
        output = arg.copy(data=output).drop_encoding()
        output.name = name

    for key in drop_attrs:
        if key in output.attrs:
            del output.attrs[key]
    
    for key, value in attributes.items():
        output.attrs[key] = value
    
    return output

def configure_dataarray(
    name: Optional[str] = None, 
    drop_attrs: list[str] = ["valid_range", "cell_methods", "units_metadata", "_FillValue", "missing_value"],
    **attributes, 
) -> Callable:
    def configure_dataarray_wrapper(func) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, name=name, drop_attrs=drop_attrs, attributes=attributes, **kwargs) -> Any:
            for arg in args:
                if isinstance(arg, xr.DataArray):
                    break
            else:
                for _, arg in kwargs.items():
                    if isinstance(arg, xr.DataArray):
                        break
                else:
                    return func(*args, **kwargs)
            
            output = func(*args, **kwargs)

            if type(output) == tuple:
                output = tuple(
                    handle_output(arg, out, name, drop_attrs, attributes)
                    for out in output
                )
            else:
                output = handle_output(arg, output, name, drop_attrs, attributes)

            return output
        return wrapper
    return configure_dataarray_wrapper