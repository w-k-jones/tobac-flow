from datetime import datetime, timedelta
import pathlib
from typing import Callable
import numpy as np
import xarray as xr

from tobac_flow.dataset import (
    flag_edge_labels,
    flag_nan_adjacent_labels,
    add_step_labels,
    add_label_coords,
    link_step_labels,
)
from tobac_flow.utils.datetime_utils import get_dates_from_filename
from tobac_flow.utils.label_utils import find_overlapping_labels


# Functions to link overlapping labels
def recursive_linker(
    links_list1: list | None = None,
    links_list2: list | None = None,
    label_list1: list | None = None,
    label_list2: list | None = None,
    overlap_list1: list | None = None,
    overlap_list2: list | None = None,
) -> tuple[list, list]:
    recursive = False
    if links_list1 is None:
        links_list1 = []
    if links_list2 is None:
        links_list2 = []
    if label_list1 is None:
        label_list1 = []
    if label_list2 is None:
        label_list2 = []
    if overlap_list1 is None:
        overlap_list1 = []
    if overlap_list2 is None:
        overlap_list2 = []
    for i in links_list1:
        if i in label_list1:
            loc = label_list1.index(i)
            label = label_list1.pop(loc)
            overlap = overlap_list1.pop(loc)
            for j in overlap:
                if j not in links_list2:
                    links_list2.append(j)
                    recursive = True
    if recursive:
        links_list2, links_list1 = recursive_linker(
            links_list1=links_list2,
            links_list2=links_list1,
            label_list1=label_list2,
            label_list2=label_list1,
            overlap_list1=overlap_list2,
            overlap_list2=overlap_list1,
        )
    return links_list1, links_list2


def link_labels(
    labels1: np.ndarray[int],
    labels2: np.ndarray[int],
    overlap: float = 0,
    absolute_overlap: int = 0,
) -> tuple[list, list]:
    label_list1 = np.unique(labels1[labels1 != 0]).tolist()
    label_list2 = np.unique(labels2[labels2 != 0]).tolist()

    bins1 = np.cumsum(np.bincount(labels1.ravel()))
    args1 = np.argsort(labels1.ravel())

    bins2 = np.cumsum(np.bincount(labels2.ravel()))
    args2 = np.argsort(labels2.ravel())

    overlap_list1 = [
        find_overlapping_labels(
            labels2,
            label,
            args1,
            bins1,
            overlap=overlap,
            absolute_overlap=absolute_overlap,
        )
        for label in label_list1
    ]

    overlap_list2 = [
        find_overlapping_labels(
            labels1,
            label,
            args2,
            bins2,
            overlap=overlap,
            absolute_overlap=absolute_overlap,
        )
        for label in label_list2
    ]

    links_list1 = []
    links_list2 = []
    while len(label_list1) > 0:
        temp_links1, temp_links2 = recursive_linker(
            [label_list1[0]],
            label_list1=label_list1,
            label_list2=label_list2,
            overlap_list1=overlap_list1,
            overlap_list2=overlap_list2,
        )
        links_list1.append(temp_links1)
        links_list2.append(temp_links2)
    return links_list1, links_list2


# Link overlapping cores
def link_dcc_cores(
    dcc_ds1: xr.Dataset,
    dcc_ds2: xr.Dataset,
    overlap: float = 0,
    absolute_overlap: int = 0,
) -> tuple[list, list, list, list]:
    t_overlap_list = sorted(list(set(dcc_ds1.t.data) & set(dcc_ds2.t.data)))[1:-1]

    core_step_links1, core_step_links2 = link_labels(
        dcc_ds1.core_step_label.sel(t=t_overlap_list).data,
        dcc_ds2.core_step_label.sel(t=t_overlap_list).data,
        overlap=overlap,
        absolute_overlap=absolute_overlap,
    )

    cores_list1 = [
        [dcc_ds1.core_step_core_index.sel(core_step=j).item() for j in i]
        for i in core_step_links1
    ]
    cores_list2 = [
        [dcc_ds2.core_step_core_index.sel(core_step=j).item() for j in i]
        for i in core_step_links2
    ]

    label_list1, label_list2 = (
        np.unique(sum(cores_list1, [])).tolist(),
        np.unique(sum(cores_list2, [])).tolist(),
    )

    overlap_list1 = [
        list(
            set(
                sum(
                    [
                        cores_list2[index]
                        for index, j in enumerate(cores_list1)
                        if i in j
                    ],
                    [],
                )
            )
        )
        for i in np.unique(sum(cores_list1, []))
    ]
    overlap_list2 = [
        list(
            set(
                sum(
                    [
                        cores_list1[index]
                        for index, j in enumerate(cores_list2)
                        if i in j
                    ],
                    [],
                )
            )
        )
        for i in np.unique(sum(cores_list2, []))
    ]

    core_links1, core_links2 = [], []

    while len(label_list1) > 0:
        temp_links1, temp_links2 = recursive_linker(
            [label_list1[0]],
            label_list1=label_list1,
            label_list2=label_list2,
            overlap_list1=overlap_list1,
            overlap_list2=overlap_list2,
        )
        core_links1.append(temp_links1)
        core_links2.append(temp_links2)

    return core_step_links1, core_step_links2, core_links1, core_links2


# Link overlapping anvils
def link_dcc_anvils(
    dcc_ds1: xr.Dataset,
    dcc_ds2: xr.Dataset,
    overlap: float = 0,
    absolute_overlap: int = 0,
) -> tuple[list, list, list, list]:
    t_overlap_list = sorted(list(set(dcc_ds1.t.data) & set(dcc_ds2.t.data)))[1:-1]

    anvil_step_links1, anvil_step_links2 = link_labels(
        dcc_ds1.thick_anvil_step_label.sel(t=t_overlap_list).data,
        dcc_ds2.thick_anvil_step_label.sel(t=t_overlap_list).data,
        overlap=overlap,
        absolute_overlap=absolute_overlap,
    )

    anvils_list1 = [
        [dcc_ds1.thick_anvil_step_anvil_index.sel(thick_anvil_step=j).item() for j in i]
        for i in anvil_step_links1
    ]
    anvils_list2 = [
        [dcc_ds2.thick_anvil_step_anvil_index.sel(thick_anvil_step=j).item() for j in i]
        for i in anvil_step_links2
    ]

    label_list1, label_list2 = (
        np.unique(sum(anvils_list1, [])).tolist(),
        np.unique(sum(anvils_list2, [])).tolist(),
    )

    overlap_list1 = [
        list(
            set(
                sum(
                    [
                        anvils_list2[index]
                        for index, j in enumerate(anvils_list1)
                        if i in j
                    ],
                    [],
                )
            )
        )
        for i in label_list1
    ]
    overlap_list2 = [
        list(
            set(
                sum(
                    [
                        anvils_list1[index]
                        for index, j in enumerate(anvils_list2)
                        if i in j
                    ],
                    [],
                )
            )
        )
        for i in label_list2
    ]

    anvil_links1, anvil_links2 = [], []

    while len(label_list1) > 0:
        temp_links1, temp_links2 = recursive_linker(
            [label_list1[0]],
            label_list1=label_list1,
            label_list2=label_list2,
            overlap_list1=overlap_list1,
            overlap_list2=overlap_list2,
        )
        anvil_links1.append(temp_links1)
        anvil_links2.append(temp_links2)

    return anvil_step_links1, anvil_step_links2, anvil_links1, anvil_links2


class File_Linker:
    def __init__(
        self,
        files: list[pathlib.Path | str],
        output_func: Callable,
        output_path: str | pathlib.Path | None = None,
        output_file_suffix: str | None = None,
        overlap: float = 0.5,
    ) -> None:
        self.files = [pathlib.Path(filename) for filename in files]
        for filename in self.files:
            if not filename.exists:
                raise ValueError(f"File {filename} does not exist")

        self.output_func = output_func
        self.output_path = output_path

        if isinstance(self.output_path, str):
            self.output_path = pathlib.Path(self.output_path)

        if self.output_path is not None and not self.output_path.exists():
            self.output_path.mkdir()

        if output_file_suffix is None or output_file_suffix == "":
            self.file_suffix = "_linked"
        else:
            self.file_suffix = output_file_suffix

        if len(self.file_suffix) > 0 and self.file_suffix[0] != "_":
            self.file_suffix = "_" + self.file_suffix

        self.overlap = overlap

        self.current_max_core_label = 0
        self.current_max_anvil_label = 0
        self.current_max_core_step_label = 0
        self.current_max_thick_anvil_step_label = 0
        self.current_max_thin_anvil_step_label = 0

        self.current_filename = self.files.pop(0)
        self.current_ds = xr.open_dataset(self.current_filename)

    def process_next_file(self) -> None:
        self.next_filename = self.files.pop(0)
        self.start_date, self.end_date = get_dates_from_filename(self.current_filename)
        self.next_ds = xr.open_dataset(self.next_filename)
        self.relabel_next_ds()

        # Check if there is some overlap
        self.t_overlap = sorted(
            list(set(self.current_ds.t.data) & set(self.next_ds.t.data))
        )
        if len(self.t_overlap) > 2:
            self.relabel_cores()
            self.relabel_anvils()
        else:
            self.current_max_core_label = np.maximum(
                np.max(
                    self.current_ds.core_label.sel(
                        t=slice(None, self.end_date - timedelta(seconds=1))
                    ).data
                ),
                self.current_max_core_label,
            )
            self.current_max_anvil_label = np.maximum(
                np.maximum(
                    np.max(
                        self.current_ds.thick_anvil_label.sel(
                            t=slice(None, self.end_date - timedelta(seconds=1))
                        ).data
                    ),
                    np.max(
                        self.current_ds.thin_anvil_label.sel(
                            t=slice(None, self.end_date - timedelta(seconds=1))
                        ).data
                    ),
                ),
                self.current_max_anvil_label,
            )

        self.output_current_ds()
        self.current_ds = self.next_ds
        self.current_filename = self.next_filename

    def process_files(self) -> None:
        while len(self.files) > 0:
            self.process_next_file()
        self.start_date, self.end_date = get_dates_from_filename(self.current_filename)
        self.output_current_ds()

    def output_current_ds(self) -> None:
        default_vars = [
            "goes_imager_projection",
            "lat",
            "lon",
            "area",
            "BT",
            "core_label",
            "thick_anvil_label",
            "thin_anvil_label",
        ]
        data_vars = [var for var in default_vars if var in self.current_ds.data_vars]
        self.current_ds = self.current_ds.get(data_vars)
        cores = np.unique(self.current_ds.core_label.data).astype(np.int32)
        if cores[0] == 0 and cores.size > 1:
            cores = cores[1:]
        anvils = np.asarray(
            list(
                set(np.unique(self.current_ds.thick_anvil_label.data))
                | set(np.unique(self.current_ds.thin_anvil_label.data))
            )
        ).astype(np.int32)
        if anvils[0] == 0 and anvils.size > 1:
            anvils = anvils[1:]
        self.current_ds = self.current_ds.assign_coords(
            {
                "core": cores,
                "anvil": anvils,
            }
        )
        # Add error flags
        flag_edge_labels(self.current_ds, self.start_date, self.end_date)
        if "BT" in self.current_ds.data_vars:
            flag_nan_adjacent_labels(self.current_ds, self.current_ds.BT)
        # Select only between current start and end date
        self.current_ds = self.current_ds.sel(
            t=slice(self.start_date, self.end_date - timedelta(seconds=1))
        )

        cores = np.unique(self.current_ds.core_label.data).astype(np.int32)
        if cores[0] == 0 and cores.size > 1:
            cores = cores[1:]
        anvils = np.unique(self.current_ds.thick_anvil_label.data).astype(np.int32)
        if anvils[0] == 0 and anvils.size > 1:
            anvils = anvils[1:]

        self.current_ds = self.current_ds.sel({"core": cores, "anvil": anvils})

        # Add step labels and coords
        add_step_labels(self.current_ds)
        # Increase step label values according the previous max
        self.current_ds.core_step_label.data[
            self.current_ds.core_step_label.data != 0
        ] += self.current_max_core_step_label
        self.current_ds.thick_anvil_step_label.data[
            self.current_ds.thick_anvil_step_label.data != 0
        ] += self.current_max_thick_anvil_step_label
        self.current_ds.thin_anvil_step_label.data[
            self.current_ds.thin_anvil_step_label.data != 0
        ] += self.current_max_thin_anvil_step_label

        self.current_ds = add_label_coords(self.current_ds)

        self.current_max_core_step_label = np.max(self.current_ds.core_step.data)
        self.current_max_thick_anvil_step_label = np.max(
            self.current_ds.thick_anvil_step.data
        )
        self.current_max_thin_anvil_step_label = np.max(
            self.current_ds.thin_anvil_step.data
        )

        link_step_labels(self.current_ds)

        # do something else...
        self.output_func(self.current_ds)

        if self.output_path is None:
            new_filename = self.current_filename.parent / (
                self.current_filename.stem + self.file_suffix + ".nc"
            )
        else:
            new_filename = self.output_path / (
                self.current_filename.stem + self.file_suffix + ".nc"
            )

        # Add compression encoding
        comp = dict(zlib=True, complevel=5, shuffle=True)
        for var in self.current_ds.data_vars:
            self.current_ds[var].encoding.update(comp)

        print(datetime.now(), "Saving to %s" % (new_filename), flush=True)

        self.current_ds.to_netcdf(new_filename)
        self.current_ds.close()

    def generate_label_map(
        self, unique_labels: np.ndarray, links1: list, links2: list, previous_max: int
    ) -> np.ndarray:
        """
        Generate a label map of contiguous, linked labels for the given unique label list
        """
        max_label = unique_labels.max()

        label_map = np.zeros(max_label + 1, dtype=int)

        label_map[unique_labels] = unique_labels

        # Now reassign labels based on links
        for current_cores, next_cores in zip(links1, links2):
            new_label = current_cores[0]
            if len(current_cores) > 1:
                for core in current_cores[1:]:
                    label_map[core] = new_label
            if len(next_cores) > 0:
                for core in next_cores:
                    label_map[core] = new_label

        # Reassign to contiguous integers
        unique_labels = np.unique(label_map)

        remapper = np.zeros(max_label + 1, dtype=int)

        # First maintain existing labels
        existing_labels = unique_labels[unique_labels <= previous_max]
        remapper[existing_labels] = existing_labels

        # Then add new labels
        new_labels = unique_labels[unique_labels > previous_max]
        remapper[new_labels] = np.arange(new_labels.size) + previous_max + 1

        return remapper[label_map]

    def core_label_map(self) -> np.ndarray:
        """
        Get a label map that maps the cores in the current_ds and next_ds to their new, linked values
        """
        _, _, core_links1, core_links2 = link_dcc_cores(
            self.current_ds.sel(t=slice(self.start_date, None)),
            self.next_ds,
            overlap=self.overlap,
        )

        unique_labels = np.asarray(
            sorted(
                list(
                    set(np.unique(self.current_ds.core_label.data))
                    | set(np.unique(self.next_ds.core_label.data))
                )
            )
        )

        label_map = self.generate_label_map(
            unique_labels, core_links1, core_links2, self.current_max_core_label
        )
        return label_map

    def anvil_label_map(self) -> np.ndarray:
        """
        Get a label map that maps the anvils in the current_ds and next_ds to their new, linked values
        """
        _, _, anvil_links1, anvil_links2 = link_dcc_anvils(
            self.current_ds.sel(t=slice(self.start_date, None)),
            self.next_ds,
            overlap=self.overlap,
        )

        unique_labels = np.asarray(
            sorted(
                list(
                    set(np.unique(self.current_ds.thick_anvil_label.data))
                    | set(np.unique(self.next_ds.thick_anvil_label.data))
                    | set(np.unique(self.current_ds.thin_anvil_label.data))
                    | set(np.unique(self.next_ds.thin_anvil_label.data))
                )
            )
        )

        label_map = self.generate_label_map(
            unique_labels, anvil_links1, anvil_links2, self.current_max_anvil_label
        )
        return label_map

    def relabel_cores(self) -> None:
        label_map = self.core_label_map()

        self.remap_core_labels(label_map)

        self.combine_labels(self.current_ds.core_label, self.next_ds.core_label)

    def remap_core_labels(self, label_map) -> None:
        """
        Relabel cores of current and next ds to contiguous integers, while maintaining the same
        labels for overlapping regions
        """
        # Relabel cores
        self.current_ds.core_label.data = label_map[self.current_ds.core_label.data]
        new_cores = np.unique(label_map[self.current_ds.core.data])
        self.current_ds = self.current_ds.isel(
            {"core": np.arange(new_cores.size)}
        ).assign_coords({"core": new_cores})
        self.current_ds.core_step_core_index.data = label_map[
            self.current_ds.core_step_core_index.data
        ]

        self.next_ds.core_label.data = label_map[self.next_ds.core_label.data]
        new_cores = np.unique(label_map[self.next_ds.core.data])
        self.next_ds = self.next_ds.isel(
            {"core": np.arange(new_cores.size)}
        ).assign_coords({"core": new_cores})
        self.next_ds.core_step_core_index.data = label_map[
            self.next_ds.core_step_core_index.data
        ]

        self.current_max_core_label = np.maximum(
            np.max(self.current_ds.core_label.data), self.current_max_core_label
        )

    #         print(self.current_max_core_label)

    def relabel_anvils(self) -> None:
        label_map = self.anvil_label_map()

        self.remap_anvil_labels(label_map)

        self.combine_labels(
            self.current_ds.thick_anvil_label, self.next_ds.thick_anvil_label
        )
        self.combine_labels(
            self.current_ds.thin_anvil_label, self.next_ds.thin_anvil_label
        )

    def remap_anvil_labels(self, label_map) -> None:
        """
        Relabel anvils of current and next ds to contiguous integers, while maintaining the same
        labels for overlapping regions. Note that we must account for both the thin and thick anvil
        labels
        """
        self.current_ds.thick_anvil_label.data = label_map[
            self.current_ds.thick_anvil_label.data
        ]
        self.next_ds.thick_anvil_label.data = label_map[
            self.next_ds.thick_anvil_label.data
        ]
        self.current_ds.thin_anvil_label.data = label_map[
            self.current_ds.thin_anvil_label.data
        ]
        self.next_ds.thin_anvil_label.data = label_map[
            self.next_ds.thin_anvil_label.data
        ]

        new_anvils = np.unique(label_map[self.current_ds.anvil.data])
        self.current_ds = self.current_ds.isel(
            {"anvil": np.arange(new_anvils.size)}
        ).assign_coords({"anvil": new_anvils})
        new_anvils = np.unique(label_map[self.next_ds.anvil.data])
        self.next_ds = self.next_ds.isel(
            {"anvil": np.arange(new_anvils.size)}
        ).assign_coords({"anvil": new_anvils})

        self.current_ds.thick_anvil_step_anvil_index.data = label_map[
            self.current_ds.thick_anvil_step_anvil_index.data
        ]
        self.current_ds.thin_anvil_step_anvil_index.data = label_map[
            self.current_ds.thin_anvil_step_anvil_index.data
        ]

        self.next_ds.thick_anvil_step_anvil_index.data = label_map[
            self.next_ds.thick_anvil_step_anvil_index.data
        ]
        self.next_ds.thin_anvil_step_anvil_index.data = label_map[
            self.next_ds.thin_anvil_step_anvil_index.data
        ]

        self.current_max_anvil_label = np.maximum(
            np.maximum(
                np.max(self.current_ds.thick_anvil_label.data),
                np.max(self.current_ds.thin_anvil_label.data),
            ),
            self.current_max_anvil_label,
        )

    def combine_labels(
        self, current_labels: xr.DataArray, next_labels: xr.DataArray
    ) -> None:
        """
        Combine the labels from the overlapping regions of two datasets
        """

        # Transfer from next_labels to current_labels

        # This set operation removes "stubs" i.e. labels that should be flagged as end labels
        # Labels that appear in the first step of next_labels, but don't appear in current_labels should be removed
        combine_label_set = (
            (
                set(np.unique(next_labels.sel(t=self.t_overlap[1:-1]).data))
                - set(np.unique(next_labels.sel(t=self.t_overlap[0]).data))
            )
            | set(np.unique(current_labels.sel(t=self.t_overlap[:-1]).data))
        ) - set([0])

        wh_combine = np.logical_and(
            np.isin(
                next_labels.sel(t=self.t_overlap[1:-1]).data, list(combine_label_set)
            ),
            current_labels.sel(t=self.t_overlap[1:-1]).data == 0,
        )

        current_labels.loc[self.t_overlap[1:-1]] += (
            next_labels.sel(t=self.t_overlap[1:-1]).data * wh_combine
        )

        # Now transfer from current_labels to next_labels

        combine_label_set = (
            (
                set(np.unique(current_labels.sel(t=self.t_overlap[1:-1]).data))
                - set(np.unique(current_labels.sel(t=self.t_overlap[-1]).data))
            )
            | set(np.unique(next_labels.sel(t=self.t_overlap[1:]).data))
        ) - set([0])

        wh_combine = np.logical_and(
            np.isin(
                current_labels.sel(t=self.t_overlap[1:-1]).data, list(combine_label_set)
            ),
            next_labels.sel(t=self.t_overlap[1:-1]).data == 0,
        )

        next_labels.loc[self.t_overlap[1:-1]] += (
            current_labels.sel(t=self.t_overlap[1:-1]).data * wh_combine
        )

    def relabel_next_ds(self) -> None:
        """
        Change all labels in self.next_ds to start with values larger than those in self.current_ds
        """
        max_core = np.maximum(
            self.current_max_core_label, self.current_ds.core.data.max()
        )
        max_anvil = np.maximum(
            self.current_max_anvil_label, self.current_ds.anvil.data.max()
        )

        # Relabel coords
        self.next_ds = self.next_ds.assign_coords(
            {
                "core": self.next_ds.core.data + max_core,
                "anvil": self.next_ds.anvil.data + max_anvil,
            }
        )

        # Relabel core labels
        self.next_ds.core_label.data[self.next_ds.core_label.data != 0] += max_core
        self.next_ds.core_step_core_index.data += max_core

        # Relabel anvil labels
        self.next_ds.thick_anvil_label.data[
            self.next_ds.thick_anvil_label.data != 0
        ] += max_anvil
        self.next_ds.thin_anvil_label.data[
            self.next_ds.thin_anvil_label.data != 0
        ] += max_anvil
        self.next_ds.thick_anvil_step_anvil_index.data += max_anvil
        self.next_ds.thin_anvil_step_anvil_index.data += max_anvil
