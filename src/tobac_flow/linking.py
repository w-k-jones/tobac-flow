import numpy as np
import xarray as xr
import pathlib
from typing import Callable

from tobac_flow.utils import get_dates_from_filename

# Functions to link overlapping labels
def recursive_linker(
    links_list1=None,
    links_list2=None,
    label_list1=None,
    label_list2=None,
    overlap_list1=None,
    overlap_list2=None,
):
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


def link_labels(labels1, labels2, overlap=0):
    label_list1 = np.unique(labels1[labels1 != 0]).tolist()
    label_list2 = np.unique(labels2[labels2 != 0]).tolist()

    bins1 = np.cumsum(np.bincount(labels1.ravel()))
    args1 = np.argsort(labels1.ravel())

    bins2 = np.cumsum(np.bincount(labels2.ravel()))
    args2 = np.argsort(labels2.ravel())

    overlap_list1 = [
        [
            j
            for j in np.unique(labels2.ravel()[args1[bins1[i - 1] : bins1[i]]])
            if j > 0
            and (
                np.count_nonzero(labels2.ravel()[args1[bins1[i - 1] : bins1[i]]] == j)
                >= overlap
                * np.minimum(bins1[i] - bins1[i - 1], bins2[j] - bins2[j - 1])
            )
        ]
        for i in label_list1
    ]
    overlap_list2 = [
        [
            j
            for j in np.unique(labels1.ravel()[args2[bins2[i - 1] : bins2[i]]])
            if j > 0
            and (
                np.count_nonzero(labels1.ravel()[args2[bins2[i - 1] : bins2[i]]] == j)
                >= overlap
                * np.minimum(bins2[i] - bins2[i - 1], bins1[j] - bins1[j - 1])
            )
        ]
        for i in label_list2
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
def link_dcc_cores(dcc_ds1, dcc_ds2, overlap=0):
    t_overlap_list = sorted(list(set(dcc_ds1.t.data) & set(dcc_ds2.t.data)))[1:-1]

    core_step_links1, core_step_links2 = link_labels(
        dcc_ds1.core_step_label.sel(t=t_overlap_list).data,
        dcc_ds2.core_step_label.sel(t=t_overlap_list).data,
        overlap=overlap,
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
def link_dcc_anvils(dcc_ds1, dcc_ds2, overlap=0):
    t_overlap_list = sorted(list(set(dcc_ds1.t.data) & set(dcc_ds2.t.data)))[1:-1]

    anvil_step_links1, anvil_step_links2 = link_labels(
        dcc_ds1.thick_anvil_step_label.sel(t=t_overlap_list).data,
        dcc_ds2.thick_anvil_step_label.sel(t=t_overlap_list).data,
        overlap=overlap,
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
        overlap: float = 0.5,
    ) -> None:
        self.files = [pathlib.Path(filename) for filename in files]
        for filename in self.files:
            if not filename.exists:
                raise ValueError(f"File {filename} does not exist")
        self.output_func = output_func
        self.overlap = overlap
        self.current_max_core_label = 0
        self.current_max_anvil_label = 0

        self.current_filename = self.files.pop(0)
        self.start_date = get_dates_from_filename(self.current_filename)[0]
        self.current_ds = xr.open_dataset(self.current_filename).sel(
            t=slice(self.start_date, None)
        )

    def process_next_file(self) -> None:
        self.next_filename = self.files.pop(0)
        self.start_date = get_dates_from_filename(self.next_filename)[0]
        self.next_ds = xr.open_dataset(self.next_filename)
        self.relabel_next_ds()

        # Check if there is some overlap
        self.t_overlap = sorted(
            list(set(self.current_ds.t.data) & set(self.next_ds.t.data))
        )
        if len(self.t_overlap) > 2:
            self.relabel_cores()
            self.relabel_anvils()

        self.output_current_ds()
        self.current_ds = self.next_ds.sel(t=slice(self.start_date, None))
        self.current_filename = self.next_filename

    def process_files(self) -> None:
        while len(self.files) > 0:
            self.process_next_file()
        self.start_date = get_dates_from_filename(self.next_filename)[1]
        self.output_current_ds()

    def output_current_ds(self) -> None:
        self.current_ds = self.current_ds.sel(t=slice(None, self.start_date))

        # do something else...
        self.output_func(self.current_ds)

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

        # First maintain existing labels
        existing_labels = unique_labels[unique_labels <= previous_max]
        label_map[existing_labels] = existing_labels

        # Then add new labels
        new_labels = unique_labels[unique_labels > previous_max]
        label_map[new_labels] = np.arange(new_labels.size) + previous_max + 1

        return label_map

    def core_label_map(self) -> np.ndarray:
        """
        Get a label map that maps the cores in the current_ds and next_ds to their new, linked values
        """
        _, _, core_links1, core_links2 = link_dcc_cores(
            self.current_ds, self.next_ds, overlap=self.overlap
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
            self.current_ds, self.next_ds, overlap=0.5
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
        self.combine_labels(self.current_ds.core_label, self.next_ds.core_label)

        label_map = self.core_label_map()

        self.remap_core_labels(label_map)

    def remap_core_labels(self, label_map) -> None:
        """
        Relabel cores of current and next ds to contiguous integers, while maintaining the same
        labels for overlapping regions
        """
        # Relabel cores
        self.current_ds.core_label.data = label_map[self.current_ds.core_label.data]
        self.current_ds = self.current_ds.assign_coords(
            {"core": label_map[self.current_ds.core.data]}
        )
        self.current_ds.core_step_core_index.data = label_map[
            self.current_ds.core_step_core_index.data
        ]

        self.next_ds.core_label.data = label_map[self.next_ds.core_label.data]
        self.next_ds = self.next_ds.assign_coords(
            {"core": label_map[self.next_ds.core.data]}
        )
        self.next_ds.core_step_core_index.data = label_map[
            self.next_ds.core_step_core_index.data
        ]

        self.current_max_core_label = np.maximum(
            np.max(self.current_ds.core_label.data), self.current_max_core_label
        )

    #         print(self.current_max_core_label)

    def relabel_anvils(self) -> None:
        self.combine_labels(
            self.current_ds.thick_anvil_label, self.next_ds.thick_anvil_label
        )
        self.combine_labels(
            self.current_ds.thin_anvil_label, self.next_ds.thin_anvil_label
        )

        label_map = self.anvil_label_map()

        self.remap_anvil_labels(label_map)

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

        self.current_ds = self.current_ds.assign_coords(
            {"anvil": label_map[self.current_ds.anvil.data]}
        )
        self.next_ds = self.next_ds.assign_coords(
            {"anvil": label_map[self.next_ds.anvil.data]}
        )

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
        wh_zero = current_labels.sel(t=self.t_overlap[1:-1]).data != 0
        current_labels.sel(t=self.t_overlap[1:-1]).data[wh_zero] = next_labels.sel(
            t=self.t_overlap[1:-1]
        ).data[wh_zero]
        next_labels.sel(t=self.t_overlap[1:-1]).data = current_labels.sel(
            t=self.t_overlap[1:-1]
        ).data

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
                # "core_step":self.next_ds.core_step.data + self.current_ds.core_step.data.max(),
                "core": self.next_ds.core.data + max_core,
                # "thick_anvil_step":self.next_ds.thick_anvil_step.data + self.current_ds.thick_anvil_step.data.max(),
                # "thin_anvil_step":self.next_ds.thin_anvil_step.data + self.current_ds.thin_anvil_step.data.max(),
                "anvil": self.next_ds.anvil.data + self.current_ds.anvil.data.max(),
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
