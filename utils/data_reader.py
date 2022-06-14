#!/usr/bin/env python3
import os
import re
from typing import (
    Optional,
    Union,
    NewType,
    Sequence,
)

import numpy as np


Coef = NewType('Coef', Union[Sequence[float], float])


class GULPHistory:

    atom_pattern = re.compile(r"^([a-zA-Z]+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")
    xyz_pattern = re.compile(r"-?\d\.\d+E[-+]\d{2}")

    def __init__(self, file_path, structure_size):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path).split('.')[0]

        self.struct_box_size = structure_size

        # params to be extracted
        self.elements = None
        self.number_of_atoms = None
        self.snapshots = None

    def center_origins(self, shift: Optional[Coef] = None):
        """
        Shift origins (0,0,0) to the center of given structure
        (by default GULP/DL_POLY put origin point in left down corner of the structure box)
        """
        if shift:
            if isinstance(shift, float):
                for v in 'xyz':
                    j = np.where((self.snapshots[v] < -shift/2) | (self.snapshots[v] > shift/2))
                    self.snapshots[v][j] = (np.abs(self.snapshots[v][j]) / self.snapshots[v][j]) * np.abs(
                        self.snapshots[v][j]) - shift
            elif isinstance(shift, Sequence) and len(shift) == 3:
                for i, v in enumerate('xyz'):
                    j = np.where((self.snapshots[v] < -shift[i]/2) | (self.snapshots[v] > shift[i]/2))
                    self.snapshots[v][j] = (np.abs(self.snapshots[v][j]) / self.snapshots[v][j]) * np.abs(
                        self.snapshots[v][j]) - shift[i]
            else:
                raise ValueError("Value 'shift' must be float or sequence of floats with length eq. 3")

    def extreact_data(self, shift: bool):
        """
        Extract list of snapshots from *.history file (GULP/DL_POLY)
        """
        self._read_history_file()
        print("All snapshots are extracted: {0}".format(self.snapshots.__len__()))
        if shift:
            self.center_origins(self.struct_box_size)
            print("Coordinates origin was moved to the center of the cell box")

        self.number_of_atoms = self.snapshots[0].__len__()
        print("Expected Number of atoms in each snapshot: {0}".format(self.number_of_atoms))

        self.elements = np.unique(self.snapshots[0]['label']).tolist()
        print("Chemical elements in structure:", self.elements)

    def _read_history_file(self):
        """
        Read *.history file line by line and create list of numpy arrays (structured) from each unit snapshot
        """
        snapshots = []
        _data = []
        _row = ''

        with open(self.file_path, 'r') as input_file:
            for line in input_file:
                if line.strip().startswith('timestep'):
                    if _row:
                        _data.append(" ".join(_row.split()))
                    if _data:
                        snapshots.append(self._convert_string_array_to_numpy(data=_data))
                    _data = []
                    _row = ''
                    continue
                else:
                    if re.findall(self.xyz_pattern, line):
                        _row += line
                    else:
                        if re.findall(self.atom_pattern, line):
                            if _row:
                                _data.append(" ".join(_row.split()))
                            _row = line
            if _row:
                _data.append(" ".join(_row.split()))
            if _data:
                snapshots.append(self._convert_string_array_to_numpy(data=_data))
        self.snapshots = np.r_[snapshots]

    @staticmethod
    def _convert_string_array_to_numpy(data):
        """
        Convert list of strings to numpy array:
        data entries example:
        'Mn    1     54.940000    0.000000 8.6029E-02 -1.6953E-02  6.6800E-04 2.2712E+00  1.0618E-01  8.7233E-01'
        :param data: list of strings
        :return: named numpy array with mixed data type:
        | <label> | <idx> |  <mass> | <charge> | <x> | <y> | <z> | <vx> | <vy> | <vz> |
        -------------------------------------------------------------------------------
        |   U10   |  i4   |   f4    |    f4    |  f4 |  f4 |  f4 |  f4  |  f4  |  f4  |
        """

        column_type = [
                ("label", "U5"),  # chemical element tag
                ("idx", "i4"),  # atom index in structure
                ("mass", "f4"),  # atomic mass
                ("charge", "f4"),  # charge in certain site
                ("x", "f8"),
                ("y", "f8"),
                ("z", "f8"),
                ("vx", "f8"),  # velocity projection on 'x' direction
                ("vy", "f8"),  # velocity projection on 'y' direction
                ("vz", "f8"),  # velocity projection on 'z' direction
            ]

        return np.genfromtxt(data, delimiter=" ", dtype=column_type)
