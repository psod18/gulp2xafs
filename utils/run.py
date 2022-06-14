import os
import re
from typing import List

import numpy as np

from utils.tools import (
    FEFFInputs,
    FEFFChiSpectra,
    XYZStructure,
)
from utils.data_reader import GULPHistory


# TODO: replace print statement with logging object

class Master:
    atom_pattern = re.compile(r"^([a-zA-Z]+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")
    bases_pattern = re.compile(r"^\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+")

    def __init__(
            self, input_file: str, absorber: str, modes: List[str],
            shift: bool, frames: int = None, cores: int = 1,

    ):
        """
        :input_file: GULP MD *.history file output (DL_POLY)
        :absorber: chemical element as X-Ray absorbing center (e.g. Mn)
        :center: if True, shift origins of the structure to the structure box center
        :frames: number of frames (snapshots) to be saved as xyz file structures: if 0 write all
        :md_frames: number of frames (snapshots) to be used as xyz file for JMOL structure file: if 0 write all
        :cores: number of processors that can be utilise during calculation
        """
        print("Next modes will be initialized: {0}".format(modes))
        self.modes = modes

        self.history_file_path = input_file
        self.root_path = os.path.dirname(input_file)
        self.move_origin = shift
        self.frames = frames
        self.model_name = os.path.splitext(os.path.basename(input_file))[0]

        # get lattice basis set
        self.lattice_vectors = self._pars_cell_parameters()

        # snapshots extractor
        self.history = GULPHistory(
            self.history_file_path,
            self.get_structure_box_size(),
        )

        # *.xyz structure files creator
        self.xyz = XYZStructure(
            root=self.root_path,
            cores=cores,
            model_name=self.model_name,
            frames=self.frames,
        )
        # feff.inp structure files creator
        self.feff = FEFFInputs(
            root=self.root_path,
            cores=cores,
            absorber=absorber,
            structure_size=self.get_structure_box_size(),
        )
        # chi.dat spectra calculator
        self.feff_runner = FEFFChiSpectra(
            root=self.root_path,
            cores=cores,
            model_name=self.model_name
        )

    def get_structure_box_size(self) -> List[float]:
        """
        return vector of (a,b,c) values, where a,b,c are the width, depth, high (in x, y, z direction respectively)
        """
        return (np.dot(self.lattice_vectors, np.ones((3, 1)))).squeeze().tolist()

    def _pars_cell_parameters(self):
        """
        Extract basis matrix for given structure:
        ...
        timestep        20       513         1         3    0.000200
           22.60       0.000       0.000                                # target line
           0.000       22.60       0.000                                # target line
           0.000       0.000       22.60                                # target line
        Mn               1   54.940000    0.000000                      # break line
        ...
        """
        bases_set = []
        with open(self.history_file_path, 'r') as input_file:
            for line in input_file:
                if re.findall(self.bases_pattern, line):
                    bases_set.append(line.strip())
                elif re.findall(self.atom_pattern, line):
                    break
        if bases_set:
            print("Basis set was successfully extracted:")
        return np.genfromtxt(bases_set)

    def execute(self):
        """
        Run all chosen procedures one by one:
        Parse data from *.history file -> create set of *.xyz structure files -> create set of feff.inp files ->
        calculate chi.dat spectra (and average at the end of execution)
        """
        for v in self.lattice_vectors:
            print("{: .3f}\t{: .3f}\t{: .3f}".format(*v))

        if any([mode in self.modes for mode in ("xyz", "feff")]):
            self.history.extreact_data(shift=self.move_origin)

        if "xyz" in self.modes:
            try:
                print("Creating xyz structure files")
                self.xyz(self.history.snapshots)
            except Exception as e:  # TODO: do narrow try-except block
                raise e
        if "feff" in self.modes:
            try:
                print("Creating feff.inp files")
                self.feff(self.history.snapshots)
            except Exception as e:  # TODO: do narrow try-except block
                raise e
        if "chi" in self.modes:
            print("Run feff spectra calculation")
            try:
                self.feff_runner()
            except Exception as e:  # TODO: do narrow try-except block
                raise e
