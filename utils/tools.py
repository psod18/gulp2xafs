"""
Configuration of output files bundle (e.g. chi spectra, feff inputs, xyz)
"""
import os
import platform
import shutil

from abc import ABC, abstractmethod
from typing import List

import multiprocessing as mp
import numpy as np
import numpy.lib.recfunctions as rfn
import subprocess as sp

from feff.exe.sys_path import feff_exec
from utils.elements import atomic_number


class BaseManager(ABC):

    @staticmethod
    def _create_catalog(location: str) -> None:
        if not os.path.exists(location) or not os.path.isdir(location):
            os.mkdir(location)

    @abstractmethod
    def _execute(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class XYZStructure(BaseManager):

    name = "xyz"
    fmt = "{:3s}\t{: .5f}\t{: .5f}\t{: .5f}\n"

    def __init__(self, root: str, cores, model_name: str, frames: int):
        self.root = root
        self.cores = cores
        self.model_name = model_name
        self.frames = frames
        self.target_catalog = os.path.join(self.root, self.name)

    def _execute(self, idx: int, snapshot: np.array):
        out = os.path.join(self.target_catalog, "{}_{:05d}.xyz".format(self.name, idx))
        with open(out, "w") as output_file:
            output_file.write("{0}\n\n".format(len(snapshot)))
            for line in snapshot:
                output_file.write(self.fmt.format(line["label"], line["x"], line["y"], line["z"]))

    def __call__(self, snapshots: np.array):
        self._create_catalog(self.target_catalog)
        with mp.Pool(self.cores) as pool:
            pool.starmap(self._execute, enumerate(snapshots, start=1))

        if self.frames is not None:
            if self.frames == 0:
                self._write_jmol_md(snapshots)
            else:
                self._write_jmol_md(snapshots[:self.frames])

    def _write_jmol_md(self, snapshots: np.array, ):
        out = os.path.join(self.root, "{}_md.xyz".format(self.model_name))
        with open(out, "w") as output_file:
            for snapshot in snapshots:
                output_file.write("{0}\n\n".format(len(snapshot)))
                for line in snapshot:
                    output_file.write(self.fmt.format(line["label"], line["x"], line["y"], line["z"]))
                output_file.write("\n")


class FEFFInputs(BaseManager):

    feff_input_header = None
    feff_potential_table = None

    name = "feff"
    #         x        y        z     ipot  tag   distance idx
    fmt = "\t{: .5f}\t{: .5f}\t{: .5f}\t{:2d}\t{:3s}\t{: .5f}\t{:5d}\n"

    def __init__(self, root: str, cores: int, absorber: str, structure_size: List[float]):
        self.root = root
        self.cores = cores
        self.absorber = absorber
        self.tec = 0  # target element count
        self.box = [i / 2 for i in structure_size]
        self.target_catalog = os.path.join(self.root, self.name)
        self.ipot_dict = {f" {self.absorber}": 0}  # whitespace is workaround to have pseudo-identical keys in dict

    def load_header(self, header_file_path):
        """
        load FEFF header from `feff_header.txt` file located in `src` files
        """
        with open(header_file_path) as header_file:
            self.feff_input_header = "".join([line for line in header_file])

    def construct_potential_table(self, snapshot: np.array):
        """
        Create
                0	25		Mn
                1	33		As
                ...
        """
        ipot_str = "\t\t{ipot}\t{Z}\t{element}\n"
        self.feff_potential_table = ''
        atoms_count = dict(zip(*np.unique(snapshot["label"], return_counts=True)))
        if self.absorber not in atoms_count:
            raise ValueError("Target absorbing atom '{0}' was not find in structure".format(self.absorber))
        self.tec = atoms_count[self.absorber]  # target element count
        if atoms_count[self.absorber] == 1:
            del atoms_count[self.absorber]
        self.ipot_dict.update((v, i) for i, v in enumerate(atoms_count.keys(), start=1))

        for el, ipot in sorted(self.ipot_dict.items(), key=lambda x: x[1]):
            self.feff_potential_table += ipot_str.format(ipot=ipot, Z=atomic_number[el.strip()], element=el.strip())

    def prepare_feff_inputs(self, snapshots: np.array):
        _data = []
        for s in snapshots:
            _ipot = np.array([self.ipot_dict.get(el, 0) for el in s["label"]])

            _data.append(rfn.append_fields(s, ('ipot',), [_ipot], ('i4',)))

        return np.array(_data)

    def __call__(self, snapshots: np.array):

        print("Load header from:")
        print(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feff_header.txt'))
        self.load_header(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feff_header.txt'))
        self.construct_potential_table(snapshots[0])

        _snapshots = self.prepare_feff_inputs(snapshots)
        self._create_catalog(self.target_catalog)
        with mp.Pool(self.cores) as pool:
            pool.starmap(self._execute, zip(np.arange(start=1, stop=self.tec * len(_snapshots)+1, step=self.tec),
                                            _snapshots))

    def _execute(self, idx: int, snapshot: np.array):
        _idx = idx
        _centers = snapshot[snapshot["label"] == self.absorber]
        for _c in _centers:
            _data = np.empty_like(snapshot)
            _data[:] = snapshot
            for coord in "xyz":
                _data[coord] = _data[coord] - _c[coord]

            for i, v in enumerate('xyz'):
                j = np.where((_data[v] < -self.box[i]) | (_data[v] > self.box[i]))
                _data[v][j] = (np.abs(_data[v][j]) / _data[v][j]) * (self.box[i] - (np.abs(_data[v][j]) - self.box[i]))

            _dist = np.sqrt(_data["x"] ** 2 + _data["y"] ** 2 + _data["z"] ** 2)
            _data = rfn.append_fields(_data, ('distance',), [_dist], ('f8',))
            _data = _data[_data['distance'].argsort()]
            _data[0]["ipot"] = 0
            out = os.path.join(self.target_catalog, "{}_{:05d}.inp".format(self.name, _idx))
            with open(out, 'w') as output_file:
                output_file.write(self.feff_input_header)
                output_file.write(self.feff_potential_table)
                output_file.write("ATOMS\t\t\t\t* this list contains {0} atoms\n".format(len(_data)))
                for line in _data:
                    output_file.write(
                        self.fmt.format(
                            line["x"],
                            line["y"],
                            line["z"],
                            line["ipot"],
                            line["label"],
                            line["distance"],
                            line["idx"],
                        )
                    )
                output_file.write("END\n")
            _idx += 1


class FEFFChiSpectra(BaseManager):
    name = "chi"

    def __init__(self, root: str, cores: int, model_name: str):
        self.root = root
        self.target_catalog = os.path.join(self.root, self.name)
        self.cores = cores
        self.model_name = model_name
        self.feff_inputs_path = os.path.join(self.root, "feff")

        # do some additional stuff
        self.input_files_list = []
        self.temp_catalogs = [os.path.join(self.target_catalog, "tmp{0}".format(i)) for i in range(cores)]

    def _prepare_slices(self):
        def slicer(data, step):
            if data:
                while len(data) // step > 1:
                    yield data[:step]
                    data = data[step:]
                else:
                    yield data
        list_of_files = [file for file in os.listdir(self.feff_inputs_path) if file.endswith(".inp")]
        self.input_files_list = [s for s in slicer(list_of_files, len(list_of_files)//self.cores)]

    def calculate_average_chi(self):
        header = "\t\tk\t\t<chi>\t\tstd"

        files = [f for f in os.listdir(self.target_catalog) if f.endswith('.dat')]
        if not files:
            raise NoChiSpectraException
        spectra = np.full((len(files), 401), np.nan)

        k = None
        for idx, file in enumerate(files):
            data = np.loadtxt(os.path.join(self.target_catalog, file), dtype=np.float64, usecols=(0, 1))
            if data.shape[0] == 400:
                spectra[idx][1:] = data[:, 1]
            else:
                spectra[idx] = data[:, 1]
                if k is None:
                    k = data[:, 0]

        aver_spectrum = np.nanmean(spectra, axis=0)
        std_spectrum = np.nanstd(spectra, axis=0)
        outfile = os.path.join(self.target_catalog, "result_{}.txt".format(self.model_name))

        np.savetxt(outfile, np.stack([k, aver_spectrum, std_spectrum]).T, header=header, fmt="%.5e")

    def _execute(self, inputs: List[str], tmp_dir: str, err_dir: str):
        # go to the tmp directory:
        os.chdir(tmp_dir)

        for inp in inputs:
            file_index = os.path.splitext(inp)[0].split("_")[-1]  # get numeric label
            # copy feff.inp file to tmp directory:
            shutil.copyfile(os.path.join(self.root, "feff", inp), os.path.join(tmp_dir, 'feff.inp'))

            # run the feff calculation:
            if platform.system().lower() == 'windows':
                cmd = "feff.exe"
            else:
                cmd = 'wine feff.exe'  # assumes that wine is installed on linux machine
            sp.call(cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)

            # Check if chi.dat is created:
            if os.path.isfile(os.path.join(tmp_dir, 'chi.dat')):
                chi_out_path = os.path.join(self.target_catalog, "chi_{}.dat".format(file_index))
                shutil.move(os.path.join(tmp_dir, 'chi.dat'), chi_out_path)
            else:
                # if chi.dat is absent
                shutil.move(os.path.join(tmp_dir, 'feff.inp'), os.path.join(err_dir, "{0}.inp".format(file_index)))

        # to release directory from subprocesses
        os.chdir("..")

    def __call__(self):
        self._prepare_slices()
        if self.input_files_list:
            self._create_catalog(self.target_catalog)
            for tmp in self.temp_catalogs:
                self._create_catalog(tmp)
                shutil.copyfile(feff_exec, os.path.join(tmp, "feff.exe"))

            # create dir for error
            err_path = os.path.join(self.target_catalog, 'err')
            self._create_catalog(err_path)

            with mp.Pool(self.cores) as pool:
                pool.starmap(self._execute, zip(self.input_files_list, self.temp_catalogs, [err_path] * self.cores))

            print("All calculation is done. Now is time to clean up!")
            for catalog in self.temp_catalogs:
                shutil.rmtree(catalog)

            if len(os.listdir(err_path)) == 0:
                shutil.rmtree(err_path)
            else:
                print("Some calculations failed: {0}. Check the {1}".format(len(os.listdir(err_path)), err_path))
            try:
                self.calculate_average_chi()
            except NoChiSpectraException:
                print("Cannot calculate average spectra, because any chi spectra was not found")

        else:
            print("Catalog with feff inputs files is empty!")


class NoChiSpectraException(Exception):
    pass
