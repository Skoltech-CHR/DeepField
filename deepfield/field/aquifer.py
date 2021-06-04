"""Classes and routines for handling aquifers."""
from collections import OrderedDict
import numpy as np
import h5py

from .base_component import BaseComponent

class Aquifers(BaseComponent):
    """
    Aquifers component class.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._aquifers = OrderedDict()

    def items(self):
        """Returns pairs of aquifer's names and data."""
        return self._aquifers.items()

    def copy(self):
        """Returns a deepcopy of attributes. Cached properties are not copied."""
        copy = super().copy()
        for name in self.names:
            copy[name] = self[name].copy()
        return copy

    @property
    def names(self):
        """Returns names of the aquifers."""
        return tuple(self._aquifers.keys())

    def randomize(self, randomizers, inplace=False):
        """Randomize properties of the aquifers

        Parameters
        ----------
        randomizers : dict
            Dict of the structure {parameter_name: Callable returning value, being added to the original value}.
        inplace : bool, optional
            Perform operations inplace or return copy, by default False

        Returns
        -------
        Aquifers
            Randomized aquifers.
        """
        if inplace:
            aquifers = self
        else:
            aquifers = self.copy()
        for aq_name in aquifers.names:
            aquif = aquifers[aq_name]
            for name, r in randomizers.items():
                setattr(aquif, name, getattr(aquif, name) + r())
        return aquifers

    def __getitem__(self, key):
        return self._aquifers[key]

    def __setitem__(self, key, value):
        self._aquifers[key] = value

    def _read_buffer(self, buffer, attr, **kwargs):
        """Load aquifers data from an ASCII file.

        Parameters
        ----------
        buffer : StringIteratorIO
            Buffer to get string from.
        attr : str
            Data format.

        Returns
        -------
        comp : Aquifers
            Aquifers component with loaded aquifers data.
        """
        if attr == 'AQCT':
            return self._load_aqct(buffer, **kwargs)
        if attr == 'AQCO':
            return self._load_aqco(buffer, **kwargs)
        if attr == 'AQUANCON':
            return self._load_aquancon(buffer, **kwargs)
        if attr == 'AQUCT':
            return self._load_aquct(buffer, **kwargs)
        return self

    def _load_aquancon(self, buffer, logger=None, **kwargs):
        """load AQUANCON keyword"""
        _ = kwargs
        while True:
            line = next(buffer)
            if line.strip()[0] == '/':
                break
            if len(line.strip()) == 0:
                continue
            params = (line.replace('/', '').split())
            if params[0] not in self._aquifers:
                self[params[0]] = Aquifer(name=params[0])

            properties = {
                'slices' : (
                    slice(int(params[1])-1, int(params[2])),
                    slice(int(params[3])-1, int(params[4])),
                    slice(int(params[5])-1, int(params[6]))
                    ),
                'face': params[7]
            }
            for key, value in properties.items():
                if key in self[params[0]].attributes and logger is not None:
                    logger.warning('Aquifer {} has attribute {}. Will be replaced.'.format(
                        params[0], key
                    ))
                setattr(self[params[0]], key, value)
        return self

    def _load_aquct(self, buffer, logger=None, **kwargs):
        """load AQUCT keyword"""
        _ = kwargs
        while True:
            line = next(buffer)
            if line.strip()[0] == '/':
                break
            if len(line.strip()) == 0:
                continue
            params = (line.replace('/', '').split())
            if params[0] not in self._aquifers:
                self[params[0]] = Aquifer(name=params[0])

            properties = {
                'depth' : float(params[1]),
                'initial_pressure': float(params[2]),
                'perm' : float(params[3]),
                'poro' : float(params[4]),
                'compressibility': float(params[5]),
                'r': float(params[6]),
                'height': float(params[7]),
                'angle': float(params[8]),
            }
            for key, value in properties.items():
                if key in self[params[0]].attributes and logger is not None:
                    logger.warning('Aquifer {} has attribute {}. Will be replaced.'.format(
                        params[0], key
                    ))
                setattr(self[params[0]], key, value)
        return self

    def _load_aqct(self, buffer, logger=None, **kwargs):
        """load AQCT keyword."""
        _ = kwargs
        line = next(buffer)
        params = line.split()
        if params[1] not in self._aquifers:
            self[params[1]] = Aquifer(name=params[0])
        properties = {
            'depth' : float(params[2]),
            'perm' : float(params[3]),
            'poro' : float(params[4]),
            'compressibility': float(params[5]),
            'r': float(params[6]),
            'angle': float(params[7]),
            'height': float(params[8]),
            'initial_pressure': float(params[9]),
            'viscosity': float(params[10]),
            'equi': params[11],
        }
        for key, value in properties.items():
            if key in self[params[1]].attributes and logger is not None:
                logger.warning('Aquifer {} has attribute {}. Will be replaced.'.format(
                    params[0], key
                ))
            setattr(self[params[1]], key, value)
        return self

    def crop_minimal_cube(self, slices):
        """
        Modify aquifers' cell inndices with respect to
        slices representing grid crop.

        Parameters
        ----------
        slices : Iterable(slices)
            slices

        Returns
        -------
        Aquifers
            Aquifers with changed well indices.
        """
        for aquifer in self._aquifers.values():
            aquifer.crop_minimal_cube(slices)
        return self

    def _load_aqco(self, buffer, logger=None):
        """load AQCO keyword."""
        line = next(buffer)
        params = line.split()

        if params[1] not in self._aquifers:
            self[params[1]] = Aquifer(name=params[1])
        properties = {
            'slices' : (
                slice(int(params[2])-1, int(params[3])),
                slice(int(params[4])-1, int(params[5])),
                slice(int(params[6])-1, int(params[7]))
                ),
            'face': params[8]
        }
        for key, value in properties.items():
            if key in self[params[1]].attributes and logger is not None:
                logger.warning('Aquifer {} has attribute {}. Will be replaced.'.format(
                    params[0], key
                ))
            setattr(self[params[1]], key, value)
        return self

    def _dump_hdf5(self, path, mode='a', state=True, **kwargs):  #pylint: disable=too-many-branches
        """Save data into HDF5 file.

        Parameters
        ----------
        path : str
            Path to output file.
        mode : str
            Mode to open file.
            'w': write, a new file is created (an existing file with
            the same name would be deleted).
            'a': append, an existing file is opened for reading and writing,
            and if the file does not exist it is created.
            Default to 'a'.
        state : bool
            Dump compoments's state.

        Returns
        -------
        comp : Aquifers
            Aquifers unchanged.
        """
        _ = kwargs
        with h5py.File(path, mode) as f:
            aquifers = f[self.class_name] if self.class_name in f else f.create_group(self.class_name)
            if state:
                for k, v in self.state.as_dict().items():
                    aquifers.attrs[k] = v
            for aquif in self._aquifers.values():
                aq_path = aquif.name
                grp = aquifers[aq_path] if aq_path in aquifers else aquifers.create_group(aq_path)

                if 'data' not in grp:
                    grp_data = aquifers.create_group(aq_path + '/data')
                else:
                    grp_data = grp['data']
                data = aquif.data_dict()
                for att, data in data.items():
                    if att in grp_data:
                        del grp[att]
                    grp_data.create_dataset(att, data=data)
        return self

    def _load_hdf5(self, path, attrs=None, **kwargs):
        """Load data from a HDF5 file.

        Parameters
        ----------
        path : str
            Path to file to load data from.
        attrs : str or array of str, optional
            Array of dataset's names to get from file. If not given, loads all.

        Returns
        -------
        Aquifers
            Aquifers with loaded attributes.
        """
        _ = kwargs
        if isinstance(attrs, str):
            attrs = [attrs]

        with h5py.File(path, 'r') as f:
            self.set_state(**dict(f[self.class_name].attrs.items()))
            aquifers = f[self.class_name]
            for name, group in aquifers.items():
                self._aquifers[name] = Aquifer(**{
                    k: v[()] for k, v in group['data'].items()
                })
        return self

    def _dump_ascii(self, path, mode='w', **kwargs):
        """Save data into text file.

        Parameters
        ----------
        path : str
            Path to output file.
        attr : str
            Attribute to dump into file.
        mode : str
            Mode to open file.
            'w': write, a new file is created (an existing file with
            the same name would be deleted).
            'a': append, an existing file is opened for reading and writing,
            and if the file does not exist it is created.
            Default to 'w'.

        Returns
        -------
        comp : Aquifers
            Aquifers unchanged.
        """
        # imort pdb; pdb.set_trace()
        with open(path, mode) as f:
            f.write('AQUANCON\n')
            for i, aquifer in enumerate(self._aquifers.values()):
                f.write(
                    (
                        '{name}\t{i1}\t{i2}\t{j1}\t{j2}\t{k1}\t{k2}\t' +
                        '{face} /\n').format(
                            name=i,
                            i1=aquifer.slices[0].start+1,
                            i2=aquifer.slices[0].stop,
                            j1=aquifer.slices[1].start+1,
                            j2=aquifer.slices[1].stop,
                            k1=aquifer.slices[2].start+1,
                            k2=aquifer.slices[2].stop,
                            face=aquifer.face
                        ))
            f.write('/\n\n')
            f.write('AQUCT\n')
            for i, aquifer in enumerate(self._aquifers.values()):
                f.write(
                    (
                        '{name}\t{depth}\t{initial_pressure}\t{perm}\t{poro}\t{compressibility}\t{r}\t' +
                        '{height}\t{angle} /\n').format(
                            name=i,
                            depth=aquifer.depth,
                            perm=aquifer.perm,
                            poro=aquifer.poro,
                            compressibility=aquifer.compressibility,
                            r=aquifer.r,
                            angle=aquifer.angle,
                            height=aquifer.height,
                            initial_pressure=aquifer.initial_pressure,
                        )
                    )
            f.write('/\n')
        return self

class Aquifer(BaseComponent):
    "Class representing specific aquifer object"
    def __init__(self, **kwargs):
        if 'SLICES' in kwargs:
            kwargs['SLICES'] = tuple(a if isinstance(a, slice) else slice(*a) for a in kwargs['SLICES'])
        super().__init__(**kwargs)

    def data_dict(self):
        """create dictionary with aquifer representation
        suitable for further serialization to hdf5

        Returns
        -------
        dict
            dict with aquifer data
        """
        data = {}
        for att in self.attributes:
            if att == 'SLICES':
                d = np.array([
                    (s.start, s.stop) for s in getattr(self, att)
                ])
            else:
                d = getattr(self, att)
            data[att] = d
        return data

    def crop_minimal_cube(self, slices):
        """
        Modify aquifer cell inndices with respect to
        slices representing grid crop.

        Parameters
        ----------
        slices : Iterable(slices)
            slices

        Returns
        -------
        Aquifer
            Aquifer with changed well indices.
        """
        new_slices = []
        for slice1, slice2 in zip(getattr(self, 'slices'), slices):
            new_start = max(slice1.start - slice2.start, 0)
            new_stop = min(slice1.stop-slice1.start, slice2.stop - slice2.start)
            new_stop = max(0, new_stop)
            new_slices.append(slice(new_start, new_stop))
        setattr(self, 'slices', new_slices)
        return self
