import tempfile

import torch
import os
import csv
import functools
import json
import os
import random
import sys
import warnings
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from jarvis.core.atoms import Atoms
import numpy as np
from pymatgen.core import Structure
if not sys.warnoptions:
    warnings.simplefilter("ignore")


class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData:
    def __init__(self, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self.max_num_nbr, self.radius = max_num_nbr, radius
        atom_init_file = 'atom_init.json'
        assert os.path.exists(atom_init_file), 'atom_init.json'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def get_graph(self, cif_path):
        # 路径这里需要修改
        crystal = Structure.from_file(cif_path)
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        a = list(map(lambda x: np.repeat(x, self.max_num_nbr), np.arange(len(atom_fea))))
        a = np.concatenate(a).reshape(1, -1)
        a = torch.Tensor(a)
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        nbr_fea = np.reshape(nbr_fea, (-1, 41))
        nbr_fea_idx = np.reshape(nbr_fea_idx, (1, -1))
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        cs_edge_index = torch.cat((a, nbr_fea_idx), dim=0)
        return atom_fea, nbr_fea, cs_edge_index




model = torch.load('save_model/CGCNN{1}.pth')
model.to('cpu')
model.eval()
from_string = """
data_Li1Ta2P1O8
_symmetry_space_group_name_H-M P-3
_cell_length_a 4.9268
_cell_length_b 4.9268
_cell_length_c 8.5022
_cell_angle_alpha 90.0000
_cell_angle_beta 90.0000
_cell_angle_gamma 120.0000
_symmetry_Int_Tables_number 147
_chemical_formula_structural LiTa2PO8
_chemical_formula_sum 'Li1 Ta2 P1 O8'
_cell_volume 178.2201
_cell_formula_units_Z 1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  '-x, -y, -z'
  2  'y, -x+y, -z'
  3  '-y, x-y, z'
  4  '-x+y, -x, z'
  5  'x, y, z'
  6  'x-y, x, -z'
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_symmetry_multiplicity
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Li Li0 1 0.0000 0.0000 0.5000 1
Ta Ta1 2 0.3333 0.6667 0.2620 1
P P2 1 0.0000 0.0000 0.0000 1
O O3 6 0.0299 0.6777 0.1351 1
O O4 2 0.3333 0.6667 0.4772 1
"""
with open('example.cif', 'w') as file:
    file.write(from_string)

cs_x, cs_edge_attr, cs_edge_index = CIFData().get_graph('example.cif')
cs_edge_source = cs_edge_index[0].to(torch.long)
cs_edge_target = cs_edge_index[1].to(torch.long)
print(cs_edge_source)
cs_node_batch = torch.LongTensor([0] * len(cs_x))
out_data = model(cs_x, cs_edge_source, cs_edge_target, cs_edge_attr, cs_node_batch).detach().numpy().flatten().tolist()
print(out_data)
reply = f"{out_data[0]}"
print(reply)
# request = 'data_Li1Ta2P1O8\n_symmetry_space_group_name_H-M P-3\n_cell_length_a 4.9268\n_cell_length_b 4.9268\n_cell_length_c 8.5022\n_cell_angle_alpha 90.0000\n_cell_angle_beta 90.0000\n_cell_angle_gamma 120.0000\n_symmetry_Int_Tables_number 147\n_chemical_formula_structural LiTa2PO8\n_chemical_formula_sum Li1 Ta2 P1 O8\n_cell_volume 178.2201\n_cell_formula_units_Z 1\nloop_\n_symmetry_equiv_pos_site_id\n_symmetry_equiv_pos_as_xyz\nloop_\n_atom_site_type_symbo\n_atom_site_label\n_atom_site_symmetry_multiplicity\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\nLi Li0 1 0.0000 0.0000 0.5000 1\nTa Ta1 2 0.3333 0.6667 0.2620 1\nP P2 1 0.0000 0.0000 0.0000 1\nO O3 6 0.0299 0.6777 0.1351 1\nO O4 2 0.3333 0.6667 0.4772 1'
# data = Structure.from_file('data/cif/icsd_14.cif')
# print(data)
# data1 = Atoms.from_cif(from_string=request)
# print(data1)new_file, filename = tempfile.mkstemp(text=True)
#                     f = open(filename, "w")
#                     f.write(from_string)
#                     f.close()
# from_string = '123'
# new_file, filename = tempfile.mkstemp(text=True)
# print(filename)
# print(new_file)
# f = open(filename, "w")
# f.write(from_string)
# f.close()



