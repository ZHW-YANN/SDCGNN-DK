import copy
import os
import time
import numpy as np
import networkx as nx
import pandas as pd
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from monty.io import zopen
from cavd.local_environment import CifParser_new, get_local_envir_fromstru
import csv
import string
from pymatgen.analysis.local_env import CrystalNN


class Void(object):
    """
    Class representing interstitial voids in crystal structures
    表示晶体结构中间隙空隙的类
    """

    def __init__(self):
        self.id = None
        self.label = None
        self.coord = None
        self.radii = None
        self.rec_radii = None
        self.env_fea = None
        self.occupy = False
        self.score = None
        self.energy = None


class Channel(object):
    """
    Class representing migration channels between voids
    表示空隙之间迁移通道的类
    """

    def __init__(self):
        self.start = None
        self.end = None
        self.phase = None
        self.coord = None
        self.radii = None
        self.rec_radii = None
        self.dist = None
        self.dist_radii = None
        self.label = None
        self.env_fea = None
        self.occupy = False
        self.score = None
        self.energy = None
        self.energy_difference = None


def atom_adjacency(file_cif, radius=8, max_num_nbr=12):
    """
    Build atom adjacency matrix based on cutoff distance
    基于截断距离构建原子邻接矩阵
    """
    adjacency_matrix = [[0] * 92 for i in range(92)]
    crystal = Structure.from_file(file_cif)
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    for i, nbr in enumerate(all_nbrs):
        atomic_number = crystal.atomic_numbers[i]
        for j, k in enumerate(nbr):
            if j == max_num_nbr: break
            link_atomic_number = crystal.atomic_numbers[k[2]]
            adjacency_matrix[atomic_number - 1][link_atomic_number - 1] += 1
    return adjacency_matrix


def atom_adjacency2(file_cif):
    """
    Calculate atom adjacency matrix based on CrystalNN
    基于CrystalNN计算原子邻接矩阵
    """
    from pymatgen.analysis.local_env import CrystalNN
    crystalNN = CrystalNN(weighted_cn=False, cation_anion=False, distance_cutoffs=(0.5, 1),
                          x_diff_weight=3, porous_adjustment=True, search_cutoff=8, fingerprint_length=None)
    structure = Structure.from_file(file_cif)
    adjacency_matrix2 = [[0] * 92 for i in range(92)]  # 90
    for index, atomic_number in enumerate(structure.atomic_numbers):
        cnn = crystalNN.get_nn_info(structure, index)
        for nbr in cnn:
            link_atomic_number = structure.atomic_numbers[nbr['site_index']]
            adjacency_matrix2[atomic_number - 1][link_atomic_number - 1] += 1
    return adjacency_matrix2


class MigrationNetwork(object):
    """
    Class for calculating transport networks and analyzing transport pathways
    用于计算传输网络和分析传输路径的类
    """

    def __init__(self, filename_cif, filename_cavd, moveion='Li', weight=None):
        """
        Initialize migration network analyzer
        初始化迁移网络分析器

        Args:
            filename_cif: CIF file path
            filename_cavd: CAVD file path
            moveion: Mobile ion species
            weight: Weight parameter (optional)
        """
        self._moveion = moveion  # Migration ion
        self._voids = {}  # All interstitial sites calculated by CAVD
        self._channels = {}  # All channel segments calculated by CAVD
        self._nonequalchannels = {}  # All non-equivalent paths starting from interstitials
        self._mignet = None  # Interstitial network graph (networkx)
        self._nonequl_paths = {}  # All non-equivalent paths starting from lattice sites
        self._voids_neighbor_atoms = {}
        self._channels_neighbor_atoms = {}
        self._filename_cif = filename_cif
        self._filename_cavd = filename_cavd
        self._struc = None
        self._struc_del_ion = None  # Structure without migration ions
        self._crystal = None
        self._crystal_del_ion = None  # Structure without migration ions
        self._rad_dict = None
        self._weight = weight
        self._void_label = None
        self.load_struc()  # Structure read by Pymatgen
        self.load_voids_channels_from_file()  # Read interstitial network structure
        self.init_voids_channels()
        self.del_ion()
        self.del_crystal_ion()

    def load_struc(self):
        """
        Read structure from CIF file
        从CIF文件读取结构
        """
        with zopen(self._filename_cif, "rt") as f:
            input_string = f.read()
        parser = CifParser_new.from_string(input_string)
        self._struc = parser.get_structures(primitive=False)[0]
        self._crystal = Structure.from_file(self._filename_cif)
        self._rad_dict = get_local_envir_fromstru(self._struc)[1]
        temp = {}
        for ele in self._crystal.symbol_set:
            temp[ele] = []
        for ele, rad in self._rad_dict.items():
            temp[ele.strip(string.digits)].append(rad)
        for ele, rad in temp.items():
            self._rad_dict[ele] = sum(rad) / len(rad)

    def load_voids_channels_from_file(self):
        """
        Read interstices and channel segments from net file calculated by cavd
        读取CAVD计算的net文件中的间隙点和通道段
        """
        voids_dict = {}
        channels_dict = {}
        flag_p = 0
        flag_n = 0
        file = open(self._filename_cavd, 'r')
        labels = set()
        for line in file.readlines():
            if 'Interstitial' in line:
                flag_p = 1
                flag_n = 0
                continue
            if 'Connection' in line:
                flag_p = 0
                flag_n = 1
                continue
            if flag_p == 1:
                line = line.split()
                if len(line) > 3:
                    void = Void()
                    void.id = int(line[0])
                    void.label = int(line[1])
                    void.coord = [np.float64(line[2]), np.float64(line[3]), np.float64(line[4])]
                    void.radii = np.float64(line[5])
                    void.rec_radii = 1 / np.float64(line[5])
                    voids_dict[void.id] = void
                    labels.add(void.label)
            if flag_n == 1:
                line = line.split()
                if len(line) > 4:
                    channel = Channel()
                    channel.start = int(line[0])
                    channel.end = int(line[1])
                    channel.phase = [int(line[2]), int(line[3]), int(line[4])]
                    channel.coord = [np.float64(line[5]), np.float64(line[6]), np.float64(line[7])]
                    channel.radii = np.float64(line[8])
                    channel.rec_radii = 1 / np.float64(line[8])
                    channel.dist_radii = np.float64(line[9]) / np.float64(line[8])
                    channel.dist = np.float64(line[9])  # ***
                    channels_dict[(channel.start, channel.end)] = channel
        self._voids = voids_dict
        self._channels = channels_dict
        self._void_label = sorted(list(labels))
        print(self._void_label)

    def init_voids_channels(self):
        """
        Initialize voids and channels with labels
        用标签初始化空隙和通道
        """
        i = 0
        for channel_id, channel in self._channels.items():  # Calculate label for each channel segment
            if self._voids[channel.start].label < self._voids[channel.end].label:
                key = (self._voids[channel.start].label, self._voids[channel.end].label,
                       round(channel.dist, 0))  # Start/end labels and length
            else:
                key = (self._voids[channel.end].label, self._voids[channel.start].label,
                       round(channel.dist, 0))
            if key not in self._nonequalchannels.keys():
                channel.label = i
                self._nonequalchannels[key] = {"label": i}  # Non-equivalent channel segments
                i += 1
            else:
                channel.label = self._nonequalchannels[key]["label"]

    def del_ion(self):
        """
        Import crystal structure and remove migration ions
        导入晶体结构并去除迁移离子
        """
        structure = copy.deepcopy(self._struc)
        del_idx = []
        for idx, site in enumerate(structure):
            if site.specie.element.value == self._moveion:
                del_idx.append(idx)
        for idx in del_idx[::-1]:
            del structure[idx]
        self._struc_del_ion = structure

    def del_crystal_ion(self):
        """
        Import crystal structure and remove migration ions
        导入晶体结构并去除迁移离子
        """
        structure = copy.deepcopy(self._crystal)
        del_idx = []
        for idx, site in enumerate(structure):
            if site.specie.element.value == self._moveion:
                del_idx.append(idx)
        for idx in del_idx[::-1]:
            del structure[idx]
        self._crystal_del_ion = structure

    def save_adjacency_matrix(self):
        """
        Save adjacency matrix of voids and channels
        保存空隙和通道的邻接矩阵
        """
        with open(self._filename_cif.split(".")[0] + '_adjacency_void_atoms1.txt', 'w') as f:
            for id, val in self._voids.items():
                if len(self._voids_neighbor_atoms[val.label][0]) == 0:
                    self._voids_neighbor_atoms[val.label][0] = [0]
                    self._voids_neighbor_atoms[val.label][1] = [0]
                f.write(str(id))
                f.write('\t')
                f.write(' '.join(map(str, val.coord)))
                f.write('\t')
                f.write(' '.join(map(str, self._voids_neighbor_atoms[val.label][0])))
                f.write('\t')
                f.write(' '.join(map(str, self._voids_neighbor_atoms[val.label][1])))
                f.write('\t')
                f.write(str(self._voids_neighbor_atoms[val.label][2]))
                f.write('\n')
        with open(self._filename_cif.split(".")[0] + '_adjacency_channel_atoms1.txt', 'w') as f:
            for id, val in self._channels.items():
                if len(self._channels_neighbor_atoms[val.label][0]) == 0:
                    self._channels_neighbor_atoms[val.label][0] = [0]
                    self._channels_neighbor_atoms[val.label][1] = [0]
                f.write(str(id))
                f.write('\t')
                f.write(' '.join(map(str, val.coord)))
                f.write('\t')
                f.write(' '.join(map(str, self._channels_neighbor_atoms[val.label][0])))
                f.write('\t')
                f.write(' '.join(map(str, self._channels_neighbor_atoms[val.label][1])))
                f.write('\t')
                f.write(str(self._channels_neighbor_atoms[val.label][2]))
                f.write('\n')

    def save_adjacency_matrix2(self):
        """
        Save simplified adjacency matrix based on symmetry
        保存基于对称性简化的邻接矩阵
        """
        with open(self._filename_cif.split(".")[0] + '_adjacency_void_atoms2.txt', 'w') as f:
            for id, label in enumerate(self._void_label):
                if len(self._voids_neighbor_atoms[label][0]) == 0:
                    self._voids_neighbor_atoms[label][0] = [0]
                    self._voids_neighbor_atoms[label][1] = [0]
                f.write(str(id))
                f.write('\t')
                f.write(' '.join(map(str, self._voids[label].coord)))
                f.write('\t')
                f.write(' '.join(map(str, self._voids_neighbor_atoms[label][0])))
                f.write('\t')
                f.write(' '.join(map(str, self._voids_neighbor_atoms[label][1])))
                f.write('\t')
                f.write(str(self._voids_neighbor_atoms[label][2]))
                f.write('\n')

        with open(self._filename_cif.split(".")[0] + '_adjacency_channel_atoms2.txt', 'w') as f:
            channels = []
            for id, val in self._channels.items():
                par = (self._void_label.index(self._voids[id[0]].label),
                       self._void_label.index(self._voids[id[1]].label))
                if par not in channels:
                    channels.append(par)
                    if len(self._channels_neighbor_atoms[val.label][0]) == 0:
                        self._channels_neighbor_atoms[val.label][0] = [0]
                        self._channels_neighbor_atoms[val.label][1] = [0]
                    f.write(str(par))
                    f.write('\t')
                    f.write(' '.join(map(str, val.coord)))
                    f.write('\t')
                    f.write(' '.join(map(str, self._channels_neighbor_atoms[val.label][0])))
                    f.write('\t')
                    f.write(' '.join(map(str, self._channels_neighbor_atoms[val.label][1])))
                    f.write('\t')
                    f.write(str(self._channels_neighbor_atoms[val.label][2]))
                    f.write('\n')

    def void_channal_neighbor_atoms(self):
        """
        Calculate neighboring atoms for voids and channels
        计算空隙和通道的邻近原子
        """
        for id, val in self._voids.items():
            if val.label not in self._voids_neighbor_atoms:
                self._voids_neighbor_atoms[val.label] = self.neighbor_atoms(val.coord, val.radii)
        for id, val in self._channels.items():
            if val.label not in self._channels_neighbor_atoms:
                self._channels_neighbor_atoms[val.label] = self.neighbor_atoms(val.coord, val.radii)
            if self._channels_neighbor_atoms[val.label][-1]:
                val.occupy = True

    def void_channal_neighbor_atoms2(self):
        """
        Calculate neighboring atoms for voids and channels (method 2)
        计算空隙和通道的邻近原子(方法2)
        """
        for id, val in self._voids.items():
            if val.label not in self._voids_neighbor_atoms:
                self._voids_neighbor_atoms[val.label] = self.neighbor_atoms2(val.coord, self._moveion + '2+', val.radii,
                                                                             id)
        for id, val in self._channels.items():
            if val.label not in self._channels_neighbor_atoms:
                self._channels_neighbor_atoms[val.label] = self.neighbor_atoms2(val.coord, self._moveion + '2+',
                                                                                val.radii, id)

    def neighbor_atoms(self, coord, radii, diff='diff'):
        """
        Build chemical environment representation based on cutoff distance
        基于截断距离构建化学环境表示
        """
        site = PeriodicSite('Ar', coord, self._struc_del_ion.lattice)
        nbrs = self._crystal_del_ion.get_neighbors(site, r=8.0, include_index=True)
        nbrs = sorted(nbrs, key=lambda x: x[1])
        n_atoms = [[], []]
        for atom in nbrs:
            if diff == 'diff':  # Void space determination
                try:
                    diff = atom.nn_distance - radii - self._rad_dict[atom.species.formula]
                except:
                    ele = atom.species.formula
                    diff = atom.nn_distance - radii - self._rad_dict[ele.strip(string.digits)]
                if diff < 1:  # Tolerance 1A
                    n_atoms[0].append(atom.specie.element.number)
                    n_atoms[1].append(round(atom.nn_distance, 6))
            else:
                n_atoms[0].append(atom.specie.element.number)
                n_atoms[1].append(round(atom.nn_distance, 6))
        if len(n_atoms[0]) > 12:
            n_atoms[0] = n_atoms[0][:12]
            n_atoms[1] = n_atoms[1][:12]
        n_atoms.append(radii)
        if len(n_atoms[0]) > 0 and n_atoms[0][0] == Element(self._moveion).Z and n_atoms[1][0] < 0.5:
            n_atoms.append(True)
        else:
            n_atoms.append(False)
        return n_atoms

    def neighbor_atoms2(self, coord, ion, radii, void_id):
        """
        Build chemical environment representation based on atomic coordination
        基于原子配位构建化学环境表示
        """
        self._struc_del_ion.append(ion, coord)
        # Calculate atom adjacency matrix based on CrystalNN
        crystalNN = CrystalNN(weighted_cn=False, cation_anion=False, distance_cutoffs=(0.5, 1),
                              x_diff_weight=3, porous_adjustment=True, search_cutoff=8, fingerprint_length=None)
        cnn = crystalNN.get_nn_info(self._struc_del_ion, -1)
        n_atoms = [[], []]
        for num, nbr in enumerate(cnn):
            if num == 12: break
            n_atoms[0].append(self._struc_del_ion.atomic_numbers[nbr['site_index']])
            n_atoms[1].append(self.get_dis(nbr['site'].frac_coords, coord))
        for i in range(len(n_atoms[1])):
            for j in range(i, len(n_atoms[1])):
                if n_atoms[1][i] > n_atoms[1][j]:
                    n_atoms[1][i], n_atoms[1][j] = n_atoms[1][j], n_atoms[1][i]
                    n_atoms[0][i], n_atoms[0][j] = n_atoms[0][j], n_atoms[0][i]
        del self._struc_del_ion[-1]
        n_atoms.append(radii)
        return n_atoms

    def get_dis(self, p1, p2):
        """
        Calculate distance between two points
        计算两点间距离
        """
        temp_site1 = PeriodicSite('Ar', p1, self._struc_del_ion.lattice)
        temp_site2 = PeriodicSite('Ar', p2, self._struc_del_ion.lattice)
        dis = temp_site1.distance(temp_site2)
        return round(dis, 6)

    def cal_save(self):
        """
        Calculate and save results
        计算并保存结果
        """
        self.void_channal_neighbor_atoms2()  # Calculate neighboring atoms of void bottlenecks
        self.save_adjacency_matrix2()  # Save symmetry-reduced void network adjacency matrix


if __name__ == '__main__':
    data = pd.read_csv('data/Zn/id_prop.csv').values
    for i in range(len(data)):
        cif_file = 'data/Zn/cif/' + data[i][0] + '.cif'
        orgin_net = 'data/Zn/origin_net/' + data[i][0] + '_origin.net'
        migration_ion = 'Zn'
        valence = 2
        MM = MigrationNetwork(cif_file, orgin_net, migration_ion)
        MM.cal_save()
