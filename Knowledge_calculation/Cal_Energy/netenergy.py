import ase
import ccnb.bvse.Structure as Structure
import numpy as np
import math
import os

import pandas as pd
from scipy.special import erfc


def loadbvparam(filename):
    """
    Load bond valence parameters from file
    从文件中加载键价参数
    """
    with open(filename, "r") as fp:
        lines = fp.readlines()
    bvmpara = {}
    for line in lines:
        varstr = line.split("\t")
        key = varstr[0] + varstr[1] + varstr[2] + varstr[3]
        if key in bvmpara:
            bvmpara[key].append([float(varstr[4]), float(varstr[5])])
        else:
            bvmpara[key] = [[float(varstr[4]), float(varstr[5])]]
    return bvmpara


def loadBVSEparam(filename):
    """
    Load BVSE parameters from file
    从文件中加载BVSE参数
    """
    with open(filename, "r") as fp:
        lines = fp.readlines()
    BVSEparam = {}
    for line in lines[1:]:
        line = line.strip("\n")
        varstr = line.split("\t")
        key = varstr[0] + varstr[1] + varstr[2] + varstr[3]
        if key in BVSEparam:
            BVSEparam[key].append(
                [
                    float(varstr[4]),
                    float(varstr[5]),
                    float(varstr[6]),
                    float(varstr[7]),
                    float(varstr[8]),
                    float(varstr[9]),
                ]
            )
        else:
            BVSEparam[key] = [
                [
                    float(varstr[4]),
                    float(varstr[5]),
                    float(varstr[6]),
                    float(varstr[7]),
                    float(varstr[8]),
                    float(varstr[9]),
                ]
            ]
    return BVSEparam


def loadElementsparam(filename):
    """
    Load element parameters from file
    从文件中加载元素参数
    """
    with open(filename, "r") as fp:
        lines = fp.readlines()
    Elementsparam = {}
    for line in lines:
        line = line.strip("\n")
        varstr = line.split("\t")
        key = varstr[1] + varstr[2]
        Elementsparam[key] = [
            float(varstr[0]),
            float(varstr[3]),
            float(varstr[4]),
            float(varstr[5]),
            float(varstr[6]),
            float(varstr[7]),
            float(varstr[8]),
            float(varstr[9]),
        ]
    return Elementsparam


class Vertex:
    """
    Vertex class representing a node in the migration network
    顶点类，表示迁移网络中的节点
    """

    def __init__(self, label=None, fraccoord=None, cartcoord=None, energy=None):
        self.label = label
        self.fraccoord = fraccoord
        self.cartcoord = cartcoord
        self.energy = energy


class Void(object):
    """
    Void class representing interstitial sites in the crystal structure
    空隙类，表示晶体结构中的间隙位点
    """

    def __init__(self):
        self.id = None
        self.coord = None
        self.chemical_environment = None
        self.atom_distance = None
        self.radii = None
        self.energy = None


class Channel(object):
    """
    Channel class representing migration pathways between voids
    通道类，表示空隙之间的迁移路径
    """

    def __init__(self):
        self.id = None
        self.start = None
        self.end = None
        self.coord = None
        self.chemical_environment = None
        self.atom_distance = None
        self.radii = None
        self.energy = None
        self.energy_difference = None


class Network_energy(object):
    """
    Network energy calculator for migration pathways
    迁移路径网络能量计算器
    """

    def __init__(self, cif_file, voids_file, channels_file, migrationion, valence, cif_name) -> None:
        """
        Initialize network energy calculator
        初始化网络能量计算器

        Args:
            cif_file: CIF file path
            voids_file: Voids data file path
            channels_file: Channels data file path
            migrationion: Migration ion species
            valence: Valence of migration ion
            cif_name: Name of CIF file
        """
        self._BVparam = loadbvparam("bvmparam.dat")
        self._BVSEparam = loadBVSEparam("bvse.dat")
        self._Elementsparam = loadElementsparam("elements.dat")
        self._stru = self.get_crystru(cif_file)
        self._migrationion = migrationion
        self._valence = valence
        self._qsum = self.get_qsum()
        self._cal_voids_channels_energy = self.cal_energy(voids_file, channels_file)
        self._save_voids_channels_energy = self.save_energy(cif_name)
        self._voids = None
        self._channels = None

    def cal_energy(self, voids_file, channels_file):
        """
        Calculate energy for voids and channels
        计算空隙和通道的能量
        """
        voids_dict = {}
        channels_dict = {}
        file1 = open(voids_file, 'r')
        file2 = open(channels_file, 'r')
        for line in file1.readlines():
            line = line.strip('\n')
            line = line.split('\t')
            void = Void()
            void.id = int(line[0])
            void.coord = line[1]
            void.chemical_environment = line[2]
            void.atom_distance = line[3]
            void.radii = line[4]
            cartcoor = self._stru.FracPosToCartPos(list(map(float, void.coord.split(' '))))
            void.energy = round(self.get_energy(cartcoor), 6)
            voids_dict[void.id] = void
        self._voids = voids_dict

        for line in file2.readlines():
            line = line.strip('\n')
            line = line.split('\t')
            channel = Channel()
            channel.id = line[0]
            channel.start = int(channel.id.strip('(').strip(')').split(',')[0])
            channel.end = int(channel.id.strip('(').strip(')').split(',')[1])
            channel.coord = line[1]
            channel.chemical_environment = line[2]
            channel.atom_distance = line[3]
            channel.radii = line[4]
            cartcoor = self._stru.FracPosToCartPos(list(map(float, channel.coord.split(' '))))
            channel.energy = round(self.get_energy(cartcoor), 6)
            channel.energy_difference = round(abs(voids_dict.get(channel.start).energy - channel.energy), 6)
            channels_dict[channel.id] = channel
        self._channels = channels_dict

    def save_energy(self, cif_name):
        """
        Save calculated energies to files
        将计算的能量保存到文件中
        """
        with open('new_data/Zn/' + cif_name + '_adjacency_void_atoms2.txt', 'w') as f:
            for id, val in self._voids.items():
                f.write(str(id))
                f.write('\t')
                f.write(val.coord)
                f.write('\t')
                f.write(val.chemical_environment)
                f.write('\t')
                f.write(val.atom_distance)
                f.write('\t')
                f.write(val.radii)
                f.write('\t')
                f.write(str(val.energy))
                f.write('\n')
            f.close()
        with open('new_data/Zn/' + cif_name + '_adjacency_channel_atoms2.txt', 'w') as f:
            for id, val in self._channels.items():
                f.write(id)
                f.write('\t')
                f.write(val.coord)
                f.write('\t')
                f.write(val.chemical_environment)
                f.write('\t')
                f.write(val.atom_distance)
                f.write('\t')
                f.write(val.radii)
                f.write('\t')
                f.write(str(val.energy))
                f.write('\t')
                f.write(str(val.energy_difference))
                f.write('\n')
            f.close()

    def get_crystru(self, ciffile):
        """
        Read crystal structure from CIF file and process it
        从CIF文件中读取晶体结构并进行处理
        """
        atoms = ase.io.read(ciffile, store_tags=True)
        struc = Structure.Structure()
        struc.GetAseStructure(atoms)
        return struc

    def get_qsum(self):
        """
        Calculate the coefficient of Coulomb interaction term in BVSE energy
        得到bvse能量值的第二项(库伦作用力)的系数
        """
        # atomsq dictionary stores charge density of each atom type
        atomsq = {}
        # _Elementsparam[key][3] stores principal quantum numbers of key
        # _Elementsparam[key][0] stores proton number of key
        # This loop calculates total charge of different elements in unit cell
        for atom in self._stru._atomsymbols:
            atomsq[atom] = 0
            for site in self._stru._Sites:
                if atom in site.GetElements():
                    key = atom + str(site.GetElementsOxiValue()[atom])
                    atomsq[atom] = atomsq[atom] + site.GetElementsOxiValue()[
                        atom
                    ] * site.GetElementsOccupy()[atom] / math.sqrt(
                        self._Elementsparam[key][3]
                    )
        # qsumanion stores total anion charge in unit cell
        qsumanion = 0
        # qsumcation stores total cation charge in unit cell
        qsumcation = 0
        for atom, value in atomsq.items():
            if value > 0:
                qsumcation = qsumcation + value
            elif value < 0:
                qsumanion = qsumanion + value
            else:
                qsumanion = 0
                qsumcation = 0
        # qsum stores charge balance coefficient
        qsum = 0.0
        if self._valence > 0:
            qsum = -qsumanion / qsumcation
        else:
            qsum = -qsumcation / qsumanion
        return qsum

    def get_energy(self, cartcoord):
        """
        Calculate BVSE value at given Cartesian coordinates
        得到一个实际坐标的bvse值
        """
        Rcutoff = 10.0
        key1 = self._migrationion + str(self._valence)
        # qm1 represents charge density of mobile ion, rm1 represents radius of mobile ion
        qm1 = self._valence / math.sqrt(self._Elementsparam[key1][3])
        rm1 = self._Elementsparam[key1][6]
        (distance, neighborsindex) = self._stru.GetKNeighbors(cartcoord, kn=128)
        data = 0.0
        cdata = 0.0
        for dindex, index in enumerate(neighborsindex):
            if distance[dindex] <= Rcutoff:
                site2 = self._stru._SuperCellusites[index]
                if site2.GetIronType() * self._valence < 0:  # Calculate attractive force between cations and anions
                    ssymbol = list(site2.GetElementsOccupy().keys())[0]
                    occupyvalue = list(site2.GetElementsOccupy().values())[0]
                    if ssymbol != "Vac" and ssymbol:
                        site2oxi = site2.GetElementsOxiValue()[ssymbol]
                        if self._valence > 0:
                            key = "".join(
                                [
                                    self._migrationion,
                                    str(self._valence),
                                    ssymbol,
                                    str(site2oxi),
                                ]
                            )
                        else:
                            key = "".join(
                                [
                                    ssymbol,
                                    str(site2oxi),
                                    self._migrationion,
                                    str(self._valence),
                                ]
                            )
                        if key in self._BVSEparam:
                            (
                                Nc,
                                r0,
                                Rcut,
                                D0,
                                Rmin,
                                alpha,
                            ) = self._BVSEparam[
                                key
                            ][0]
                            smin = np.exp((Rmin - distance[dindex]) * alpha)
                            data = data + D0 * ((smin - 1) ** 2 - 1) * occupyvalue
                        else:
                            if key not in self._BVSEparam:
                                raise Exception(
                                    "bvse {0}  param can't find!!!!".format(key)
                                )
                            else:
                                raise Exception(
                                    "bvs {0} param can't find!!!!".format(key)
                                )
                else:  # Coulomb repulsive force between mobile ion M and framework ion Mi
                    ssymbol = list(site2.GetElementsOccupy().keys())[0]
                    occupyvalue = list(site2.GetElementsOccupy().values())[0]
                    if ssymbol != "Vac" and ssymbol != self._migrationion:
                        site2oxi = site2.GetElementsOxiValue()[ssymbol]
                        key = ssymbol + str(site2oxi)
                        rm2 = self._Elementsparam[key][6]
                        qm2 = site2oxi / math.sqrt(self._Elementsparam[key][3])
                        rm1m2 = distance[dindex]
                        f = 0.74
                        if rm1m2 > rm2:
                            cdata = (
                                    cdata
                                    + occupyvalue
                                    * qm1
                                    * qm2
                                    / rm1m2
                                    * erfc(rm1m2 / (f * (rm1 + rm2)))
                                    * self._qsum
                            )
                        else:
                            cdata = 300
        cartcoord_bvse = 0.5 * (data) + 14.4 * cdata
        return cartcoord_bvse


if __name__ == "__main__":
    data = pd.read_csv('data/Zn/id_prop.csv').values
    migration_ion = 'Zn'
    valence = 2
    for i in range(len(data)):
        cif_name = data[i][0]
        cif_path = 'data/' + migration_ion + '/cif/' + cif_name + '.cif'
        voids_path = 'data/' + migration_ion + '/cif/' + cif_name + '_adjacency_void_atoms2.txt'
        channels_path = 'data/' + migration_ion + '/cif/' + cif_name + '_adjacency_channel_atoms2.txt'
        network = Network_energy(cif_path, voids_path, channels_path, migration_ion, valence, cif_name)
        print("ss")
