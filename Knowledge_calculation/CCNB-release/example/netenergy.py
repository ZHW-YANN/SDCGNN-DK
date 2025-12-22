import ase
import ccnb.bvse.Structure as Structure
import numpy as np
import math
import os
from scipy.special import erfc


def loadbvparam(filename):
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
    def __init__(self, label=None, fraccoord=None, cartcoord=None, energy=None):
        self.label = label
        self.fraccoord = fraccoord
        self.cartcoord = cartcoord
        self.energy = energy


class Network_energy(object):
    def __init__(self, ciffile, migrationion, valence, coord_dict) -> None:
        """
        ciffile:cif文件
        migrationion: 迁移离子
        valence: 迁移离子的化合阶
        coord_dict: 标签和分数坐标
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self._BVparam = loadbvparam(os.path.join(module_dir, "bvmparam.dat"))
        self._BVSEparam = loadBVSEparam(os.path.join(module_dir, "bvse.dat"))
        self._Elementsparam = loadElementsparam(
            os.path.join(module_dir, "elements.dat")
        )
        self._stru = self.get_crystru(ciffile)
        self._migrationion = migrationion
        self._valence = valence
        self._qsum = self.get_qsum()
        self._vertexs = self.read_coord(coord_dict)
        pass

    def get_crystru(self, ciffile):
        """
        从cif文件中读取晶体结构并进行括胞
        """
        atoms = ase.io.read(ciffile, store_tags=True)
        struc = Structure.Structure()
        struc.GetAseStructure(atoms)
        return struc

    def read_coord(self, coord_dict):
        vertex_list = []
        for node_dict in coord_dict.items():
            cartcoor = self._stru.FracPosToCartPos(node_dict[1])
            vertex = Vertex(
                label=node_dict[0], fraccoord=node_dict[1], cartcoord=cartcoor
            )
            vertex_list.append(vertex)
        return vertex_list

    def get_qsum(self):
        """
        得到bvse能量值的第二项(库伦作用力)的系数
        """
        # atomsq字典中存储了每种原子的电荷密度
        atomsq = {}
        # _Elementsparam[key][3]中存储key的主量子数（principal quantum numbers）
        # _Elementsparam[key][0]中存储key的质子数
        # for循环用于计算晶胞中不同元素的总电荷量
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
        # qsumanion用于保存晶胞中所有阴离子的电荷数
        qsumanion = 0
        # qsumcation用于保存晶胞中所有阳离子的电荷数
        qsumcation = 0
        for atom, value in atomsq.items():
            if value > 0:
                qsumcation = qsumcation + value
            elif value < 0:
                qsumanion = qsumanion + value
            else:
                qsumanion = 0
                qsumcation = 0
        # qsum保存电荷平衡系数
        qsum = 0.0
        if self._valence > 0:
            qsum = -qsumanion / qsumcation
        else:
            qsum = -qsumcation / qsumanion
        return qsum

    def get_energy(self, cartcoord):
        """
        得到一个实际坐标的bvse值
        """
        Rcutoff = 10.0
        key1 = self._migrationion + str(self._valence)
        # qm1表示移动离子的电荷密度，rm1表示移动离子的半径
        qm1 = self._valence / math.sqrt(self._Elementsparam[key1][3])
        rm1 = self._Elementsparam[key1][6]
        (distance, neighborsindex) = self._stru.GetKNeighbors(cartcoord, kn=128)
        data = 0.0
        cdata = 0.0
        for dindex, index in enumerate(neighborsindex):
            if distance[dindex] <= Rcutoff:
                site2 = self._stru._SuperCellusites[index]
                if site2.GetIronType() * self._valence < 0:  # 计算阳离子和阴离子相互吸引作用力
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
                else:  # 移动离子M和骨架离子Mi相互排斥作用的库仑力
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

    def calubvse(self):
        """
        获得所有顶点的能量值
        """
        for vertex in self._vertexs:
            vertex_energy = self.get_energy(vertex.cartcoord)
            vertex.energy = vertex_energy


if __name__ == "__main__":
    coord_dict = {1: [0.38798, 0.0901351, 0.330103], 2: [0.231451, 0.130307, 0.262256]}
    network = Network_energy(r"C:\Users\33389\Desktop\icsd_58.cif", "Li", 1, coord_dict)
    network.calubvse()
    print("ss")
