from pymatgen.io.cif import CifParser
import pandas as pd
import os

def readFormulaFromCif(file_path):
    parser = CifParser(file_path)
    structures = parser.get_structures()
    for structure in structures:
        real_formula = structure.composition.reduced_formula
    return real_formula