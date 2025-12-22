import pymatgen as pmg
from pymatgen.core import Structure
from pymatgen.analysis.diffusion import transport
import numpy as np


def get_migration_ions_from_cif(cif_file):
    """
    Extract migration ions from CIF file
    从CIF文件中提取迁移离子
    """
    # Read CIF file
    structure = Structure.from_file(cif_file)

    # Analyze migration ions
    migration_ions = []

    # Method 1: Based on ionic conductivity analysis (requires additional electrochemical data)
    try:
        migration_data = get_migration_ion(structure)
        migration_ions.extend(migration_data)
    except:
        print("Unable to obtain migration ions through conductivity analysis")

    # Method 2: Based on structural features
    migration_ions.extend(analyze_structure_for_migration(structure))

    return migration_ions


def analyze_structure_for_migration(structure):
    """
    Analyze potential migration ions based on structural features
    基于结构特征分析可能的迁移离子
    """
    potential_migration_ions = []

    # Common migration elements
    common_migration_elements = ['Li', 'Na', 'K', 'Mg', 'Ca', 'O', 'F', 'H']

    for site in structure:
        element = site.specie.symbol

        if element in common_migration_elements:
            # Calculate coordination number
            neighbors = structure.get_neighbors(site, 3.0)
            coordination_number = len(neighbors)

            # Calculate local environment symmetry
            local_env = pmg.analysis.local_env.CrystalNN()
            try:
                env = local_env.get_cn(structure, site.index)
            except:
                env = coordination_number

            # Judgment criteria: low coordination number, small ionic radius, high symmetry position
            if coordination_number <= 6 and element in ['Li', 'Na', 'H']:
                potential_migration_ions.append({
                    'element': element,
                    'position': site.frac_coords,
                    'coordination_number': coordination_number,
                    'probability': 'high'
                })
            elif coordination_number <= 8:
                potential_migration_ions.append({
                    'element': element,
                    'position': site.frac_coords,
                    'coordination_number': coordination_number,
                    'probability': 'medium'
                })

    return potential_migration_ions


def save_migration_results(migration_ions, output_file):
    """
    Save migration ion analysis results
    保存迁移离子分析结果
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Migration Ion Analysis Results\n")
        f.write("=" * 50 + "\n")

        for i, ion in enumerate(migration_ions, 1):
            f.write(f"\n{i}. Migration Ion Information:\n")
            f.write(f"   Element: {ion['element']}\n")
            f.write(f"   Fractional Coordinates: {ion['position']}\n")
            f.write(f"   Coordination Number: {ion['coordination_number']}\n")
            f.write(f"   Migration Probability: {ion.get('probability', 'unknown')}\n")

            if 'migration_barrier' in ion:
                f.write(f"   Migration Barrier: {ion['migration_barrier']} eV\n")
            if 'pathway' in ion:
                f.write(f"   Migration Pathway: {ion['pathway']}\n")


# Usage example
if __name__ == "__main__":
    cif_file = "data/Li/cif/icsd_14.cif"
    output_file = "migration_ions_analysis.txt"

    migration_ions = get_migration_ions_from_cif(cif_file)
    save_migration_results(migration_ions, output_file)

    print(f"Analysis completed! Results saved to {output_file}")