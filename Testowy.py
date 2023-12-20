from Bio import PDB
import numpy as np
import re

parser = PDB.PDBParser(QUIET=True)

# Atomic radii for various atom types. 
# You can comment out the ones you don't care about or add new ones
atom_radii = {
    'H': 1.2,
    'C': 1.7, 
    'N': 1.55, 
    'O': 1.52,
    'S': 1.8,
    'F': 1.47, 
    'P': 1.8, 
    'CL': 1.75, 
    'MG': 1.73
}

def count_atoms(structure):
    num_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    num_atoms.append(atom)
    return len(num_atoms)

def count_clashes(structure, clash_cutoff):
    # Set what we count as a clash for each pair of atoms
    clash_cutoffs = {i + "_" + j: (atom_radii[i] + atom_radii[j] - clash_cutoff) for i in atom_radii for j in atom_radii}

    # Extract atoms for which we have a radii
    atoms = [x for x in structure.get_atoms() if x.element in atom_radii]
    coords = np.array([a.coord for a in atoms], dtype="d")

    # Build a KDTree (speedy!!!)
    kdt = PDB.kdtrees.KDTree(coords)

    # Initialize a list to hold clashes
    clashes = []

    # Iterate through all atoms
    for atom_1 in atoms:
        # Find atoms that could be clashing
        kdt_search = kdt.search(np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values()))

        # Get index and distance of potential clashes
        potential_clash = [(a.index, a.radius) for a in kdt_search]

        for ix, distance in potential_clash:

            atom_2 = atoms[ix]

            # Exclude clashes from atoms in the same residue
            if atom_1.parent.id == atom_2.parent.id:
                continue

            if((atom_1.parent.id[1]) == (atom_2.parent.id[1]+1)):
                continue

            if((atom_1.parent.id[1]) == (atom_2.parent.id[1]-1)):
                continue

        
            x1,y1,z1 = atom_1.get_coord()
            x2,y2,z2 = atom_2.get_coord()

            distance = ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**0.5

            overlap = (atom_radii[atom_1.element] + atom_radii[atom_2.element] - distance)

            if overlap >= 0.4:
                clashes.append((atom_1, atom_2))

    return len(clashes)//2

file = "R1107_reference.pdb"
structure = parser.get_structure(file, file)
num_clashes = count_clashes(structure, 0.4)
num_atoms = count_atoms(structure)
print("Number of clashes: ", num_clashes)
print("Clash_score: ", 1000*num_clashes/num_atoms)