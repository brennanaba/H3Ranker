import numpy as np
from geometries import geom_from_residues
from ABDB import database as db

db.set_numbering_scheme("chothia")

def generate_data(pdb, i):
    """ Calculates H3 geometries from SABDAB entries and stores them into a file
    
    """
    fab = db.fetch(pdb).fabs[i]
    heavy_chain = fab.get_VH()
    try:
        chain = [ch for ch in fab.get_structure().get_chains() if ch.get_id() == heavy_chain][0]
    except AttributeError:
        struc = fab.get_structure()
        if struc.get_id() == heavy_chain:
            chain = struc
        else:
            print(pdb + heavy_chain)
            return 0
    except KeyError:
        print(pdb + heavy_chain)
        return 0
        
    residues = [r for r in chain.get_residues() if r.get_id()[0] == " " and 91 < r.get_id()[1] < 105]
    seq = fab.get_numbering()["H"]
    loopseq = "".join([seq[x] for x in seq if 91 < x[0] < 105])

    dist_mat, cb_cb_dihedral_mat, cb_ca_dihedral_mat, ca_cb_cb_planar_mat = geom_from_residues(residues, mask_value = 1e5)

    output_matrix = np.stack([dist_mat, cb_cb_dihedral_mat, cb_ca_dihedral_mat, ca_cb_cb_planar_mat])
    
    np.save("data/" + pdb + heavy_chain, output_matrix.numpy())

    with open("data.csv", "a+") as file:
        file.write(pdb + heavy_chain + "," + loopseq + "\n")
        
if __name__ == "__main__":
    # If run as main calculates H3 geometries for all antibodies in SABDAB.

    # If you do not even have a resolution what are you doing here?
    pdbs = [x for x in db.db_summary if db.db_summary[x]["resolution"].replace('.','',1).isnumeric()]
    pdbs = [x for x in pdbs if float(db.db_summary[x]["resolution"]) < 3]
        
    with open("data.csv", "w+") as file:
        file.write("ID,Sequence\n")
    
    for pdb in pdbs:
        fabs = db.fetch(pdb).fabs
        for i in range(len(fabs)):
            try:
                generate_data(pdb, i)
            except Exception:
                print(pdb)


    