import numpy as np
from H3Ranker.geometries import geom_from_residues
from ABDB import database as db

db.set_numbering_scheme("chothia")

import os
current_directory = os.path.dirname(os.path.realpath(__file__))

def generate_data(pdb,i,resol):
    fab = db.fetch(pdb).fabs[i]
    heavy_chain = fab.get_VH()
    light_chain = fab.get_VL()
    try:
        Hchain = [ch for ch in fab.get_structure().get_chains() if ch.get_id() == heavy_chain][0]
        Lchain = [ch for ch in fab.get_structure().get_chains() if ch.get_id() == light_chain][0]
    except AttributeError:
        print(pdb + heavy_chain)
    except KeyError:
        print(pdb + heavy_chain)
    
    Hresidues = [r for r in Hchain.get_residues() if r.get_id()[0] == " " and r.get_id()[1] < 114]
    Lresidues = [r for r in Lchain.get_residues() if r.get_id()[0] == " " and r.get_id()[1] < 110]
    residues = Hresidues + [[None]] + Lresidues


    Hseq = fab.get_numbering()["H"]
    Lseq = fab.get_numbering()["L"]
    fv_seq = "".join([Hseq[x] for x in Hseq if x[0] < 114]) + "-" + "".join([Lseq[x] for x in Lseq if x[0] < 109])
    assert len(fv_seq) > 1
    
    dist_mat, cb_cb_dihedral_mat, cb_ca_dihedral_mat, ca_cb_cb_planar_mat = geom_from_residues(residues)

    output_matrix = np.stack([dist_mat, cb_cb_dihedral_mat, cb_ca_dihedral_mat, ca_cb_cb_planar_mat])
    
    np.save(os.path.join(os.path.join(current_directory,"data"), pdb + heavy_chain), output_matrix)

    with open(os.path.join(current_directory,"data.csv"), "a+") as file:
        file.write(pdb + heavy_chain + "," + fv_seq + "," + resol + "\n")
        
if __name__ == "__main__":
    # If run as main calculates H3 geometries for all antibodies in SABDAB.

    # If you do not even have a resolution what are you doing here?
    pdbs = [x for x in db.db_summary if db.db_summary[x]["resolution"].replace('.','',1).isnumeric()]
    pdbs = [x for x in pdbs if float(db.db_summary[x]["resolution"]) < 3]
        
    with open(os.path.join(current_directory,"data.csv"), "w+") as file:
        file.write("ID,Sequence,Resolution\n")
    
    for pdb in pdbs:
        fabs = db.fetch(pdb).fabs
        for i in range(len(fabs)):
            try:
                generate_data(pdb, i)
            except Exception:
                print(pdb)


    
