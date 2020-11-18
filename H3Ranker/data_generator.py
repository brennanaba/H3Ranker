import numpy as np
from H3Ranker.geometries import geom_from_residues
from ABDB import database as db

db.set_numbering_scheme("chothia")

import os
current_directory = os.path.dirname(os.path.realpath(__file__))

def generate_data(pdb, i, resol):
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
        
    residues = [r for r in chain.get_residues() if r.get_id()[0] == " " and 92 < r.get_id()[1] < 106]
    seq = fab.get_numbering()["H"]
    loopseq = "".join([seq[x] for x in seq if 92 < x[0] < 106])
    assert len(loopseq) > 1

    dist_mat, cb_cb_dihedral_mat, cb_ca_dihedral_mat, ca_cb_cb_planar_mat = geom_from_residues(residues)

    output_matrix = np.stack([dist_mat, cb_cb_dihedral_mat, cb_ca_dihedral_mat, ca_cb_cb_planar_mat])
    
    np.save(os.path.join(os.path.join(current_directory,,"../../data"), pdb + heavy_chain), output_matrix)

    with open(os.path.join(current_directory,"data.csv"), "a+") as file:
        file.write(pdb + heavy_chain + "," + loopseq + "," + resol + "\n")
        
if __name__ == "__main__":
    # If run as main calculates H3 geometries for all antibodies in SABDAB.

    # Start by trying to update the AB database
    try:
        from ABDB.ABDB_updater import update
        update([])
    # If you do not even have a resolution what are you doing here?
    pdbs = [x for x in db.db_summary if db.db_summary[x]["resolution"].replace('.','',1).isnumeric()]
    pdbs = [x for x in pdbs if float(db.db_summary[x]["resolution"]) < 3]
        
    with open(os.path.join(current_directory,"data.csv"), "w+") as file:
        file.write("ID,Sequence,Resolution\n")
    
    for pdb in pdbs:
        struc = db.fetch(pdb)
        fabs = struc.fabs
        resolution = struc.get_resolution()
        for i in range(len(fabs)):
            try:
                generate_data(pdb, i, resolution)
            except Exception:
                print(pdb)


    
