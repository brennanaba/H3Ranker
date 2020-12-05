import numpy as np
from H3Ranker.geometries import geom_from_residues
from ABDB import database as db
import os

db.set_numbering_scheme("chothia")

current_directory = os.path.dirname(os.path.realpath(__file__))


def aligned_heavy(heavy_numbers, heavy_residues):
    """ Tries to align heavy chain sequence and residues to the template used for the network.
    If it can't it raises an Assertion Error saying why.
    """
    seq = ""
    res = []
    res_dict = {x.id: x for x in heavy_residues}
    assert (31, "C") not in heavy_numbers, "H1 Loop too long to be aligned"
    assert (52, "D") not in heavy_numbers, "H2 Loop too long to be aligned"
    assert (82, "D") not in heavy_numbers, "Too many residues at position 82"
    assert (100, "M") not in heavy_numbers, "H3 Loop too long to be aligned"
    assert not any([(x, "A") in heavy_numbers and x not in [31, 52, 82, 100] for x in
                    range(114)]), "Insertion on unexpected residue"

    def get_or_fill_seq(key):
        if key in heavy_numbers:
            return heavy_numbers[key]
        else:
            return "-"

    def get_or_fill_res(key):
        key = (" ",) + key
        if key in res_dict:
            return res_dict[key]
        else:
            return [None]

    for i in range(1, 114):
        base = (i, " ")
        seq += get_or_fill_seq(base)
        res.append(get_or_fill_res(base))
        if i == 31:
            for let in ["A", "B"]:
                base = (i, let)
                seq += get_or_fill_seq(base)
                res.append(get_or_fill_res(base))
        elif i == 52:
            for let in ["A", "B", "C"]:
                base = (i, let)
                seq += get_or_fill_seq(base)
                res.append(get_or_fill_res(base))
        elif i == 82:
            for let in ["A", "B", "C"]:
                base = (i, let)
                seq += get_or_fill_seq(base)
                res.append(get_or_fill_res(base))
        elif i == 100:
            for let in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]:
                base = (i, let)
                seq += get_or_fill_seq(base)
                res.append(get_or_fill_res(base))
    return seq, res


def aligned_light(heavy_numbers, heavy_residues):
    """ Tries to align light chain sequence and residues to the template used for the network.
    If it can't it raises an Assertion Error saying why.
    """
    seq = ""
    res = []
    res_dict = {x.id: x for x in heavy_residues}
    assert (30, "G") not in heavy_numbers, "L1 Loop too long to be aligned"
    assert (95, "D") not in heavy_numbers, "L3 Loop too long to be aligned"
    assert (106, "M") not in heavy_numbers, "Too many residues at position 106"
    assert not any(
        [(x, "A") in heavy_numbers and x not in [30, 95, 106] for x in range(110)]), "Insertion on unexpected residue"

    def get_or_fill_seq(key):
        if key in heavy_numbers:
            return heavy_numbers[key]
        else:
            return "-"

    def get_or_fill_res(key):
        key = (" ",) + key
        if key in res_dict:
            return res_dict[key]
        else:
            return [None]

    for i in range(1, 110):
        base = (i, " ")
        seq += get_or_fill_seq(base)
        res.append(get_or_fill_res(base))
        if i == 30:
            for let in ["A", "B", "C", "D", "E", "F"]:
                base = (i, let)
                seq += get_or_fill_seq(base)
                res.append(get_or_fill_res(base))
        elif i == 95:
            for let in ["A", "B", "C", "D", "E", "F"]:
                base = (i, let)
                seq += get_or_fill_seq(base)
                res.append(get_or_fill_res(base))
        elif i == 106:
            for let in ["A"]:
                base = (i, let)
                seq += get_or_fill_seq(base)
                res.append(get_or_fill_res(base))
    return seq, res


def generate_data(pdb, i, resol):
    """ Generates geometry matrices for the ith Fab in pdb.
    Data is generated after being aligned to the networks template.
    The name of the file where it is saved is appended to a csv file.
    """
    fab = db.fetch(pdb).fabs[i]
    heavy_chain = db.db_summary[pdb]["fabs"][i]["Hchain"]
    light_chain = db.db_summary[pdb]["fabs"][i]["Lchain"]

    Hchain = [ch for ch in fab.get_structure().get_chains() if ch.get_id() == heavy_chain][0]
    Lchain = [ch for ch in fab.get_structure().get_chains() if ch.get_id() == light_chain][0]

    Hresidues = [r for r in Hchain.get_residues() if r.get_id()[0] == " " and r.get_id()[1] < 114]
    Lresidues = [r for r in Lchain.get_residues() if r.get_id()[0] == " " and r.get_id()[1] < 110]
    Hnumb = fab.get_numbering()["H"]
    Lnumb = fab.get_numbering()["L"]

    Hseq, Hresidues = aligned_heavy(Hnumb, Hresidues)
    Lseq, Lresidues = aligned_light(Lnumb, Lresidues)

    residues = Hresidues + [[None]] + Lresidues
    fv_seq = Hseq + "-" + Lseq
    assert len(fv_seq) > 1

    dist_mat, cb_cb_dihedral_mat, cb_ca_dihedral_mat, ca_cb_cb_planar_mat = geom_from_residues(residues)

    output_matrix = np.stack([dist_mat, cb_cb_dihedral_mat, cb_ca_dihedral_mat, ca_cb_cb_planar_mat])

    np.save(os.path.join(os.path.join(current_directory, "../../data"), pdb + heavy_chain), output_matrix)

    with open(os.path.join(current_directory, "data.csv"), "a+") as file:
        file.write(pdb + heavy_chain + "," + fv_seq + "," + resol + "\n")


if __name__ == "__main__":
    # If run as main calculates H3 geometries for all antibodies in SABDAB.
    try:
        from ABDB.ABDB_updater import update
        # update([])
    except Exception:
        print("Update Failed")

    # If you do not even have a resolution what are you doing here?
    pdbs = [x for x in db.db_summary if db.db_summary[x]["resolution"].replace('.', '', 1).isnumeric()]
    # If your resolution is shit you can leave now
    pdbs = [x for x in pdbs if float(db.db_summary[x]["resolution"]) < 3]

    # Initialize csv file
    with open(os.path.join(current_directory, "data.csv"), "w+") as file:
        file.write("ID,Sequence,Resolution\n")

    for pdb in pdbs:
        p = db.fetch(pdb)
        for i in range(len(p.fabs)):
            try:
                generate_data(pdb, i, p.get_resolution())
            except Exception as e:
                # If it broke print its name.
                print(pdb)
