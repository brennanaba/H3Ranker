"""Set of functions used to calculate 4 geometrical measurements between pairs of residues
"""

import numpy as np

def cb_cb_dihedral(ca_coords, cb_coords, mask_value):
    """ Calculates the dihedral angle between the C-beta atoms.
    
    """
    mat_shape = (ca_coords.shape[1], ca_coords.shape[1], ca_coords.shape[2])

    b1 = np.tile(cb_coords - ca_coords, (mat_shape[0],1,1))
    b2 = cb_coords - cb_coords.transpose((1,0,2))
    b3 = -b1.transpose((1,0,2))

    np.seterr(divide='ignore', invalid='ignore')
    n1 = np.cross(b1, b2)
    n1 /= np.linalg.norm(n1, axis = 2, keepdims=True)
    n2 = np.cross(b2, b3)
    n2 /= np.linalg.norm(n2, axis = 2, keepdims=True)
    m1 = np.cross(b2 / np.linalg.norm(b2, axis = 2, keepdims=True), n1)

    cb_mask = np.array([[1 if sum(_) != 0 else 0 for _ in cb_coords[0]]])
    cb_mask = cb_mask * cb_mask.transpose()

    dihedral_mat = -(np.arctan2((m1 * n2).sum(-1), (n1 * n2).sum(-1)) * 180 / np.pi)
    dihedral_mat[cb_mask == 0] = mask_value
    return dihedral_mat

def cb_ca_dihedral(ca_coords, cb_coords, n_coords, mask_value):
    """ Calculates the dihedral angle between the C-beta and C-alpha atoms.
    
    """
    mat_shape = (ca_coords.shape[1], ca_coords.shape[1], ca_coords.shape[2])
    
    b1 = np.tile(ca_coords - n_coords, (mat_shape[0],1,1))
    b2 = np.tile(cb_coords - ca_coords, (mat_shape[0],1,1))
    b3 = cb_coords.transpose((1,0,2)) - cb_coords

    np.seterr(divide='ignore', invalid='ignore')
    n1 = np.cross(b1, b2)
    n1 /= np.linalg.norm(n1, axis = 2, keepdims=True)
    n2 = np.cross(b2, b3)
    n2 /= np.linalg.norm(n2, axis = 2, keepdims=True)
    m1 = np.cross(b2 / np.linalg.norm(b2, axis = 2, keepdims=True), n1)

    cb_mask = np.array([[1 if sum(_) != 0 else 0 for _ in cb_coords[0]]])
    cb_mask = cb_mask * cb_mask.transpose()

    dihedral_mat = (np.arctan2((m1 * n2).sum(-1), (n1 * n2).sum(-1)) * 180 / np.pi).transpose()
    dihedral_mat[cb_mask == 0] = mask_value
    return dihedral_mat
    
    
def ca_cb_cb_planar(ca_coords, cb_coords, mask_value):
    """ Calculates the planar angle between the C-beta and C-alpha atoms.
    
    """
    mat_shape = (ca_coords.shape[1], ca_coords.shape[1], ca_coords.shape[2])

    v1 = np.tile(ca_coords - cb_coords, (mat_shape[0],1,1))#(ca_coords - cb_coords).expand(mat_shape)
    v2 = cb_coords.transpose((1,0,2)) - cb_coords

    v1_norm = np.linalg.norm(v1, axis = 2)
    v2_norm = np.linalg.norm(v2, axis = 2)
    
    planar_mat = (v1 * v2).sum(-1) 
    planar_mat /= (v1_norm * v2_norm)
    planar_mat = np.arccos(planar_mat).transpose(0, 1)
    planar_mat *= 180 / np.pi

    cb_mask = np.array([[1 if sum(_) != 0 else 0 for _ in cb_coords[0]]])
    cb_mask = cb_mask * cb_mask.transpose()
    planar_mat[cb_mask == 0] = mask_value
    
    return planar_mat.transpose()

def geom_from_residues(residues, mask_value = -1e5):
    """ Calculates all 4 geometry measurements for a set of residues.
    
    scaling planar angle so it is in the range (-180,180). Change this back at some point.
    
    """
    def get_cb_or_ca(residue):
        if 'CB' in residue:
            return residue['CB'].get_coord()
        elif 'CA' in residue:
            return residue['CA'].get_coord()
        else:
            return [0, 0, 0]

    def get_ca(residue):
        if 'CA' in residue:
            return residue['CA'].get_coord()
        else:
            return [0, 0, 0]

    def get_cb(residue):
        if 'CB' in residue:
            return residue['CB'].get_coord()
        else:
            return [0, 0, 0]
    
    def get_n(residue):
        if 'N' in residue:
            return residue['N'].get_coord()
        else:
            return [0, 0, 0]

    cb_ca_coords = np.array([[get_cb_or_ca(r) for r in residues]])
    ca_coords = np.array([[get_ca(r) for r in residues]])
    cb_coords = np.array([[get_cb(r) for r in residues]])
    n_coords = np.array([[get_n(r) for r in residues]])
    

    dist_mat = np.sqrt(np.sum(np.subtract(cb_ca_coords, cb_ca_coords.transpose((1,0,2)))**2,axis = -1))
    cb_cb_dihedral_mat = cb_cb_dihedral(ca_coords, cb_coords, mask_value)
    cb_ca_dihedral_mat = cb_ca_dihedral(ca_coords, cb_coords, n_coords, mask_value)
    ca_cb_cb_planar_mat = ca_cb_cb_planar(ca_coords, cb_coords, mask_value)
    
    return dist_mat, cb_cb_dihedral_mat, cb_ca_dihedral_mat, ca_cb_cb_planar_mat

