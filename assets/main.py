import numpy as np
import pickle
import scipy.sparse as sp
from scipy.io import savemat
from scipy.io import loadmat

def row(input):
    output = input.reshape(1, -1)
    return output

def col(input):
    output = input.reshape(-1, 1)
    return output

def get_hres_smpl_model_data(smpl_file):

    dd = pickle.load(open(smpl_file, 'rb'), encoding='latin1')
    backwards_compatibility_replacements(dd)

    hv, hf, mapping = get_hres(dd['v_template'], dd['f'])

    num_betas = dd['shapedirs'].shape[-1]
    J_reg = dd['J_regressor'].asformat('csr')

    model = {
        'v_template': hv,
        'weights': np.hstack([np.expand_dims(mapping.dot(dd['weights'][:, i]), axis=-1)
            for i in range(24)]),
        'posedirs': mapping.dot(dd['posedirs'].reshape((-1, 207 * 3))).reshape(-1, 3, 207),
        'shapedirs': mapping.dot(dd['shapedirs'].reshape((-1, 3 * num_betas))).reshape(-1, 3, num_betas),
        'J_regressor': sp.csr_matrix((J_reg.data, J_reg.indices, J_reg.indptr), shape=(24, hv.shape[0])),
        'kintree_table': dd['kintree_table'],
        'bs_type': dd['bs_type'],
        'bs_style': dd['bs_style'],
        'J': dd['J'],
        'f': hf,
        'mapping': mapping
    }

    return model

def get_lres_smpl_model_data(smpl_file):

    dd = pickle.load(open(smpl_file, 'rb'), encoding='latin1')
    backwards_compatibility_replacements(dd)

    with open('smpl_sampling_hi.pkl', 'rb') as f:
        sampling = pickle.load(f)

    mapping3445 = sampling['down'][1].astype('float64')
    lv3445 = sampling['meshes'][2]['v']
    lf3445 = sampling['meshes'][2]['f'].astype('int64')

    num_betas = dd['shapedirs'].shape[-1]
    J_reg = dd['J_regressor'].asformat('csr')

    smpl = loadmat('./SMPL_M.mat')
    smpl_J_regressor = smpl['C']['regJoint'][0, 0]
    smpl_template = smpl['C']['meanVerts'][0, 0]
    posedirs = smpl['C']['poseDirs'][0, 0]
    shapedirs = smpl['C']['shapeDirs'][0, 0]
    weights = smpl['C']['blendWeights'][0, 0]

    model3445 = {
        'v_template': mapping3445.dot(smpl_template),
        'weights': np.hstack([np.expand_dims(mapping3445.dot(weights[:, i]), axis=-1)
            for i in range(24)]),
        'posedirs': mapping3445.dot(posedirs.reshape((-1, 207 * 3))).reshape(-1, 3, 207),
        'shapedirs': mapping3445.dot(shapedirs.reshape((-1, 3 * num_betas))).reshape(-1, 3, num_betas),
        'J_regressor': smpl_J_regressor.dot(mapping3445.T),
        'kintree_table': dd['kintree_table'],
        'J': dd['J'],
        'f': lf3445
    }

    mapping1723 = sampling['down'][2].astype('float64')
    lv1723 = sampling['meshes'][3]['v']
    lf1723 = sampling['meshes'][3]['f'].astype('int64')

    model1723 = {
        'v_template': mapping1723.dot(model3445['v_template']),
        'weights': np.hstack([np.expand_dims(mapping1723.dot(model3445['weights'][:, i]), axis=-1)
                              for i in range(24)]),
        'posedirs': mapping1723.dot(model3445['posedirs'].reshape((-1, 207 * 3))).reshape(-1, 3, 207),
        'shapedirs': mapping1723.dot(model3445['shapedirs'].reshape((-1, 3 * num_betas))).reshape(-1, 3, num_betas),
        'J_regressor': model3445['J_regressor'].dot(mapping1723.T),
        'kintree_table': dd['kintree_table'],
        'J': dd['J'],
        'f': lf1723
    }

    return model3445, model1723

def get_hres(v, f):
    """
    Get an upsampled version of the mesh.
    OUTPUT:
        - nv: new vertices
        - nf: faces of the upsampled
        - mapping: mapping from low res to high res
    """
    (mapping, nf) = loop_subdivider(v, f)
    nv = mapping.dot(v)
    return (nv, nf, mapping)


def backwards_compatibility_replacements(dd):
    # replacements
    if 'default_v' in dd:
        dd['v_template'] = dd['default_v']
        del dd['default_v']
    if 'template_v' in dd:
        dd['v_template'] = dd['template_v']
        del dd['template_v']
    if 'joint_regressor' in dd:
        dd['J_regressor'] = dd['joint_regressor']
        del dd['joint_regressor']
    if 'blendshapes' in dd:
        dd['posedirs'] = dd['blendshapes']
        del dd['blendshapes']
    if 'J' not in dd:
        dd['J'] = dd['joints']
        del dd['joints']

    # defaults
    if 'bs_style' not in dd:
        dd['bs_style'] = 'lbs'


def loop_subdivider(mesh_v, mesh_f):
    """Copied from opendr and modified to work in python3."""

    IS = []
    JS = []
    data = []

    vc = get_vert_connectivity(mesh_v, mesh_f)
    ve = get_vertices_per_edge(mesh_v, mesh_f)
    vo = get_vert_opposites_per_edge(mesh_v, mesh_f)

    if True:
        # New values for each vertex
        for idx in range(len(mesh_v)):

            # find neighboring vertices
            nbrs = np.nonzero(vc[:,idx])[0]

            nn = len(nbrs)

            if nn < 3:
                wt = 0.
            elif nn == 3:
                wt = 3./16.
            elif nn > 3:
                wt = 3. / (8. * nn)
            else:
                raise Exception('nn should be 3 or more')
            if wt > 0.:
                for nbr in nbrs:
                    IS.append(idx)
                    JS.append(nbr)
                    data.append(wt)

            JS.append(idx)
            IS.append(idx)
            data.append(1. - (wt * nn))

    start = len(mesh_v)
    edge_to_midpoint = {}

    if True:
        # New values for each edge:
        # new edge verts depend on the verts they span
        for idx, vs in enumerate(ve):

            vsl = list(vs)
            vsl.sort()
            IS.append(start + idx)
            IS.append(start + idx)
            JS.append(vsl[0])
            JS.append(vsl[1])
            data.append(3./8)
            data.append(3./8)

            opposites = vo[(vsl[0], vsl[1])]
            for opp in opposites:
                IS.append(start + idx)
                JS.append(opp)
                data.append(2./8./len(opposites))

            edge_to_midpoint[(vsl[0], vsl[1])] = start + idx
            edge_to_midpoint[(vsl[1], vsl[0])] = start + idx

    f = []

    for f_i, old_f in enumerate(mesh_f):
        ff = np.concatenate((old_f, old_f))

        for i in range(3):
            v0 = edge_to_midpoint[(ff[i], ff[i+1])]
            v1 = ff[i+1]
            v2 = edge_to_midpoint[(ff[i+1], ff[i+2])]
            f.append(row(np.array([v0,v1,v2])))

        v0 = edge_to_midpoint[(ff[0], ff[1])]
        v1 = edge_to_midpoint[(ff[1], ff[2])]
        v2 = edge_to_midpoint[(ff[2], ff[3])]
        f.append(row(np.array([v0,v1,v2])))

    f = np.vstack(f)

    IS = np.array(IS, dtype=np.uint32)
    JS = np.array(JS, dtype=np.uint32)

    if False: # for x,y,z coords
        IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
        JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
        data = np.concatenate ((data,data,data))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij))

    return mtx, f


def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12.

    Copied from opendr library.
    """

    vpv = sp.csc_matrix((len(mesh_v),len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def get_vertices_per_edge(mesh_v, mesh_f):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()

    Copied from opendr library.
    """

    vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:,0] < result[:,1]] # for uniqueness

    return result


def get_vert_opposites_per_edge(mesh_v, mesh_f):
    """Returns a dictionary from vertidx-pairs to opposites.
    For example, a key consist of [4,5)] meaning the edge between
    vertices 4 and 5, and a value might be [10,11] which are the indices
    of the vertices opposing this edge.

    Copied from opendr library.
    """
    result = {}
    for f in mesh_f:
        for i in range(3):
            key = [f[i], f[(i+1)%3]]
            key.sort()
            key = tuple(key)
            val = f[(i+2)%3]

            if key in result:
                result[key].append(val)
            else:
                result[key] = [val]
    return result

if __name__ == '__main__':
    data3445, data1723 = get_lres_smpl_model_data("./models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    # data_h = get_hres_smpl_model_data("./models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    savemat("low_res(3445)_smpl_m.mat", data3445)
    savemat("low_res(1723)_smpl_m.mat", data1723)
    print(1)