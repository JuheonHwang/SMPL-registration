import numpy as np
import pickle
import scipy.sparse as sp
from scipy.io import savemat
from scipy.io import loadmat
import json

if __name__ == '__main__':
    # with open("C:\\Users\\User\\Desktop\\camera팀\\cloth_data_4_duc\\data_hjh\\smpl\\000000.json", 'r') as f:
    #     data = json.load(f)
    # with open('D:\\SMPL_python_v.1.1.0\\smpl\\SMPL_high_res_mapping.pkl', 'rb') as f:
    #     data = pickle.load(f)
    with open('D:\\SMPL_python_v.1.1.0\\smpl\\smoothed_poisson_mesh_000001_smpl.pkl', 'rb') as f:
        data = pickle.load(f)
    json_to_pickle = {}
    data = data[0]
    json_to_pickle['pose'] = np.asarray(data['poses'][0], dtype=np.float32)
    json_to_pickle['pose'][:3] = np.asarray(data['Rh'][0], dtype=np.float32)
    json_to_pickle['betas'] = np.asarray(data['shapes'][0], dtype=np.float32)
    json_to_pickle['trans'] = np.asarray([data['Th'][0][0], data['Th'][0][1], data['Th'][0][2]], dtype=np.float32)
    # json_to_pickle['scale'] = np.asarray([1000], dtype=np.float32)

    # savemat("poisson_mesh_deform_000001_fit_param.mat", json_to_pickle)
    print()

    with open('fitted.pkl', 'wb') as h:
        pickle.dump(json_to_pickle, h, protocol=pickle.HIGHEST_PROTOCOL)
    print()