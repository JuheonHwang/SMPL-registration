from glob import glob
import pickle as pkl
import numpy as np
from scipy import signal
import torch as T
from pytorch3d.io import save_obj
from multiprocessing import Pool

from lib.smpl.smplpytorch.smplpytorch.pytorch.smpl_layer import SMPL_Layer


def smooth_mesh_seq(subject):
    smpl_pkl = sorted(glob(f"F:/2_Personal/hjh/MPI_MeshRegistration-main/data/{subject}/*smpld.pkl"))
    betas, poses, trans, offsets = [], [], [], []
    for spkl in smpl_pkl:
        with open(spkl, 'rb') as f:
            smpl_params = pkl.load(f)

        betas.append(smpl_params['betas'])
        poses.append(smpl_params['pose'])
        trans.append(smpl_params['trans'])
        offsets.append(smpl_params['offs'])

    betas = T.from_numpy(np.array(betas))
    poses = T.from_numpy(np.array(poses))
    trans = T.from_numpy(np.array(trans))
    offsets = T.from_numpy(np.array(offsets))
    smpl_layer = SMPL_Layer(center_idx=0, gender='neutral', num_betas=10,
                            model_root='F:/2_Personal/hjh/MPI_MeshRegistration-main/assets', hires=True)
    smpl_verts, _, _, _ = smpl_layer(poses, betas, trans, offsets)

    smpl_verts = signal.savgol_filter(smpl_verts.numpy(), 11, 1, axis=0)
    for smoothed_smpl, file in zip(T.from_numpy(smpl_verts), smpl_pkl):
        save_file = file[:-4] + '_smoothed.obj'
        save_obj(save_file, smoothed_smpl, smpl_layer.th_faces)
        print(save_file)


if __name__ == '__main__':
    subjects = ['01_20220530_HJH_origin_result', '01_20220530_JMG_origin_result', '01_20220530_KJK_origin_result',
                '01_20220530_LSM_origin_result', '01_20220602_LSM_origin_result', '01_20220602_SHW_origin_result',
                '01_20220603_BEJ_origin_result', '01_20220603_HJH_origin_result', '01_20220603_LJH_origin_result']

    with Pool(len(subjects)) as pool:
        pool.map(smooth_mesh_seq, subjects)
