import os
from glob import glob
import pickle as pkl
import numpy as np
import torch as T
from pytorch3d.io import save_obj, load_ply
from multiprocessing import Pool

from lib.smpl.smplpytorch.smplpytorch.pytorch.smpl_layer import SMPL_Layer


def smpl_lr(subject):
    out_root = 'F:/2_Personal/duc/data'
    os.makedirs(os.path.join(out_root, subject), exist_ok=True)


    smpl_pkl = sorted(glob(f"F:/2_Personal/hjh/MPI_MeshRegistration-main/data/{subject}/*smpld.pkl"))
    betas, poses, trans, offsets = [], [], [], []
    for spkl in smpl_pkl:
        with open(spkl, 'rb') as f:
            smpl_params = pkl.load(f)

        betas.append(smpl_params['betas'])
        poses.append(smpl_params['pose'])
        trans.append(smpl_params['trans'])

    betas = T.from_numpy(np.array(betas))
    poses = T.from_numpy(np.array(poses))
    trans = T.from_numpy(np.array(trans))
    smpl_layer = SMPL_Layer(center_idx=0, gender='neutral', num_betas=10,
                            model_root='F:/2_Personal/hjh/MPI_MeshRegistration-main/assets')
    smpl_verts, _, _, _ = smpl_layer(poses, betas, trans)
    for smoothed_smpl, file in zip(smpl_verts, smpl_pkl):
        save_file = os.path.basename(file)[:-5] + '.obj'
        save_file = os.path.join(out_root, subject, save_file)
        save_obj(save_file, smoothed_smpl, smpl_layer.th_faces)
        print(save_file)


if __name__ == '__main__':
    # subjects = ['01_20220530_HJH_origin_result', '01_20220530_JMG_origin_result', '01_20220530_KJK_origin_result',
    #             '01_20220530_LSM_origin_result', '01_20220602_LSM_origin_result', '01_20220602_SHW_origin_result',
    #             '01_20220603_BEJ_origin_result', '01_20220603_HJH_origin_result', '01_20220603_LJH_origin_result']

    subjects = ['01_20220523_KSJ2_origin_result', '01_20220523_KSJ3_origin_result', '01_20220523_KSJ_origin_result',
                '01_20220523_LHS2_origin_result', '01_20220523_LHS3_origin_result', '01_20220523_LHS_origin_result',
                '01_20220527_KSJ_origin_result', '01_20220527_LHS_origin_result', '01_20220527_LJH_origin_result',
                '01_20220527_PYS_origin_result']
    with Pool(len(subjects)) as pool:
        pool.map(smpl_lr, subjects)

    # for subject in subjects:
    #     smpl_lr(subject)
