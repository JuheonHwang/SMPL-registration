"""
fit SMPLH+offset to scans

created by Xianghui, 12 January 2022
"""
import sys, os
sys.path.append(os.getcwd())
from os.path import split, join, exists
import torch
from tqdm import tqdm
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance, mesh_edge_loss
from lib.mesh_laplacian import mesh_laplacian_smoothing
from smpl_registration.fit_SMPLH import SMPLHFitter
from smpl_registration.mesh_edge_loss_initial import mesh_edge_loss_initial
import pickle as pkl
import numpy as np
import glob


class SMPLDFitter(SMPLHFitter):
    def __init__(self, model_root, device='cuda:3', save_name='smpl', debug=False, hands=False):
        super(SMPLDFitter, self).__init__(model_root, device, save_name, debug, hands)
        self.save_name_base = 'smplh' if self.hands else 'smpl'

    def fit(self, scans, pose_files, smpl_pkl, gender='male', save_path=None):
        pose, betas, trans = self.load_smpl_params(smpl_pkl)

        betas, pose, trans = torch.tensor(betas), torch.tensor(pose), torch.tensor(trans)
        # Batch size
        batch_sz = len(scans)

        # init smpl
        smpl = self.init_smpl(batch_sz, gender, pose, betas, trans)

        if save_path is not None:
            if not exists(save_path):
                os.makedirs(save_path)
            return self.save_outputs(save_path, scans, smpl, None, save_name=self.save_name_base)

def main(args):
    fitter = SMPLDFitter(args.model_root, debug=args.display, hands=args.hands)
    if type(args.scan_path) == list:
        fitter.fit(args.scan_path, args.pose_file, args.smpl_pkl, args.gender, args.save_path)
    else:
        fitter.fit([args.scan_path], [args.pose_file], [args.smpl_pkl], args.gender, args.save_path)



if __name__ == "__main__":
    import argparse
    from utils.configs import load_config
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Run Model')
    # parser.add_argument('scan_path', type=str, help='path to the 3d scans')
    # parser.add_argument('pose_file', type=str, help='3d body joints file')
    # parser.add_argument('save_path', type=str, help='save path for all scans')
    parser.add_argument("--config-path", "-c", type=Path, default="../config.yml",
                        help="Path to yml file with config")
    # parser.add_argument("--config-path", "-c", type=Path, default="config.yml",
    #                     help="Path to yml file with config")
    parser.add_argument('-gender', type=str, default='male') # can be female
    parser.add_argument('-smpl_pkl', type=str, default=None)  # In case SMPL fit is already available
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('-hands', default=False, action='store_true', help='use SMPL+hand model or not')
    args = parser.parse_args()

    # args.scan_path = 'data/mesh_1/scan.obj'
    # args.pose_file = 'data/mesh_1/3D_joints_all.json'
    # args.display = True
    # args.save_path = 'data/mesh_1'
    # args.gender = 'male'
    # args.smpl_pkl = "data/mesh_1/scan_smpl.pkl"

    # # args.scan_path = '../data/mk/poisson_mesh_deform_000001.obj'
    # args.scan_path = '../data/mk/poisson_mesh_smooth_000001.obj'
    # args.pose_file = '../data/mk/000001.json'
    # # args.display = True
    # args.save_path = '../data/mk'
    # # args.gender = 'male'
    # args.gender = 'neutral'
    # # args.smpl_pkl = "../data/mk/poisson_mesh_smooth_000001_smpl.pkl"
    # args.smpl_pkl = "../data/mk/fitted.pkl"
    # config = load_config(args.config_path)
    # args.model_root = Path(config["SMPL_MODELS_PATH"])
    args.scan_path = glob.glob('../data/20220527_PYS_origin/meshes/Removing_Scaffold/*.ply')
    args.pose_file = glob.glob('../data/20220527_PYS_origin/keypoints3d/*.json')
    args.smpl_pkl = glob.glob("../data/20220527_PYS_origin_result/*_smpld.pkl")

    args.save_path = '../data/20220527_PYS_origin_result_smpl'
    args.gender = 'neutral'
    config = load_config(args.config_path)
    args.model_root = Path(config["SMPL_MODELS_PATH"])

    main(args)