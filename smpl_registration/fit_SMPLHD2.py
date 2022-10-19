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
    def __init__(self, model_root, device='cuda:5', save_name='smpld', debug=False, hands=False):
        super(SMPLDFitter, self).__init__(model_root, device, save_name, debug, hands)
        self.save_name_base = 'smplhd' if self.hands else 'smpld'
        self.hres_save_name_base = 'smplhd_hres' if self.hands else 'smpld_hres'
        self.hres2lres_save_name_base = 'smplhd_hres2lres' if self.hands else 'smpld_hres2lres'

    def fit(self, scans, pose_files, smpl_pkl, gender='male', save_path=None):
        # if smpl_pkl is None or smpl_pkl[0] is None:
        #     print('SMPL not specified, fitting SMPL now')
        #     pose, betas, trans = super(SMPLDFitter, self).fit(scans, pose_files, gender, save_path)
        # else:
        #     # load from fitting results
        #     pose, betas, trans = self.load_smpl_params(smpl_pkl)

        pose, betas, trans = super(SMPLDFitter, self).fit(scans, pose_files, smpl_pkl, gender, save_path)

        betas, pose, trans = torch.tensor(betas), torch.tensor(pose), torch.tensor(trans)
        # Batch size
        batch_sz = len(scans)

        # init smpl
        smpl = self.init_smpl(batch_sz, gender, pose, betas, trans)

        # Load scans and center them. Once smpl is registered, move it accordingly.
        # Do not forget to change the location of 3D joints/ landmarks accordingly.
        th_scan_meshes = self.load_scans(scans, self.device)

        # optimize offsets
        self.optimize_offsets(th_scan_meshes, smpl, 15, 20)
        self.optimize_offsets2(th_scan_meshes, smpl, 15, 20)
        hres_optimized_meshes = self.optimize_high_res(th_scan_meshes, smpl, 10, 20)

        hres_optimized_verts = torch.stack([*hres_optimized_meshes.verts_list()], 0)
        lres_mapping = pkl.load(open("../assets/SMPL_low_res_mapping.pkl", 'rb'), encoding='latin-1')
        lres_mapping = lres_mapping['mapping']
        lres_mapping = torch.tensor(lres_mapping).float().to(hres_optimized_meshes.device)
        hres2lres_verts = torch.matmul(lres_mapping, hres_optimized_verts)
        hres2lres_verts_list = [hres2lres_verts[i].clone().detach() for i in range(batch_sz)]
        hres2lres_faces_list = [smpl.faces for i in range(batch_sz)]
        hres2lres_optimized_meshes = Meshes(hres2lres_verts_list, hres2lres_faces_list)

        if save_path is not None:
            if not exists(save_path):
                os.makedirs(save_path)
            self.save_hres_outputs(save_path, scans, hres_optimized_meshes, hres2lres_optimized_meshes, save_name_hres=self.hres_save_name_base, save_name_hres2lres=self.hres2lres_save_name_base)
            return self.save_outputs(save_path, scans, smpl, th_scan_meshes, save_name=self.save_name_base)

    def forward_step_offset(self, th_scan_meshes, smpl, init_smpl_lap, target_lengths):
        """
            Performs a forward step, given smpl and scan meshes.
            Then computes the losses.
        """
        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

        # losses
        loss = dict()
        # loss['s2m'] = point_mesh_face_distance(th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list()))
        # loss['m2s'] = point_mesh_face_distance(th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list()))
        loss['s2m'] = 3 * point_mesh_face_distance(th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list()))
        loss['m2s'] = 6 * point_mesh_face_distance(th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list()))
        lap_new = mesh_laplacian_smoothing(th_smpl_meshes, reduction=None) # (V, 3)
        # init_lap = mesh_laplacian_smoothing(th_smpl_meshes, reduction=None) # (V, 3)
        # reference: https://github.com/NVIDIAGameWorks/kaolin/blob/v0.1/kaolin/metrics/mesh.py#L155
        # loss['lap'] = 25 * torch.mean(torch.sum((lap_new - init_smpl_lap)**2, 1))
        loss['lap'] = 15 * torch.mean(torch.sum(lap_new ** 2, 1))
        # loss['lap'] = mesh_laplacian_smoothing(th_smpl_meshes, method='uniform')
        # loss['edge'] = 5 * mesh_edge_loss(th_smpl_meshes)
        loss['edge'] = 3 * mesh_edge_loss_initial(th_smpl_meshes, 1.5 * target_lengths)
        # loss['edge'] = mesh_edge_loss(th_smpl_meshes)
        loss['offsets'] = torch.mean(torch.mean(0.005 * smpl.offsets ** 2, axis=1))
        return loss

    def forward_step_offset2(self, th_scan_meshes, smpl, init_smpl_lap, target_lengths):
        """
            Performs a forward step, given smpl and scan meshes.
            Then computes the losses.
        """
        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

        # losses
        loss = dict()
        loss['s2m'] = 3 * point_mesh_face_distance(th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list()))
        loss['m2s'] = 9 * point_mesh_face_distance(th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list()))
        lap_new = mesh_laplacian_smoothing(th_smpl_meshes, reduction=None) # (V, 3)
        # reference: https://github.com/NVIDIAGameWorks/kaolin/blob/v0.1/kaolin/metrics/mesh.py#L155
        # loss['lap'] = 1 * torch.mean(torch.sum((lap_new - init_smpl_lap)**2, 1))
        loss['lap'] = 3 * torch.mean(torch.sum(lap_new ** 2, 1))
        # loss['edge'] = 0.1 * mesh_edge_loss(th_smpl_meshes)
        loss['edge'] = 1 * mesh_edge_loss_initial(th_smpl_meshes, target_lengths)
        loss['offsets'] = torch.mean(torch.mean(0.001 * smpl.offsets ** 2, axis=1))
        return loss

    def forward_step_offset_hres(self, th_scan_meshes, init_hres_smpl, hres_offsets, init_smpl_lap, target_lengths):
        """
            Performs a forward step, given smpl and scan meshes.
            Then computes the losses.
        """
        # forward
        new_hres_smpl = init_hres_smpl.offset_verts(hres_offsets)

        # losses
        loss = dict()
        loss['s2m'] = 6 * point_mesh_face_distance(new_hres_smpl, Pointclouds(points=th_scan_meshes.verts_list()))
        loss['m2s'] = 11 * point_mesh_face_distance(th_scan_meshes, Pointclouds(points=new_hres_smpl.verts_list()))
        lap_new = mesh_laplacian_smoothing(new_hres_smpl, reduction=None) # (V, 3)
        # reference: https://github.com/NVIDIAGameWorks/kaolin/blob/v0.1/kaolin/metrics/mesh.py#L155
        # loss['lap'] = 1.5 * torch.mean(torch.sum((lap_new - init_smpl_lap)**2, 1))
        loss['lap'] = 1 * torch.mean(torch.sum(lap_new ** 2, 1))
        # loss['edge'] = 0.1 * mesh_edge_loss(new_hres_smpl)
        loss['edge'] = 3 * mesh_edge_loss_initial(new_hres_smpl, target_lengths)
        # loss['offsets'] = torch.mean(torch.mean(0.001 * hres_offsets ** 2, axis=1))
        return loss

    def optimize_offsets(self, th_scan_meshes, smpl, iterations, steps_per_iter):
        # Optimizer
        # optimizer = torch.optim.Adam([smpl.offsets, smpl.pose, smpl.betas], 0.005, betas=(0.9, 0.999))
        optimizer = torch.optim.Adam([smpl.offsets, smpl.pose, smpl.betas], 0.01, betas=(0.9, 0.999))

        # Get loss_weights
        weight_dict = self.get_loss_weights()

        # precompute initial laplacian of the smpl meshes
        bz = smpl.offsets.shape[0]
        verts, _, _, _ = smpl()
        verts_list = [verts[i].clone().detach() for i in range(bz)]
        faces_list = [torch.tensor(smpl.faces) for i in range(bz)]
        init_smpl = Meshes(verts_list, faces_list)
        init_lap = mesh_laplacian_smoothing(init_smpl)
        # init_smpl_edges_packed = init_smpl.edges_packed()
        # init_smpl_verts_packed = init_smpl.verts_packed()
        # init_smpl_verts_edges = init_smpl_verts_packed[init_smpl_edges_packed]
        # v0, v1 = init_smpl_verts_edges.unbind(1)
        v0, v1 = init_smpl.verts_packed()[init_smpl.edges_packed()].unbind(1)
        target_lengths = (v0 - v1).norm(dim=1, p='fro')

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL+D first part')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_step_offset(th_scan_meshes, smpl, init_lap, target_lengths)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Lx100. Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, loss_dict[k].mean().item() * 100)
                loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes)

    def optimize_offsets2(self, th_scan_meshes, smpl, iterations, steps_per_iter):
        # Optimizer
        # optimizer = torch.optim.Adam([smpl.offsets, smpl.pose, smpl.betas], 0.005, betas=(0.9, 0.999))
        optimizer = torch.optim.Adam([smpl.offsets, smpl.pose, smpl.betas], 0.01, betas=(0.9, 0.999))

        # Get loss_weights
        weight_dict = self.get_loss_weights()

        # precompute initial laplacian of the smpl meshes
        bz = smpl.offsets.shape[0]
        verts, _, _, _ = smpl()
        verts_list = [verts[i].clone().detach() for i in range(bz)]
        faces_list = [torch.tensor(smpl.faces) for i in range(bz)]
        init_smpl = Meshes(verts_list, faces_list)
        init_lap = mesh_laplacian_smoothing(init_smpl)
        # init_smpl_edges_packed = init_smpl.edges_packed()
        # init_smpl_verts_packed = init_smpl.verts_packed()
        # init_smpl_verts_edges = init_smpl_verts_packed[init_smpl_edges_packed]
        # v0, v1 = init_smpl_verts_edges.unbind(1)
        v0, v1 = init_smpl.verts_packed()[init_smpl.edges_packed()].unbind(1)
        target_lengths = (v0 - v1).norm(dim=1, p='fro')

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL+D second part')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_step_offset2(th_scan_meshes, smpl, init_lap, target_lengths)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Lx100. Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, loss_dict[k].mean().item() * 100)
                loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes)

    def optimize_high_res(self, th_scan_meshes, smpl, iterations, steps_per_iter):
        # Optimizer
        # optimizer = torch.optim.Adam([smpl.offsets, smpl.pose, smpl.betas], 0.005, betas=(0.9, 0.999))
        verts, _, _, _ = smpl()
        high_res = pkl.load(open("../assets/SMPL_high_res_mapping.pkl", 'rb'), encoding='latin-1')
        hres_face, hres_mapping = high_res['face'], high_res['mapping']
        # mapping = torch.tensor(mapping).to_sparse().float()
        hres_face = torch.tensor(hres_face.astype(np.int32)).long().to(verts.device)
        hres_mapping = torch.tensor(hres_mapping).float().to(verts.device)
        hres_verts = torch.matmul(hres_mapping, verts)

        # Get loss_weights
        # weight_dict = self.get_loss_weights()
        weight_dict = self.get_loss_weights_hres()

        # precompute initial laplacian of the smpl meshes
        bz = smpl.offsets.shape[0]

        hres_verts_list = [hres_verts[i].clone().detach() for i in range(bz)]
        hres_faces_list = [hres_face for i in range(bz)]
        init_hres_smpl = Meshes(hres_verts_list, hres_faces_list)
        init_hres_lap = mesh_laplacian_smoothing(init_hres_smpl)
        # init_hres_smpl_edges_packed = init_hres_smpl.edges_packed()
        # init_hres_smpl_verts_packed = init_hres_smpl.verts_packed()
        # init_hres_smpl_verts_edges = init_hres_smpl_verts_packed[init_hres_smpl_edges_packed]
        # v0, v1 = init_hres_smpl_verts_edges.unbind(1)
        v0, v1 = init_hres_smpl.verts_packed()[init_hres_smpl.edges_packed()].unbind(1)
        target_lengths = (v0 - v1).norm(dim=1, p='fro')

        hres_offsets = torch.full(init_hres_smpl.verts_packed().shape, 0.0, device=verts.device, requires_grad=True)

        optimizer = torch.optim.Adam([hres_offsets], 0.01, betas=(0.9, 0.999))

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing High Resolution')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_step_offset_hres(th_scan_meshes, init_hres_smpl, hres_offsets, init_hres_lap, target_lengths)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Lx100. Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, loss_dict[k].mean().item() * 100)
                loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes)

        return init_hres_smpl.offset_verts(hres_offsets)

    def get_loss_weights(self):
        """Set loss weights"""
        # loss_weight = {'s2m': lambda cst, it: 30. ** 2 * cst * (1 + it),
        #                'm2s': lambda cst, it: 30. ** 2 * cst / (1 + it),
        #                'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
        #                'offsets': lambda cst, it: 150. ** -1 * cst / (1 + it),
        #                'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
        #                'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
        #                'lap': lambda cst, it: 2000**2*cst / (1 + it),
        #                'edge': lambda cst, it: 30 ** 2 * cst / (1 + it), # mesh edge
        #                'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it)
        #                }
        loss_weight = {'s2m': lambda cst, it: 30. ** 2 * cst / (1 + it),
                       'm2s': lambda cst, it: 30. ** 2 * cst * (1 + it),
                       'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                       'offsets': lambda cst, it: 150. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: 2000 ** 2 * cst / (1 + it),
                       'edge': lambda cst, it: 30 ** 2 * cst * (1 + it),  # mesh edge
                       'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it)
                       }
        return loss_weight

    def get_loss_weights_hres(self):
        """Set loss weights"""
        # loss_weight = {'s2m': lambda cst, it: 30. ** 2 * cst * (1 + it),
        #                'm2s': lambda cst, it: 30. ** 2 * cst / (1 + it),
        #                'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
        #                'offsets': lambda cst, it: 150. ** -1 * cst / (1 + it),
        #                'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
        #                'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
        #                'lap': lambda cst, it: 2000**2*cst / (1 + it),
        #                'edge': lambda cst, it: 30 ** 2 * cst / (1 + it), # mesh edge
        #                'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it)
        #                }
        loss_weight = {'s2m': lambda cst, it: 30. ** 2 * cst / (1 + it),
                       'm2s': lambda cst, it: 30. ** 2 * cst * (1 + it),
                       'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                       'offsets': lambda cst, it: 150. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: 2000 ** 2 * cst / (1 + it),
                       'edge': lambda cst, it: 30 ** 2 * cst * (1 + it),  # mesh edge
                       'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it)
                       }
        return loss_weight


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
    args.scan_path = glob.glob('../data/02_20220527_JMG_origin/meshes/*.ply')
    args.pose_file = glob.glob('../data/02_20220527_JMG_origin/keypoints3d/*.json')
    args.smpl_pkl = glob.glob("../data/02_20220527_JMG_origin/SMPL_pkls/*.pkl")
    # args.scan_path = glob.glob('../data/mk2/poisson_mesh_smooth_000004.obj')
    # args.pose_file = glob.glob('../data/mk2/poisson_mesh_smooth_000004.json')
    # args.smpl_pkl = glob.glob("../data/mk2/poisson_mesh_smooth_000004.pkl")
    # args.scan_path = glob.glob('../data/mk2/poisson_mesh_smooth_000003.obj')
    # args.pose_file = glob.glob('../data/mk2/poisson_mesh_smooth_000003.json')
    # args.smpl_pkl = glob.glob("../data/mk2/poisson_mesh_smooth_000003.pkl")
    args.save_path = '../data/02_20220527_JMG_origin_result'
    args.gender = 'neutral'
    config = load_config(args.config_path)
    args.model_root = Path(config["SMPL_MODELS_PATH"])

    main(args)