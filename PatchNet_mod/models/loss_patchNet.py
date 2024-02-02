import sys, os

import torch
import torch.nn as nn
import numpy as np
import math
import open3d as o3d

import models.config as cfg

#from utils.pcd_utils import *
from models.node_proc import convert_embedding_to_explicit_params, compute_inverse_occupancy, \
    sample_rbf_surface, sample_rbf_weights, bounding_box_error, extract_view_omegas_from_embedding

colors_rgb = [
    torch.tensor([0, 0, 0]),        # Black
    torch.tensor([128, 0, 0]),      # Dark Red
    torch.tensor([0, 128, 0]),      # Dark Green
    torch.tensor([0, 0, 128]),      # Dark Blue
    torch.tensor([128, 128, 0]),    # Dark Yellow
    torch.tensor([128, 0, 128]),    # Dark Magenta
    torch.tensor([0, 128, 128]),    # Dark Cyan
    torch.tensor([64, 64, 64]),     # Dark Grey
    torch.tensor([20, 20, 20]),     # Very Dark Grey
    torch.tensor([100, 50, 50]),    # Dark Maroon
    torch.tensor([50, 100, 50]),    # Dark Forest Green
    torch.tensor([50, 50, 100]),    # Dark Navy Blue
    torch.tensor([100, 100, 50]),   # Dark Olive
    torch.tensor([100, 50, 100]),   # Dark Purple
    torch.tensor([50, 100, 100]),   # Dark Teal
    torch.tensor([80, 80, 80]),     # Darker Grey
    torch.tensor([100, 80, 80]),    # Darker Maroon
    torch.tensor([80, 100, 80]),    # Darker Forest Green
    torch.tensor([80, 80, 100]),    # Darker Navy Blue
    torch.tensor([70, 70, 0]),      # Dark Olive Green
]


class PatchNetLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ext_loss = ExtLoss()
        self.recon_loss = ReconLoss()
    def forward(self,input):
        loss_total = 0
        loss_ext_loss = ExtLoss(input)
        loss_total += loss_ext_loss
        return loss_total

class ReconLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class ExtLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        '''Sample Loss'''
        self.sur_loss = SurLoss()
        self.cov_loss = CovLoss()
        self.rot_loss = RotLoss()

        '''Latent Loss'''
        self.scl_loss = SclLoss()
        self.var_loss = VarLoss()


        '''Params'''
        self.variable_patch_scaling = True
        self.minimum_scale = 0.01
        self.maximum_scale = 0.5
        self.use_rotations = True

    def forward(self,input):
        # input["xyz"]:   # num_scenes x mixture_latent_size
        # input["mixture_latent_vectors"]  # num_scenes x samp_per_scene x 3
        pass
        '''Prepare_loss_components'''
        mixture_latent_mode = "all_explicit" # PatchNet Training

        mixture_latent_vectors = input["mixture_latent_vectors"]  # num_scenes x mixture_latent_size
        xyz = input["xyz"]  # num_scenes x samp_per_scene x 3

        patch_metadata_input = mixture_latent_vectors

        # transform input latent vectors to patch metadata
        if mixture_latent_mode == "all_explicit":
            patch_metadata = patch_metadata_input  # num_scenes x (num_patches * (patch_latent_size + 3 + 3 + 1)) # patch latent, position, rotation, scaling

        patch_metadata = patch_metadata.reshape(-1, self.num_patches, self.patch_latent_size + 3 + 3 + (
            1 if self.variable_patch_scaling else 0))  # num_scenes x num_patches x (patch_latent_size + 3 + 3 + 1)
        global_xyz = xyz.repeat(1, 1, self.num_patches).view(-1, input["num_samp_per_scene"], self.num_patches,
                                                             3)  # num_scenes x samp_per_scene x num_patches x 3

        patch_latent_vectors = patch_metadata[:, :, :self.patch_latent_size]  # num_scenes x num_patches x patch_latent_size
        patch_position = patch_metadata[:, :, self.patch_latent_size:self.patch_latent_size + 3]  # num_scenes x num_patches x 3
        patch_rotation = patch_metadata[:, :, self.patch_latent_size + 3:self.patch_latent_size + 6]  # num_scenes x num_patches x 3

        if self.variable_patch_scaling:
            patch_scaling = patch_metadata[:, :, self.patch_latent_size + 6:self.patch_latent_size + 7]  # num_scenes x num_patches x 1. this is the scaling of the patch size, i.e. a value of 2 means that the patch's radius is twice as big
            if self.minimum_scale > 0.:
                # minimum_scaling = 0.01
                patch_scaling = torch.clamp(patch_scaling, min=self.minimum_scale)
            if self.maximum_scale != 1000.:
                # maximum_scaling = 0.5
                patch_scaling = torch.clamp(patch_scaling, max=self.maximum_scale)


        '''transform_global_to_local'''
        patch_xyz = global_xyz.clone()
        patch_xyz -= patch_position.unsqueeze(1)  # num_scenes x samp_per_scene x num_patches x 3
        if self.use_rotations:
            rotations = self._convert_euler_to_matrix(patch_rotation.reshape(-1, 3)).view(-1, self.num_patches, 3, 3)  # num_scenes x num_patches x 3 x 3
            # first argument: num_scenes x 1 x num_patches x 3 x 3
            # second argument: num_scenes x samp_per_scene x num_patches x 3 x 1
            patch_xyz = torch.matmul(torch.transpose(rotations, -2, -1).unsqueeze(1),
                                     patch_xyz.unsqueeze(-1))  # num_scenes x samp_per_scene x num_patches x 3 x 1

        # patch_scaling: num_scenes x num_patches x 1
        # patch_xyz: num_scenes x samp_per_scene x num_patches x 3 x 1
        patch_scaling =  patch_xyz/patch_scaling.unsqueeze(1).unsqueeze(-1)

        '''calculate distance'''
        unscaled_center_distances_nonflat = torch.norm(patch_xyz, dim=-1)  # num_scenes x samp_per_scene x num_patches
        scaled_center_distances_nonflat = unscaled_center_distances_nonflat / patch_scaling.squeeze(-1).unsqueeze(1)
        scaled_center_distances = scaled_center_distances_nonflat.flatten()  # scaled distances to patch center
        unscaled_center_distances = unscaled_center_distances_nonflat.flatten()  # unscaled distances to patch center


        '''calculating Gaussian Weights'''
        patch_weight_type = "gaussian"
        if patch_weight_type == "binary":
            patch_weights = (scaled_center_distances < 1.).to(
                torch.float).detach()  # num_scenes * samp_per_scene * num_patches
        elif patch_weight_type == "gaussian":
            std_cutoff = 3.
            smooth_patch_seam_weights = True
            patch_weight_std = 1. / std_cutoff
            import numpy as np
            distances_to_use = scaled_center_distances  # if self.patch_scaling else unscaled_center_distances
            patch_weights = torch.zeros_like(scaled_center_distances)
            patch_mask = scaled_center_distances < 1.
            patch_weights[patch_mask] = torch.exp(
                -0.5 * (scaled_center_distances[patch_mask] / patch_weight_std) ** 2) - (np.exp(
                -0.5 * std_cutoff ** 2) if smooth_patch_seam_weights else 0.)  # samples * num_patches
            patch_weights[~patch_mask] = 0.
        else:
            raise RuntimeError("missing patch_weight_type")


        '''Calculating Patch Weights (MASKING)'''
        patch_weights = patch_weights.view(-1, self.num_patches)  # samples x num_patches
        patch_weight_normalization = torch.sum(patch_weights, 1)  # samples
        patch_weight_norm_mask = patch_weight_normalization == 0.
        patch_weights[patch_weight_norm_mask, :] = 0.0
        patch_weights[~patch_weight_norm_mask, :] = patch_weights[~patch_weight_norm_mask, :] / \
                                                    patch_weight_normalization[~patch_weight_norm_mask].unsqueeze(-1)
        patch_weights = patch_weights.view(-1)  # samples * num_patches





        loss_ext = 0
        w_sur = 1.0

        loss_sur =SurLoss(unscaled_center_distances)
        loss_ext += w_sur * loss_sur

        return loss_ext


class SurLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class CovLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,unscaled_center_distances):
        free_space_distance_threshold = 0.2 * self.non_variable_patch_radius
        free_space_loss_weight = 5.0

        surface_sdf_threshold = 0.02
        sdf_gt = input["sdf_gt"].squeeze(-1).flatten()
        surface_mask = torch.abs(sdf_gt) <= surface_sdf_threshold

        if "sdf_gt" in input and "num_samp_per_scene" in input:
            masked_distances = unscaled_center_distances.clone().view(-1,
                                                                      self.num_patches)  # distances: samples * num_patches
            masked_distances[~surface_mask, :] = 10000000.
            masked_distances = masked_distances.view(-1, input["num_samp_per_scene"],
                                                     self.num_patches)  # num_scenes x samples_per_scene x num_patches

            closest_surface_distances, closest_surface_indices = torch.min(masked_distances,
                                                                           dim=1)  # num_scenes x num_patches

            free_space_patches = closest_surface_distances > free_space_distance_threshold

            closest_surface_distances[~free_space_patches] = 0.
            free_space_scene_normalization = torch.sum(free_space_patches, dim=1)  # num_scenes
            free_space_scenes = free_space_scene_normalization > 0
            eps = 0.001
            free_space_scene_losses = torch.sum(closest_surface_distances[free_space_scenes, :], dim=1) / (
                        free_space_scene_normalization[free_space_scenes].float() + eps)  # num_scenes
            free_space_loss = torch.sum(free_space_scene_losses) / (torch.sum(free_space_scenes) + eps)
        pass



class RotLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class VarLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class SclLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass