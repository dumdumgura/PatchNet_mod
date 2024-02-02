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
def gradient_func(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def vis_data(pts):
    import pyrender
    xyz = pts[:, 0:3]
    sdf = pts[:, 3]
    colors = np.zeros(xyz.shape)
    colors[sdf <= 0.00, 2] = 1
    colors[sdf > 0.00, 0] = 1
    cloud = pyrender.Mesh.from_points(xyz, colors)

    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=3)


NORMALIZATION_EPS = 1e-8

class SamplerLoss(torch.nn.Module):
    def __init__(self):
        super(SamplerLoss, self).__init__()

        self.point_loss = PointLoss()
        self.node_center_loss = NodeCenterLoss()
        self.affinity_loss = AffinityLoss()
        self.unique_neighbor_loss = UniqueNeighborLoss()
        self.viewpoint_consistency_loss = ViewpointConsistencyLoss()
        self.surface_consistency_loss = SurfaceConsistencyLoss()
        self.node_sparsity_loss =NodeSparsityLoss()
        self.node_similiar_loss = NodeSimiliarityLoss()
        self.ldif_recon_loss = LDIF_Reconstruction_Loss()
        self.center_var_loss = CenterVarLoss()

        #reconstruction loss
        self.reconstruction_loss =ReconstructionLoss(type=cfg.supervision)






    def bounding_box(self,points):
        """
        Calculate the bounding box for each set of points.

        Parameters:
        - points (torch.Tensor): Tensor of points with shape (batch_size, num_pts, 3).

        Returns:
        - torch.Tensor: Tensor containing the bounding box for each set of points.
        """
        min_values, _ = torch.min(points, dim=1)  #batch,3
        max_values, _ = torch.max(points, dim=1)  #batch,3
        return min_values,max_values

    def calculate_3d_bounding_box_corners(self,left_upper, right_bottom, batch):
        x1, y1, z1 = left_upper
        x2, y2, z2 = right_bottom

        # Calculate corner points
        corners = torch.tensor([
            (x1, y2, z1),
            (x2, y2, z1),
            (x1, y1, z1),
            (x2, y1, z1),
            (x1, y2, z2),
            (x2, y2, z2),
            (x1, y1, z2),
            (x2, y1, z2),
        ])

        return corners[None].expand(batch,-1,-1)


    def forward(self, embedding, # prediction of 3D Gaussians ball Batch,num_node*11
                pred_sdf,
                ldif_gaussian,
                xyz_normal_sel_vis,
                near_surface_pts,
                on_surface_pts,
                coords,
                grid, # sdf_grid
                epoch
                #uniform_samples, near_surface_samples, surface_samples, \
                #grid, world2grid, world2orig, rotated2gaps, bbox_lower, bbox_upper, \
                #source_idxs, target_idxs, pred_distances, pair_weights, affinity_matrix, evaluate=False
                ):

        B,num_pts,_ = near_surface_pts.shape

        uniform_samples=near_surface_pts[:,-num_pts//2:,0:4]
        near_surface_samples=near_surface_pts[:,0:num_pts//2,0:4]
        on_surface_samples = on_surface_pts[:,0:num_pts//2,0:4]
        #vis_data(near_surface_samples[0,:,:].detach().cpu().numpy())
        #vis_data(uniform_samples[0, :, :].detach().cpu().numpy())


        num_node = embedding.shape[1]//11
        loss_total = torch.zeros((1), dtype=embedding.dtype, device=embedding.device)

        loss_dict = {}

        #view_omegas = extract_view_omegas_from_embedding(embedding, cfg.num_nodes)
        constants, scales, rotations, centers = convert_embedding_to_explicit_params(embedding, num_node) # batch_size, num_nodes ( () ,(3), (3,3), (3))
        #print(torch.mean(scales,dim=[0,1]))

        #duplicate the ball based on symmetry:
        # flip_axis : x:0, y:1,z:2
        #'''
        if cfg.use_mirror:
            flip_axis = 0
            constants_dup = constants[:, :num_node // 2].clone()
            scales_dup = scales[:, :num_node // 2, :].clone()
            # scales_dup[:,:,flip_axis] = -scales_dup[:,:,flip_axis]
            rotations_dup = rotations[:, :num_node // 2, :].clone()
            centers_dup = centers[:, :num_node // 2, :].clone()

            rotations_dup[:, :, flip_axis, :] = -rotations_dup[:, :, flip_axis, :]
            centers_dup[:, :, flip_axis] = -centers_dup[:, :, flip_axis]

            constants = torch.cat([constants, constants_dup], dim=1)
            scales = torch.cat([scales, scales_dup], dim=1)
            rotations = torch.cat([rotations, rotations_dup], dim=1)
            centers = torch.cat([centers, centers_dup], dim=1)

            num_node = num_node + num_node // 2
        #'''

        # calculate corner points for initialization
        left_upper = [-1, -1, -1]
        right_bottom = [1, 1, 1]
        corner_pts = self.calculate_3d_bounding_box_corners(left_upper, right_bottom, centers.shape[0]).to(
            centers.device)
        center_with_corner = torch.cat([corner_pts, centers], dim=1)

        # Create batch ellipsoids
        ellipsoids=[]




        # Uniform sampling loss.
        loss_uniform = None
        if cfg.lambda_sampling_uniform is not None and epoch>=cfg.warm_up_epoch:
            loss_uniform = self.point_loss(uniform_samples, constants, scales, rotations, centers,epoch)
            print("loss_uniform: "+str(loss_uniform))
            loss_total += cfg.lambda_sampling_uniform * loss_uniform

            loss_dict["loss_uniform"] = loss_uniform

        if cfg.lambda_sampling_uniform_ldif is not None and epoch>=cfg.warm_up_epoch:
            loss_uniform = self.ldif_recon_loss(uniform_samples,pred_sdf[:,-num_pts//2:,:],epoch)
            print("loss_uniform: "+str(loss_uniform))
            loss_total += cfg.lambda_sampling_uniform_ldif * loss_uniform

            loss_dict["loss_uniform"] = loss_uniform

        # Near surface sampling loss.
        loss_near_surface = None
        if cfg.lambda_sampling_near_surface is not None and epoch>=cfg.warm_up_epoch:
            hybid_samples = torch.cat([near_surface_samples,on_surface_samples],dim=1)
            loss_near_surface = self.point_loss(hybid_samples, constants, scales, rotations,centers,epoch)
            print("loss_nearsurface: " + str(loss_near_surface))
            loss_total += cfg.lambda_sampling_near_surface * loss_near_surface

            loss_dict["loss_near_surface"] = loss_near_surface

        if cfg.lambda_sampling_near_surface_ldif is not None and epoch>=cfg.warm_up_epoch:
            #hybid_samples = torch.cat([near_surface_samples,on_surface_samples],dim=1)
            loss_near_surface = self.ldif_recon_loss(near_surface_samples,pred_sdf[:,:num_pts//2,:] ,epoch)
            print("loss_nearsurface: " + str(loss_near_surface))
            loss_total += cfg.lambda_sampling_near_surface_ldif * loss_near_surface

            loss_dict["loss_near_surface"] = loss_near_surface

        # Node center loss.
        loss_node_center = None
        if cfg.lambda_sampling_node_center is not None and epoch>=cfg.warm_up_epoch:
            bbox_lower,bbox_upper =self.bounding_box(near_surface_samples[:,:,0:3])
            loss_node_center = self.node_center_loss(constants, scales, centers, grid,near_surface_pts,epoch)
            print("loss_node_center: " + str(loss_node_center))
            loss_total += cfg.lambda_sampling_node_center * loss_node_center

            loss_dict["loss_node_center"] = loss_node_center


        # Node Sparsity initial loss
        if cfg.lambda_sampling_node_sparse or cfg.lambda_bbox is not None:
            bbox_lower, bbox_upper = self.bounding_box(near_surface_samples[:, :, 0:3])

            loss_node_sparse = self.node_sparsity_loss(centers)
            bbox_error = bounding_box_error(centers, bbox_lower, bbox_upper)
            #if epoch<=100:
            #    loss_node_sparse *=10
            print("loss_node_sparse: " + str(loss_node_sparse))
            loss_total += cfg.lambda_sampling_node_sparse * loss_node_sparse + cfg.lambda_bbox*bbox_error.mean()


        # Node Sparisity Loss
        if cfg.lambda_sampling_node_sparse is not None:
            #loss_node_sparse = self.node_sparsity_loss(centers)
            #if epoch<=100:
            #    loss_node_sparse *=10
            pass
            #print("loss_node_sparse: " + str(loss_node_sparse))
            #loss_total += cfg.lambda_sampling_node_sparse * loss_node_sparse

            #loss_dict["loss_node_sparse"] = loss_node_sparse

        # Node Similiarity Loss
        if cfg.lambda_sampling_node_similiar is not None:
            loss_node_similiar = self.node_similiar_loss(constants,scales)
            #print("loss_node_similiar: " + str(loss_node_similiar))
            loss_total += cfg.lambda_sampling_node_similiar * loss_node_similiar
            #loss_dict["loss_node_similiar"] = loss_node_similiar

        # reconstrcution_loss
        if cfg.lambda_recon is not None and epoch>=100:
            print(cfg.lambda_recon)
            loss_recon,record = self.reconstruction_loss(pred_sdf,near_surface_pts,on_surface_pts,coords)
            loss_total +=cfg.lambda_recon*  loss_recon
            print("loss_recon: "+str(loss_recon))
            #loss_dict["loss_recon/total"] = loss_recon

            sdf_loss, off_surface_loss, normal_loss, grad_loss =record
            loss_dict["loss_recon/sdf"] = sdf_loss
            loss_dict["loss_recon/off"] = off_surface_loss
            loss_dict["loss_recon/normal"] = normal_loss
            loss_dict["loss_recon/grad"] = grad_loss

        if cfg.lambda_recon_g is not None and epoch>=cfg.warm_up_epoch:
            loss_recon,record = self.reconstruction_loss(ldif_gaussian,near_surface_pts,on_surface_pts,coords)
            loss_total += cfg.lambda_recon_g*  loss_recon
            #print("loss_recon: "+str(loss_recon))
        #print("loss_scale: " + str(torch.mean(torch.norm(scales,dim=-1))))
        #loss_total += torch.mean(torch.norm(scales,dim=-1))

        if cfg.lambda_center_var is not None and epoch>=cfg.warm_up_epoch:
            loss_center_var = self.center_var_loss(centers,epoch)
            loss_total += cfg.lambda_center_var *  loss_center_var
            #print("loss_center_var: "+str(loss_center_var))


        '''
        # Affinity loss.
        loss_affinity_rel = None
        loss_affinity_abs = None

        if (cfg.lambda_affinity_rel_dist is not None) or (cfg.lambda_affinity_abs_dist is not None):
            loss_affinity_rel, loss_affinity_abs = self.affinity_loss(centers, source_idxs, target_idxs, pred_distances,
                                                                      pair_weights)

            if cfg.lambda_affinity_rel_dist is not None: loss_total += cfg.lambda_affinity_rel_dist * loss_affinity_rel
            if cfg.lambda_affinity_abs_dist is not None: loss_total += cfg.lambda_affinity_abs_dist * loss_affinity_abs

        # Unique neighbor loss.
        loss_unique_neighbor = None
        if cfg.lambda_unique_neighbor is not None and affinity_matrix is not None:
            loss_unique_neighbor = self.unique_neighbor_loss(affinity_matrix)
            loss_total += cfg.lambda_unique_neighbor * loss_unique_neighbor

        # Viewpoint consistency loss.
        loss_viewpoint_position = None
        loss_viewpoint_scale = None
        loss_viewpoint_constant = None
        loss_viewpoint_rotation = None

        if (cfg.lambda_viewpoint_position is not None) or (cfg.lambda_viewpoint_scale is not None) or \
                (cfg.lambda_viewpoint_constant is not None) or (cfg.lambda_viewpoint_rotation is not None):
            loss_viewpoint_position, loss_viewpoint_scale, loss_viewpoint_constant, loss_viewpoint_rotation = \
                self.viewpoint_consistency_loss(constants, scales, rotations, centers)

            if cfg.lambda_viewpoint_position is not None:
                loss_total += cfg.lambda_viewpoint_position * loss_viewpoint_position
            if cfg.lambda_viewpoint_scale is not None:
                loss_total += cfg.lambda_viewpoint_scale * loss_viewpoint_scale
            if cfg.lambda_viewpoint_constant is not None:
                loss_total += cfg.lambda_viewpoint_constant * loss_viewpoint_constant
            if cfg.lambda_viewpoint_rotation is not None:
                loss_total += cfg.lambda_viewpoint_rotation * loss_viewpoint_rotation

        # Surface consistency loss.
        loss_surface_consistency = None
        if cfg.lambda_surface_consistency is not None:
            loss_surface_consistency = self.surface_consistency_loss(constants, scales, rotations, centers,
                                                                     surface_samples, grid, world2grid)
            loss_total += cfg.lambda_surface_consistency * loss_surface_consistency
        
        if evaluate:
            return loss_total, {
                "loss_uniform": loss_uniform,
                "loss_near_surface": loss_near_surface,
                "loss_node_center": loss_node_center,
                "loss_affinity_rel": loss_affinity_rel,
                "loss_affinity_abs": loss_affinity_abs,
                "loss_unique_neighbor": loss_unique_neighbor,
                "loss_viewpoint_position": loss_viewpoint_position,
                "loss_viewpoint_scale": loss_viewpoint_scale,
                "loss_viewpoint_constant": loss_viewpoint_constant,
                "loss_viewpoint_rotation": loss_viewpoint_rotation,
                "loss_surface_consistency": loss_surface_consistency
            }
            
        else:
            return loss_total
        '''
        return loss_total,loss_dict

class CenterVarLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,centers,epoch):
        # input: centers(batch,num_balls,3)
        variance = torch.var(centers, dim=[1, 2]) # batch
        loss_max = cfg.var_loss_max
        loss = torch.mean(
            torch.maximum(loss_max - variance, torch.zeros_like(variance)))
        return loss


class PointLoss(nn.Module):
    def __init__(self):
        super(PointLoss, self).__init__()

    def forward(self, points_with_sdf, constants, scales,rotations, centers,epoch):
        #print(rotations[0,0,:,:])
        batch_size = points_with_sdf.shape[0]

        #vis_data(points_with_sdf[0,:,:].detach().cpu().numpy())
        points = points_with_sdf[:, :, :3]
        is_outside = (points_with_sdf[:, :, 3] > 0)
        class_gt = is_outside.float()  # outside: 1, inside: 0


        # Evaluate predicted class at given points.
        sdf_pred = sample_rbf_surface(points, constants, scales,rotations, centers, cfg.use_constants,
                                      cfg.aggregate_coverage_with_max)

        class_pred = compute_inverse_occupancy(sdf_pred, cfg.soft_transfer_scale, cfg.level_set)

        #'''
        if epoch%cfg.vis_epoch == 0:
            import pyrender
            #num_pts = points.shape[1]//2
            xyz = points[0].detach().cpu().numpy()
            sdf = class_pred[0].detach().cpu().numpy()
            #colors = gt_normals[0,:num_pts].detach().cpu().numpy()
            colors = np.zeros(xyz.shape)
            colors[sdf <= 0.5, 2] = 1
            colors[sdf > 0.5, 0] = 1
            cloud = pyrender.Mesh.from_points(xyz, colors)

            scene = pyrender.Scene()
            scene.add(cloud)
            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=10)

        #'''
        # We apply weight scaling to interior points.
        weights = is_outside.float() + cfg.interior_point_weight * (~is_outside).float()

        # Compute weighted L2 loss.
        diff = class_gt - class_pred
        diff2 = diff * diff
        weighted_diff2 = weights * diff2

        loss = weighted_diff2.mean()
        return loss


class LDIF_Reconstruction_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points_with_sdf,ldif_sdf,epoch):
        #print(rotations[0,0,:,:])
        batch_size = points_with_sdf.shape[0]

        #vis_data(points_with_sdf[0,:,:].detach().cpu().numpy())
        points = points_with_sdf[:, :, :3]
        is_outside = (points_with_sdf[:, :, 3] > 0)
        class_gt = is_outside.float()  # outside: 1, inside: 0

        class_pred = compute_inverse_occupancy(ldif_sdf.squeeze(-1), cfg.soft_transfer_scale, cfg.level_set)

        #'''
        if False:
            import pyrender
            #num_pts = points.shape[1]//2
            xyz = points[0].detach().cpu().numpy()
            sdf = class_pred[0].detach().cpu().numpy()
            #colors = gt_normals[0,:num_pts].detach().cpu().numpy()
            colors = np.zeros(xyz.shape)
            colors[sdf <= 0.5, 2] = 1
            colors[sdf > 0.5, 0] = 1
            cloud = pyrender.Mesh.from_points(xyz, colors)

            scene = pyrender.Scene()
            scene.add(cloud)
            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=10)

        #'''
        # We apply weight scaling to interior points.
        weights = is_outside.float() + cfg.interior_point_weight * (~is_outside).float()

        # Compute weighted L2 loss.
        diff = class_gt - class_pred
        diff2 = diff * diff
        weighted_diff2 = weights * diff2

        loss = weighted_diff2.mean()
        return loss

class ReconstructionLoss(nn.Module):
    def __init__(self, type='sdf'):
        super().__init__()
        self.supervision = type
    def forward(self, preds,gt,coords, ns_pts=None):
        sdf_loss = torch.Tensor([0]).squeeze()
        spatial_loss = torch.Tensor([0]).squeeze()
        div_loss = torch.Tensor([0]).squeeze()
        normal_loss = torch.Tensor([0]).squeeze()
        grad_loss = torch.Tensor([0]).squeeze()
        bce_loss = torch.Tensor([0]).squeeze()
        off_surface_loss = torch.Tensor([0]).squeeze()


        if self.supervision == 'sdf':
            gt_sdf = ns_pts[:,:,3][:,:,None]
            gt_sdf = torch.clamp(gt_sdf,-0.1,0.1)
            #preds:batch, num_pts,1
            #gt_sdf: batch, num_pts,1
            batch_size = preds.shape[0]

            # preds = torch.clamp(preds, -0.1, 0.1)
            sdf_loss = torch.nn.functional.l1_loss(preds, gt_sdf, reduction='none')
            sdf_loss = torch.reshape(sdf_loss, (batch_size, -1)).mean(dim=-1).mean()

            total_loss = 1e1* sdf_loss

        else:
            batch_size = preds.shape[0]
            preds = preds.unsqueeze(-1)
            num_pts_sel = coords.shape[1]//2
            gt_sdf = torch.zeros_like(coords[:,:,:,0])[:,:,:,None]
            gt_sdf[:,num_pts_sel:,:,:] = -1
            gt_normals = gt[:,:,:,3:6]

            gradient = gradient_func(preds, coords)
            '''
            import pyrender
            on_pts = coords.reshape(batch_size,-1,3)
            on_pts_sdf = gt_sdf.reshape(batch_size,-1,1)
            #coords = coords.reshape(batch_size,-1,3)
            xyz = on_pts[0,:,:].detach().cpu().numpy()
            sdf = on_pts_sdf[0, :, 0].detach().cpu().numpy()
            sel_xyz = xyz[sdf != -1,:]
            xyz_n = gt_normals[0].reshape(-1,3).detach().cpu().numpy()
            sel_xyz_n = xyz_n[sdf!=-1,:]
            #colors = np.zeros(sel_xyz.shape)
            #colors[sdf == -1, 2] = 1
            #colors[sdf == 0.00, 0] = 1
            cloud = pyrender.Mesh.from_points(sel_xyz, sel_xyz_n)

            scene = pyrender.Scene()
            scene.add(cloud)
            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=10)
            '''

            # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.

            sdf_loss = torch.abs(torch.where(gt_sdf != -1, preds, torch.zeros_like(preds)))

            off_surface_loss = torch.where(gt_sdf != -1, torch.zeros_like(preds),
                                           torch.exp(-1e2 * torch.abs(preds)))

            normal_loss = torch.where(gt_sdf != -1,
                                      1 - torch.nn.functional.cosine_similarity(gradient, gt_normals, dim=-1)[
                                          ..., None],
                                      torch.zeros_like(gradient[..., :1]))



            grad_loss = torch.abs(gradient.norm(dim=-1,keepdim=True) - 1) #Eikonal Equation for points anywhere

            #sdf_loss = torch.reshape(sdf_loss, (batch_size, -1)).mean(dim=-1).mean()
            #off_surface_loss = torch.reshape(off_surface_loss, (batch_size, -1)).mean(dim=-1).mean()
            #normal_loss = torch.reshape(normal_loss, (batch_size, -1)).mean(dim=-1).mean()
            #grad_loss = torch.reshape(grad_loss, (batch_size, -1)).mean(dim=-1).mean()

            total_loss = 3e3 * sdf_loss  + 5e1 * grad_loss + 1e2 * normal_loss+ 1e2 * off_surface_loss #normal settings
            '''
            sdf_loss = torch.nn.functional.l1_loss(preds, gt_sdf, reduction='none')
            sdf_loss = torch.reshape(sdf_loss, (batch_size, -1)).mean(dim=-1).mean()

            total_loss = 1e1 * sdf_loss
            '''
            sdf_loss = torch.reshape(sdf_loss, (batch_size, -1)).mean(dim=-1).mean()
            off_surface_loss = torch.reshape(off_surface_loss, (batch_size, -1)).mean(dim=-1).mean()
            normal_loss = torch.reshape(normal_loss, (batch_size, -1)).mean(dim=-1).mean()
            grad_loss = torch.reshape(grad_loss, (batch_size, -1)).mean(dim=-1).mean()

        return total_loss, [sdf_loss,off_surface_loss.mean(),normal_loss,grad_loss]


class ReconstructionLoss_NEW(nn.Module):
    def __init__(self, type='sdf'):
        super().__init__()
        self.supervision = type
    def forward(self, preds,samples,coords):
        sdf_loss = torch.Tensor([0]).squeeze()
        spatial_loss = torch.Tensor([0]).squeeze()
        div_loss = torch.Tensor([0]).squeeze()
        normal_loss = torch.Tensor([0]).squeeze()
        grad_loss = torch.Tensor([0]).squeeze()
        bce_loss = torch.Tensor([0]).squeeze()
        off_surface_loss = torch.Tensor([0]).squeeze()


        batch_size = preds.shape[0]
        gt_sdf = samples[:, :, 3][:, :, None]
        gt_sdf = gt_sdf.clamp(-0.1, 0.1)
        gt_normals = samples[:,:,4:7]
        gradient = gradient_func(preds, coords)

        # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.

        sdf_loss = torch.nn.functional.l1_loss(preds, gt_sdf, reduction='none')

        normal_loss =1 - torch.nn.functional.cosine_similarity(gradient, gt_normals, dim=-1)[
                                      ..., None]

        total_loss = sdf_loss+  0.01 * normal_loss


        return total_loss, (sdf_loss,off_surface_loss,normal_loss,grad_loss)


class NodeCenterLoss(nn.Module):
    def __init__(self):
        super(NodeCenterLoss, self).__init__()

    def forward(self, constants, scales, centers, grid_orig,os_pts,epoch):
        batch_size,num_nodes = constants.shape
        num_pts = os_pts.shape[1]
        num_pts_grid =grid_orig.shape[1]
        #extract sdf from grid:
        #sdf = grid[:,3]
        #check_data
        #vis_data(grid_orig[0].detach().cpu().numpy())
        bbox_upper,_ = torch.max(os_pts[:,:num_pts//2,0:3],dim=1)
        bbox_lower,_ = torch.min(os_pts[:,:num_pts//2,0:3],dim=1)

        # Check if centers are inside the bounding box.
        # If not, we penalize them by using L2 distance to nearest bbox corner,
        # since there would be no SDF gradients there.
        bbox_error = bounding_box_error(centers, bbox_lower, bbox_upper)  # (bs, num_nodes)

        # Query SDF grid, to encourage centers to be inside the shape.
        # Convert center positions to grid CS.d

        centers_grid_cs = centers.view(batch_size, num_nodes, 3)
        points = centers_grid_cs[0,:,:].detach().cpu().numpy()  #num_nodes,3
        #A_world2grid = world2grid[:, :3, :3].view(batch_size, 1, 3, 3).expand(-1, cfg.num_nodes, -1, -1)
        #t_world2grid = world2grid[:, :3, 3].view(batch_size, 1, 3, 1).expand(-1, cfg.num_nodes, -1, -1)

        #centers_grid_cs = torch.matmul(A_world2grid, centers_grid_cs) + t_world2grid
        #centers_grid_cs = centers_grid_cs.view(batch_size, -1, 3) #batch_size, num_nodes, 3

        res=64
        grid = grid_orig[:,:,3].reshape(batch_size,res*res*res)
        # Sample signed distance field.
        #dim_z = grid.shape[1]
        #dim_y = grid.shape[2]
        #dim_x = grid.shape[3]
        #grid = grid.view(batch_size, 1, res,res,res).permute(0,1,4,2,3) # THIS IS THE KEY ! but why???
        grid = grid.view(batch_size, 1, res, res, res)
        # batch_size, 1, res_z+1, res_y+1, res_x+1



        #normalize the center_coordinate to grid_coordinate #batchsize,3
        bbox_scale = bbox_upper-bbox_lower
        bbox_center =  bbox_upper-0.5*bbox_scale

        centers_grid_cs -= bbox_center.reshape(batch_size,1,3).expand(-1,num_nodes,-1)
        centers_grid_cs /= (bbox_scale/2).reshape(batch_size,1,3).expand(-1,num_nodes,-1)

        centers_grid_cs = centers_grid_cs[:,:,[2,0,1]]
        centers_grid_cs = centers_grid_cs.view(batch_size, -1, 1, 1, 3)  #batch_size, num_nodes,1,1,3


        # We use border values for out-of-the-box queries, to have gradient zero at boundaries.
        '''
        #resolution=128
        #num_pts_grid=128**3
        #min_bound = np.array([-1, -1, -1])
        #max_bound = np.array([1, 1, 1])
        #xyz_range = np.linspace(min_bound, max_bound, num=resolution)

        #grid_pts = np.meshgrid(*xyz_range.T)
        #points_uniform = torch.Tensor(np.stack(grid_pts, axis=-1).astype(np.float32)).to(grid_orig.device)
        #grid_pts = points_uniform.reshape(1,-1,3).expand(batch_size,-1,-1)

        bbox_length = bbox_upper-bbox_lower
        bbox_center =  bbox_upper-0.5*bbox_length

        #for
        grid_points_tmp = os_pts[:,:,:3].clone()
        grid_points_tmp = grid_points_tmp - bbox_center.reshape(batch_size,1,3).expand(-1,num_pts,-1)
        grid_points_tmp /= (bbox_length/2).reshape(batch_size,1,3).expand(-1,num_pts,-1)
        #
        print(bbox_center)


        grid_points_tmp = grid_points_tmp[:, :, [2,0,1]]
        grid_points_tmp = grid_points_tmp.reshape(batch_size, -1, 1, 1, 3)
        '''

        centers_sdf_gt = torch.nn.functional.grid_sample((grid).double(), centers_grid_cs.double(),padding_mode="zeros",align_corners=True)
        #centers_sdf_gt = torch.nn.functional.grid_sample(grid.double(), grid_points_tmp.double(), padding_mode="zeros",align_corners=True)


        # If SDF value is higher than 0, we penalize it.
        centers_sdf_gt = centers_sdf_gt.view(batch_size, num_nodes)

        #centers_sdf_gt = centers_sdf_gt.view(batch_size, -1)


        if False:
            # visualize 0 shape
            import  pyrender

            sdf = centers_sdf_gt[0,:].detach().cpu().numpy()
            #points = os_pts[0,:,:3].detach().cpu().numpy().reshape(-1,3)
            #print(sdf)
            colors = np.zeros(points.shape)
            colors[sdf < 0.000, 2] = 1
            colors[sdf > 0.000, 0] = 1
            colors[sdf == 0.000, 1] = 1

            cloud = pyrender.Mesh.from_points(points, colors)
            scene = pyrender.Scene()
            scene.add(cloud)
            #viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=10)
            grid_orig_sdf = grid_orig[0,:,3].detach().cpu().numpy()
            grid = grid_orig[0,:,:3].detach().cpu().numpy()
            grid_points = grid[grid_orig_sdf<0]
            cloud = pyrender.Mesh.from_points(grid_points)
            scene.add(cloud)
            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=10)

        center_distance_error = torch.max(centers_sdf_gt, torch.zeros_like(centers_sdf_gt))  # (bs, num_nodes)


        # Final loss is just a sum of both losses.
        node_center_loss = 5*bbox_error+center_distance_error
        #node_center_loss = center_distance_error

        return torch.mean(node_center_loss)

class NodeSparsityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,centers):
        #centers.shape: (batch_size, num_nodes, 3)
        batch_size,num_nodes,_=centers.shape

        # Compute pairwise distances between points
        distances = torch.norm(centers[:, :, None, :] - centers[:, None, :, :], dim=-1)       #batch_size, num_nodes,num_nodes

        # Exclude the distance to self
        mask = (torch.eye(num_nodes, dtype=torch.bool).to(distances.device)).expand(batch_size, -1, -1)
        distances = distances.masked_fill(mask, float('inf'))

        # Find the index of the ne.1#arest neighbor for each point
        min_distances, min_distances_idx = torch.min(distances, dim=-1)  # batch_size, num_nodes

        # Extract the actual minimum distances
        #min_distances = torch.gather(distances, dim=-1, index=min_distances_idx.unsqueeze(-1)) # batch_size,num_nodes,1

        threshold_distance = 1.0
        # Compute regularization loss based on distances
        #loss_nearst_neighbour=torch.where(min_distances > threshold_distance, torch.zeros_like(distances), min_distances)
        #loss_nearst_neighbour = 0.5*torch.sum(loss_nearst_neighbour)
        #'''

        # Mask distances exceeding the threshold
        #mask_min_distances = torch.where(min_distances > threshold_distance, 1e8*torch.ones_like(min_distances), min_distances)

        # Compute mean distances to neighbors
        #mean_distances = torch.mean(distances, dim=-1)  # batch_size, num_nodes

        # Compute regularization loss based on distances
        loss_neighbour = torch.mean(0.5*threshold_distance*(torch.exp(-1/2/threshold_distance*min_distances) ))
        #loss_neighbour = torch.mean(min_distances)
        #'''
        loss = loss_neighbour
        return loss

class NodeSimiliarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,scale,radius):
        radius_weighted = 5 * radius # batch, num_nodes,3
        volume = torch.abs(radius_weighted[:,:,0]*radius_weighted[:,:,1]*radius_weighted[:,:,2])
        return self.ellipsoid_volume_loss(volume)# batch,num_nodes

    def ellipsoid_volume_loss(self, volumes):
        mean_volume = torch.mean(volumes,dim=-1,keepdim=True) # batch
        loss = torch.nn.functional.mse_loss(volumes, mean_volume * torch.ones_like(volumes)).mean()
        return loss



class AffinityLoss(nn.Module):
    def __init__(self):
        super(AffinityLoss, self).__init__()

    def forward(self, centers, source_idxs, target_idxs, pred_distances, pair_weights):
        batch_size = centers.shape[0]
        num_pairs = pred_distances.shape[1]

        loss_rel = 0.0
        loss_abs = 0.0

        if num_pairs > 0:
            source_positions = centers[:, source_idxs]
            target_positions = centers[:, target_idxs]

            diff = (source_positions - target_positions)
            dist2 = torch.sum(diff * diff, 2)  # (bs, num_pairs)

            abs_distance2 = pair_weights * dist2
            loss_abs = abs_distance2.mean()

            pred_distances2 = pred_distances * pred_distances
            pred_distances2 = pred_distances2  # (bs, num_pairs)

            weights_dist = pair_weights * torch.abs(pred_distances2 - dist2)  # (bs, num_pairs)
            loss_rel = weights_dist.mean()

        return loss_rel, loss_abs


class UniqueNeighborLoss(nn.Module):
    def __init__(self):
        super(UniqueNeighborLoss, self).__init__()

    def forward(self, affinity_matrix):
        assert affinity_matrix.shape[0] == cfg.num_neighbors and affinity_matrix.shape[1] == cfg.num_nodes and \
               affinity_matrix.shape[2] == cfg.num_nodes

        loss = 0.0

        for source_idx in range(cfg.num_neighbors):
            for target_idx in range(source_idx + 1, cfg.num_neighbors):
                affinity_source = affinity_matrix[source_idx].view(cfg.num_nodes, cfg.num_nodes)
                affinity_target = affinity_matrix[target_idx].view(cfg.num_nodes, cfg.num_nodes)

                # We want rows of different neighbors to be unique.
                affinity_dot = affinity_source * affinity_target
                affinity_dist = torch.sum(affinity_dot, dim=1)

                loss += affinity_dist.sum()

        # Normalize the loss by dividing with the number of pairs.
        num_pairs = (cfg.num_neighbors * (cfg.num_neighbors - 1)) / 2
        loss = loss / float(num_pairs)

        return loss


class ViewpointConsistencyLoss(nn.Module):
    def __init__(self):
        super(ViewpointConsistencyLoss, self).__init__()

    def forward(self, constants, scales, rotations, centers):
        batch_size = constants.shape[0]
        assert batch_size % 2 == 0

        # We expect every two consecutive samples are different viewpoints at same time step.
        loss_viewpoint_position = 0.0
        if cfg.lambda_viewpoint_position is not None:
            centers_pairs = centers.view(batch_size // 2, 2, cfg.num_nodes, -1)
            centers_diff = centers_pairs[:, 0, :, :] - centers_pairs[:, 1, :, :]
            centers_dist2 = centers_diff * centers_diff
            loss_viewpoint_position += centers_dist2.mean()

        loss_viewpoint_scale = 0.0
        if cfg.lambda_viewpoint_scale is not None:
            scales_pairs = scales.view(batch_size // 2, 2, cfg.num_nodes, -1)
            scales_diff = scales_pairs[:, 0, :, :] - scales_pairs[:, 1, :, :]
            scales_dist2 = scales_diff * scales_diff
            loss_viewpoint_scale += scales_dist2.mean()

        loss_viewpoint_constant = 0.0
        if cfg.lambda_viewpoint_constant is not None:
            constants_pairs = constants.view(batch_size // 2, 2, cfg.num_nodes, -1)
            constants_diff = constants_pairs[:, 0, :, :] - constants_pairs[:, 1, :, :]
            constants_dist2 = constants_diff * constants_diff
            loss_viewpoint_constant += constants_dist2.mean()

        loss_viewpoint_rotation = 0.0
        if cfg.lambda_viewpoint_rotation is not None:
            rotations_pairs = rotations.view(batch_size // 2, 2, cfg.num_nodes, 3, 3)
            rotations_diff = rotations_pairs[:, 0, :, :, :] - rotations_pairs[:, 1, :, :, :]
            rotations_dist2 = rotations_diff * rotations_diff
            loss_viewpoint_rotation += rotations_dist2.mean()

        return loss_viewpoint_position, loss_viewpoint_scale, loss_viewpoint_constant, loss_viewpoint_rotation


class SurfaceConsistencyLoss(nn.Module):
    def __init__(self):
        super(SurfaceConsistencyLoss, self).__init__()

    def forward(self, constants, scales, rotations, centers, surface_samples, grid, world2grid):
        batch_size = constants.shape[0]
        num_points = surface_samples.shape[1]

        loss = 0.0

        surface_points = surface_samples[:, :, :3]

        # Compute skinning weights for sampled points.
        skinning_weights = sample_rbf_weights(surface_points, constants, scales, centers,
                                              cfg.use_constants)  # (bs, num_points, num_nodes)

        # Compute loss for pairs of frames.
        for source_idx in range(batch_size):
            target_idx = source_idx + 1 if source_idx < batch_size - 1 else 0

            # Get source points and target grid.
            source_points = surface_points[source_idx]  # (num_points, 3)
            target_grid = grid[target_idx]  # (grid_dim, grid_dim, grid_dim)

            # Get source and target rotations.
            R_source = rotations[source_idx]  # (num_nodes, 3, 3)
            R_target = rotations[target_idx]  # (num_nodes, 3, 3)

            # Compute relative frame-to-frame rotation and translation estimates.
            t_source = centers[source_idx]
            t_target = centers[target_idx]

            R_source_inv = R_source.permute(0, 2, 1)
            R_rel = torch.matmul(R_target, R_source_inv)  # (num_nodes, 3, 3)

            # Get correspondending skinning weights and normalize them to sum up to 1.
            weights = skinning_weights[source_idx].view(num_points, cfg.num_nodes)
            weights_sum = weights.sum(dim=1, keepdim=True)
            weights = weights.div(weights_sum)

            # Apply deformation to sampled points.
            t_source = t_source.view(1, cfg.num_nodes, 3, 1).expand(num_points, -1, -1,
                                                                    -1)  # (num_points, num_nodes, 3, 1)
            t_target = t_target.view(1, cfg.num_nodes, 3, 1).expand(num_points, -1, -1,
                                                                    -1)  # (num_points, num_nodes, 3, 1)
            R_rel = R_rel.view(1, cfg.num_nodes, 3, 3).expand(num_points, -1, -1, -1)  # (num_points, num_nodes, 3, 3)
            source_points = source_points.view(num_points, 1, 3, 1).expand(-1, cfg.num_nodes, -1,
                                                                           -1)  # (num_points, num_nodes, 3, 1)
            weights = weights.view(num_points, cfg.num_nodes, 1, 1).expand(-1, -1, 3,
                                                                           -1)  # (num_points, num_nodes, 3, 1)

            transformed_points = torch.matmul(R_rel,
                                              (source_points - t_source)) + t_target  # (num_points, num_nodes, 3, 1)
            transformed_points = torch.sum(weights * transformed_points, dim=1).view(num_points, 3)

            # Convert transformed points to grid CS.
            transformed_points = transformed_points.view(num_points, 3, 1)
            A_world2grid = world2grid[target_idx, :3, :3].view(1, 3, 3).expand(num_points, -1, -1)
            t_world2grid = world2grid[target_idx, :3, 3].view(1, 3, 1).expand(num_points, -1, -1)

            transformed_points_grid_cs = torch.matmul(A_world2grid, transformed_points) + t_world2grid
            transformed_points_grid_cs = transformed_points_grid_cs.view(num_points, 3)

            # Sample signed distance field.
            dim_z = target_grid.shape[0]
            dim_y = target_grid.shape[1]
            dim_x = target_grid.shape[2]
            target_grid = target_grid.view(1, 1, dim_z, dim_y, dim_x)

            transformed_points_grid_cs[..., 0] /= float(dim_x - 1)
            transformed_points_grid_cs[..., 1] /= float(dim_y - 1)
            transformed_points_grid_cs[..., 2] /= float(dim_z - 1)
            transformed_points_grid_cs = 2.0 * transformed_points_grid_cs - 1.0
            transformed_points_grid_cs = transformed_points_grid_cs.view(1, -1, 1, 1, 3)

            # We use border values for out-of-the-box queries, to have gradient zero at boundaries.
            transformed_points_sdf_gt = torch.nn.functional.grid_sample(target_grid, transformed_points_grid_cs,
                                                                        align_corners=True, padding_mode="border")

            # If SDF value is different than 0, we penalize it.
            transformed_points_sdf_gt = transformed_points_sdf_gt.view(num_points)
            df_error = torch.mean(transformed_points_sdf_gt * transformed_points_sdf_gt)

            loss += df_error

        return loss



class PatchNetLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ext_loss = ExtLoss()
        self.recon_loss = ReconLoss()
    def forward(self):
        loss_total = 0
        loss_ext_loss = ExtLoss()
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


from models.node_proc import global_to_local_with_normal
class SurLoss(torch.nn.Module):
    # the center of every patch should at least cover one point!
    def __init__(self):
        super().__init__()
        self.free_space_distance_threshold = 0.06

    def forward(self,scaled_distances,scales):
        # coords_sel: batch, num_pts_sel, num_nodes, 3
        # batch_ext: batch, num_nodes, 10
        # unscaled_distances: batch, num_pts_sel, num_nodes
        pass
        '''calculation of 1/Np * max(min(distances),0.06)'''
        closest_surface_distances, closest_surface_indices = torch.min(scaled_distances,dim=1) #batch, num_nodes
        free_space_patches = closest_surface_distances > self.free_space_distance_threshold
        closest_surface_distances[~free_space_patches] = 0.
        free_space_scene_normalization = torch.sum(free_space_patches, dim=1)  # batch

        '''Sum across num_nodes'''
        free_space_scenes = free_space_scene_normalization > 0
        eps = 0.001
        free_space_scene_losses = torch.sum(closest_surface_distances[free_space_scenes, :], dim=1) / (
                    free_space_scene_normalization[free_space_scenes].float() + eps)  # num_scenes

        '''sum across scenes'''
        free_space_loss = torch.sum(free_space_scene_losses) / (torch.sum(free_space_scenes) + eps)
        return free_space_loss


class CovLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,scaled_center_distances,ext):
        # ...distances: batch, num_pts_sel, num_nodes
        # scales: batch, num_pts_sel, num_nodes, 3
        num_pts = scaled_center_distances.shape[1]
        print(ext)
        free_space_distance_threshold = ext.unsqueeze(1).expand(-1,num_pts,-1)

        #free_space_distance_threshold = 0.15
        pull_std = 0.05 #cant be too small otherwise gaussians get to zero
        eps = 0.0001

        uncoverd_pts_mask = scaled_center_distances >= free_space_distance_threshold  #batch, num_pts_sel, num_nodes


        pull_distances = (scaled_center_distances - free_space_distance_threshold) #batch, num_pts_sel, num_nodes
        pull_distances = torch.clamp(pull_distances,min=0)

        #pull_distances[~uncoverd_pts_mask] =0
        print(pull_distances[0, 0, :])

        pull_weight = torch.exp(-0.5 * (pull_distances/pull_std) ** 2) /pull_std  # batch, num_pts_sel,num_nodes
        pull_weight *= uncoverd_pts_mask

        print(pull_weight[0, 0, :])
        print("weight_sum:" + str(torch.sum(pull_weight[0])))

        #pull_weight = torch.clamp(pull_weight,max=0.999)
        #print("weight_sum:" + str(torch.sum(pull_weight[0])))

        # Wrong normalization #
        pull_weight_normalization = torch.sum(pull_weight,dim=-1) # batch, numpts_sel
        print("mask_sum:"+str(pull_weight_normalization[0]))


        mask,_ = torch.min(uncoverd_pts_mask,dim=-1) #batch, numpts_sel  #0 means already coverd
        print(num_pts)
        #print("mask_sum:"+str(torch.count_nonzero(mask[0])))#number of uncovered pts
        #norm_mask = pull_weight_normalization == 0.

        norm_mask = mask == 0.
        pull_weight_normalized = torch.zeros_like(pull_weight)
        pull_weight_normalized[~norm_mask,:] = pull_weight[~norm_mask,:] / (pull_weight_normalization[~norm_mask].unsqueeze(-1))
        #print(pull_weight[0, 0, :])
        weighted_pulls = pull_distances * pull_weight_normalized
        #weighted_pulls[~uncoverd_pts_mask] = 0

        weighted_pulls = torch.sum(weighted_pulls,dim=1) # batch,num_balls


        norm_mask = torch.sum(uncoverd_pts_mask,dim=[1,2])  # batch
        norm_scenes_mask = norm_mask > 0
        coverage_scene_losses = torch.sum(weighted_pulls[norm_scenes_mask, :], dim=1) / (
                    norm_mask[norm_scenes_mask].float() + eps)  # num_scenes
        coverage_loss = torch.sum(coverage_scene_losses) / (torch.sum(norm_scenes_mask) + eps)

        return coverage_loss



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