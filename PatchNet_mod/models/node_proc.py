import torch.nn as nn
import torch
import math
from scipy.spatial.transform import Rotation

NORMALIZATION_EPS = 1e-8


def extract_view_omegas_from_embedding(embedding, num_nodes):
    batch_size = embedding.shape[0]

    embedding = embedding.view(batch_size, num_nodes, 10)
    omegas = embedding[:, :, 7:10]

    return omegas

import config as cfg
from train_patchnet import  convert_embedding_to_explicit_params


def batch_angle_axis_to_rotation_matrix(angle, axes):
    """
    Convert a batch of angle-axis representations to rotation matrices.

    Parameters:
    - angles (torch.Tensor): Tensor of rotation angles in degrees.
    - axes (torch.Tensor): Tensor of rotation axes as 3-element tensors.

    Returns:
    - torch.Tensor: Tensor of 3x3 rotation matrices.
    """
    B_num_node = angle.shape[0]
    #angle = torch.deg2rad(angles)                                   #shape = B*num_node,1
    axes = axes / axes.norm(dim=1, keepdim=True)  # Normalize axes   shape= B*num_node,3

    # Element-wise operations to compute rotation matrices
    c = torch.cos(angle)
    s = torch.sin(angle)
    t = 1 - c
    x, y, z =axes[:,:,None].permute(1,0,2) # axes.shape = (4,3,1)  x.shape=(4,)
    rotation_matrices = torch.stack([
        t*x*x + c, t*x*y - s*z, t*x*z + s*y,
        t*x*y + s*z, t*y*y + c, t*y*z - s*x,
        t*x*z - s*y, t*y*z + s*x, t*z*z + c
    ]).view(3,3,B_num_node).permute(2,0,1)  # B*num_node,3,3

    return rotation_matrices


def compute_inverse_occupancy(vals, soft_transfer_scale, level_set):
    # vals are negative everywhere, with higher negative values inside the object,
    # slowly decaying to zero outside.
    return torch.sigmoid(soft_transfer_scale * (vals - level_set))


def global_to_local(points, scales, rotations,centers):
    batch_size = points.shape[0]
    num_points = points.shape[1]
    assert points.shape[2] == 3
    num_nodes = centers.shape[1]
    # Compute centered points.
    points = points.view(batch_size, num_points, 1, 3).expand(-1, -1, num_nodes, -1)  # (bs, num_points, num_nodes,3
    centers = centers.view(batch_size, 1, num_nodes, 3).expand(-1, num_points, -1, -1)  # (bs, num_points, num_nodes,3
    delta = points - centers  # (bs, num_points, num_nodes, 3)
    # rotation!
    #'''
    rotations = rotations.permute(0,1,3,2)[:,None].expand(-1,num_points,-1,-1,-1).double()  # batch_size,num_pts,n
    delta = delta.reshape(-1,3)[...,None].double()                                                   # batch_size*num_pts*n
    rotations = rotations.reshape(-1,3,3)                                                   # batch_size*num_pts*n
    delta = torch.matmul(rotations,delta)                                                   # batch_sie*num_pts*nu
    delta = delta.reshape(batch_size,num_points,num_nodes,3)
    #'''
    #delta2 = delta * delta  # (bs, num_points, num_nodes, 3)
    # Add anisotropic scaling.
    # For numerical stability we use at least eps.
    scales = torch.max(scales, NORMALIZATION_EPS * torch.ones_like(scales))  # (bs, num_nodes, 3)
    inv_scales = 1.0 / scales
    inv_scales = inv_scales.view(batch_size, 1, num_nodes, 3).expand(-1, num_points, -1,
                                                                       -1)  # (bs, num_points, num_nodes, 3)

    delta_length_scaled = delta *inv_scales# (bs, num_points, num_nodes,3)
    #delta_length_scaled = delta2.float()
    return  delta_length_scaled.float()




def global_to_local_with_normal(points, scales, rotations,centers):
    batch_size = points.shape[0]
    num_points = points.shape[1]
    assert points.shape[2] == 6 or points.shape[2] == 3
    num_nodes = centers.shape[1]
    # Compute centered points.
    compute_normal = points.shape[2]==6


    xyz = points[:,:,:3]
    xyz = xyz.reshape(batch_size, num_points, 1, 3).expand(-1, -1, num_nodes, -1)  # (bs, num_points, num_nodes,3
    centers = centers.reshape(batch_size, 1, num_nodes, 3).expand(-1, num_points, -1, -1)  # (bs, num_points, num_nodes,3
    delta = xyz - centers  # (bs, num_points, num_nodes, 3)
    # rotation!
    #'''
    rotations = rotations.permute(0,1,3,2)[:,None].expand(-1,num_points,-1,-1,-1).double()  # batch_size,num_pts,n
    delta = delta.reshape(-1,3)[...,None].double()                                                   # batch_size*num_pts*n
    rotations = rotations.reshape(-1,3,3)                                                   # batch_size*num_pts*n
    delta = torch.matmul(rotations,delta)                                                   # batch_sie*num_pts*nu
    delta = delta.reshape(batch_size,num_points,num_nodes,3)


    #'''
    #delta2 = delta * delta  # (bs, num_points, num_nodes, 3)
    # Add anisotropic scaling.
    # For numerical stability we use at least eps.
    scales = torch.max(scales, NORMALIZATION_EPS * torch.ones_like(scales))  # (bs, num_nodes, 3)
    inv_scales = 1.0 / scales
    inv_scales = inv_scales.reshape(batch_size, 1, num_nodes, 3).expand(-1, num_points, -1, -1)  # (bs, num_points, num_nodes, 3)

    delta_length_scaled = delta*inv_scales# (bs, num_points, num_nodes,3)

    # for normal, only rotation
    if compute_normal:
        normal = points[:,:,3:6]
        normal = normal.reshape(batch_size, num_points, 1, 3).expand(-1, -1, num_nodes, -1)          # batch_size*num_pts*n,3,1
        normal = normal.reshape(-1, 3)[..., None].double()
        rotations = rotations.reshape(-1,3,3)                                                   # batch_size*num_pts*n
        normal = torch.matmul(rotations,normal)                                                   # batch_sie*num_pts*nu
        normal = normal.reshape(batch_size,num_points,num_nodes,3)
        delta_length_scaled = torch.cat([delta_length_scaled,normal],dim=-1)

    return  delta_length_scaled.float()


def local_to_global_with_normal(points, scales, rotations,centers):
    batch_size = points.shape[0]
    num_points = points.shape[1]
    num_nodes = points.shape[2]
    compute_normal =  points.shape[3] == 6
    #num_nodes = centers.shape[1]

    scales = torch.max(scales, NORMALIZATION_EPS * torch.ones_like(scales))  # (bs, num_nodes, 3)
    scales = scales.reshape(batch_size, 1, num_nodes, 3).expand(-1, num_points, -1, -1)  # (bs, num_points, num_nodes, 3)



    # Compute centered points.
    xyz = points[:,:,:,:3]

    xyz = xyz *scales# (bs, num_points, num_nodes,3)
    # (bs, num_points, num_nodes,3


    centers = centers.reshape(batch_size, 1, num_nodes, 3).expand(-1, num_points, -1, -1)  # (bs, num_points, num_nodes,3

    # rotation!
    #'''
    rotations = rotations[:,None].expand(-1,num_points,-1,-1,-1).double()  # batch_size,num_pts,n
    xyz = xyz.reshape(-1,3)[...,None].double()                                                   # batch_size*num_pts*n
    rotations = rotations.reshape(-1,3,3)                                                   # batch_size*num_pts*n
    xyz = torch.matmul(rotations,xyz)                                                   # batch_sie*num_pts*nu
    xyz = xyz.reshape(batch_size,num_points,num_nodes,3)

    delta = xyz+centers

    # for normal, only rotation
    #normal = normal.reshape(batch_size, num_points, 1, 3)          # batch_size*num_pts*n,3,1
    if compute_normal:
        normal = points[:,:,:,3:6]
        normal = normal.reshape(-1, 3)[..., None].double()

        normal = torch.matmul(rotations,normal)                                                   # batch_sie*num_pts*nu
        normal = normal.reshape(batch_size,num_points,num_nodes,3)

        #'''
        #delta2 = delta * delta  # (bs, num_points, num_nodes, 3)
        # Add anisotropic scaling.
        # For numerical stability we use at least eps.

        delta = torch.cat([delta,normal],dim=-1)
    return  delta.float()



def sample_rbf_weights(points, constants, scales, rotations,centers, use_constants):
    batch_size = points.shape[0]
    num_points = points.shape[1]
    #assert points.shape[2] == 3

    num_nodes = centers.shape[1]

    # Compute centered points.
    #points = points.view(batch_size, num_points, 1, 3).expand(-1, -1, num_nodes, -1)  # (bs, num_points, num_nodes, 3)
    centers = centers.view(batch_size, 1, num_nodes, 3).expand(-1, num_points, -1, -1)  # (bs, num_points, num_nodes, 3)


    delta = points - centers  # (bs, num_points, num_nodes, 3)
    # rotation!
    #'''
    rotations = rotations.permute(0,1,3,2)[:,None].expand(-1,num_points,-1,-1,-1).double()  # batch_size,num_pts,num_nodes,3,3
    delta = delta.reshape(-1,3)[...,None].double()                                                   # batch_size*num_pts*num_nodes,3,1
    rotations = rotations.reshape(-1,3,3)                                                   # batch_size*num_pts*num_nodes,3,3
    delta = torch.matmul(rotations,delta)                                                   # batch_sie*num_pts*num_nodes,3
    delta = delta.reshape(batch_size,num_points,num_nodes,3)
    #'''


    # Add anisotropic scaling.
    # For numerical stability we use at least eps.
    scales = torch.max(scales, NORMALIZATION_EPS * torch.ones_like(scales))  # (bs, num_nodes, 3)

    inv_scales = 1.0 / scales
    inv_scales = inv_scales.view(batch_size, 1, num_nodes, 3).expand(-1, num_points, -1,
                                                                       -1)  # (bs, num_points, num_nodes, 3)



    delta_length_scaled = torch.norm(inv_scales * delta, dim=-1)  # (bs, num_points, num_nodes)
    delta_length_unscaled = torch.norm(delta,dim=-1) #bs, num_pts, num_nodes
    #apply truncation:
    mask = delta_length_scaled < 1.

    weight_vals = torch.zeros_like(delta_length_scaled)
    std_cutoff= torch.Tensor([3]).to(delta_length_scaled.device)

    # Apply scaled exponential kernel.
    if use_constants:
        constants = constants.view(batch_size, 1, num_nodes).expand(-1, num_points, -1)  # (bs, num_points, num_nodes)
        weight_vals[mask] = (torch.exp(-0.5 * (delta_length_scaled[mask]*std_cutoff)**2 ) -torch.exp(-0.5*std_cutoff**2))
        weight_vals[~mask]=0
        weight_vals = constants*weight_vals
    else:
        weight_vals[mask] = (torch.exp(-0.5 * (delta_length_scaled[mask]*std_cutoff)**2 ) -torch.exp(-0.5*std_cutoff**2))
        #weight_vals[mask] = (torch.exp(-0.5 * (delta_length_scaled[mask] * std_cutoff) ** 2) )
        weight_vals[~mask] = 0

    return weight_vals.view(batch_size, num_points, num_nodes),delta_length_unscaled



def sample_rbf_surface(points, constants, scales, rotations,centers, use_constants, aggregate_coverage_with_max):
    batch_size = points.shape[0]
    num_points = points.shape[1]

    # Sum the contributions of all kernels.
    weights_vals = sample_rbf_weights(points, constants, scales,rotations, centers, use_constants)  # (bs, num_points, num_nodes)

    if aggregate_coverage_with_max:
        sdf_vals, _ = torch.min(-weights_vals, 2)  # (bs, num_points)
    else:
        sdf_vals = torch.sum(-weights_vals, axis=2)  # (bs, num_points)

    return sdf_vals.view(batch_size, num_points)


def bounding_box_error(points, bbox_lower, bbox_upper):
    #points->center of nodes: BS,num_nodes,3

    batch_size = points.shape[0]
    num_points = points.shape[1]
    assert points.shape[2] == 3


    bbox_lower_vec = bbox_lower.view(batch_size, 1, 3).expand(-1, num_points, -1)  # (bs, num_points, 3)
    bbox_upper_vec = bbox_upper.view(batch_size, 1, 3).expand(-1, num_points, -1)  # (bs, num_points, 3)

    lower_error = torch.max(bbox_lower_vec - points, torch.zeros_like(points))
    upper_error = torch.max(points - bbox_upper_vec, torch.zeros_like(points))

    constraint_error = torch.sum(lower_error * lower_error + upper_error * upper_error, axis=2)  # (bs, num_points)
    return constraint_error