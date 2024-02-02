"""From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
"""
#!/usr/bin/env python3

import logging
import os
import time

import numpy as np
import plyfile
import skimage.measure
import torch



def create_meshes(
    decoder,
    batch_ext,
    batch_latent,
    type,
    filename=None,
    N=256,
    max_batch=1024,
    offset=None,
    scale=None,
    level=0,
):
    #especailly for hyponet visualization

    batch = batch_ext.shape[0]
    start = time.time()
    ply_filename = filename
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1] * 3
    voxel_size = -2 * voxel_origin[0] / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    samples = samples[np.newaxis, :, :].expand(batch,-1,-1).clone()
    samples.requires_grad = False

    head = 0
    from models.node_proc import convert_embedding_to_explicit_params


    while head < num_samples:
        # print(head)
        sample_subset = samples[:,head: min(head + max_batch, num_samples), 0:3].cuda()
        if type == 'ldif':
            tmp = decoder(sample_subset,batch_ext,batch_latent,type='evaluate')[0]
            samples[:,head : min(head + max_batch, num_samples), 3] = (
                tmp.squeeze(-1).clone().detach().cpu()# .squeeze(1)
            )
        elif type == 'guassians':
            _,_,tmp= decoder(sample_subset,batch_ext,batch_latent)
            samples[:,head : min(head + max_batch, num_samples), 3] = (
                tmp.squeeze(-1).clone().detach().cpu()# .squeeze(1)
            )


        head += max_batch

    meshes=[]
    tmp = {}


    for i in range(batch):
        sdf_values = samples[i,:, 3]
        sdf_values = sdf_values.reshape(N, N, N)
        tmp['vertices'], tmp['faces'],_ = convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            None if ply_filename is None else ply_filename + ".ply",
            level,
            offset,
            scale,
        )
        meshes.append(tmp)
        tmp={}

    end = time.time()
    print("sampling takes: %f" % (end - start))

    return meshes


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    level,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    # v, f = mcubes.marching_cubes(mcubes.smooth(pytorch_3d_sdf_tensor.numpy()), 0)
    # mcubes.export_obj(v, f, ply_filename_out.split(".")[-2][1:] + ".obj")
    # return
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    # print(numpy_3d_sdf_tensor.min(), numpy_3d_sdf_tensor.max())
    verts, faces, normals, values = (
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros(0),
    )
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
    except Exception as e:
        print(e)
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset


    return mesh_points, faces, pytorch_3d_sdf_tensor
