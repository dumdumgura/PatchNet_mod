import numpy as np
import open3d as o3d
import torch

def to_homogeneous(points, is_point=True):
    # Add an additional dimension for homogeneous coordinates
    ones = torch.ones((*points.shape[:-1], 1), device=points.device, dtype=points.dtype)
    if is_point:
        return torch.cat([points, ones], dim=-1)
    else:
        return torch.cat([points, torch.zeros_like(ones)], dim=-1)

def local_views_of_shape(global_points, world2local, local_point_count, global_normals=None, global_features=None, zeros_invalid=False, zero_threshold=1e-6, expand_region=True, threshold=4.0):
    batch_size = global_points.shape[0]
    frame_count = world2local.shape[1]

    if zeros_invalid:
        is_zero = torch.abs(global_points) < zero_threshold
        is_zero = torch.all(is_zero, dim=-1, keepdim=True)
        global_points = torch.where(is_zero, torch.full_like(global_points, 100.0), global_points)

    #local2world = torch.inverse(world2local)

    tiled_global = to_homogeneous(global_points, is_point=True).unsqueeze(1).expand(-1, frame_count, -1, -1)
    all_local_points = torch.matmul(tiled_global, world2local.transpose(-1, -2))
    distances = torch.norm(all_local_points[..., :3], p=2, dim=-1)

    probabilities = torch.rand(distances.shape, device=global_points.device)
    is_valid = distances < threshold

    sample_order = torch.where(is_valid, probabilities, -distances)
    _, top_indices = torch.topk(sample_order, k=local_point_count, dim=2, largest=True, sorted=False)

    # gather_dim = len(global_points.shape) - 2
    # local_points = torch.gather(all_local_points, gather_dim, top_indices.unsqueeze(-1).expand(-1, -1, -1, 4))[..., :3]

    gather_dim = len(global_points.shape) - 2  # This is the dimension along which we're gathering

    # Before gathering, ensure that 'top_indices' are within the bounds of 'all_local_points'
    # Note: Adjust the dimension size based on your specific tensor shapes
    adjusted_top_indices = top_indices % all_local_points.shape[gather_dim]

    local_points = torch.gather(all_local_points, gather_dim, adjusted_top_indices.unsqueeze(-1).expand(-1, -1, -1, 4))[..., :3]

    
    # points_valid = torch.gather(is_valid.unsqueeze(-1), gather_dim, top_indices.unsqueeze(-1).expand(-1, -1, -1, 1))

    adjusted_top_indices = top_indices % is_valid.shape[1]

    # Use the adjusted indices for gathering
    points_valid = torch.gather(is_valid.unsqueeze(-1), gather_dim, adjusted_top_indices.unsqueeze(-1).expand(-1, -1, -1, 1))


    if not expand_region:
        local_points = torch.where(points_valid, local_points, torch.zeros_like(local_points))

    local_normals = None
    if global_normals is not None:
        tiled_global_normals = to_homogeneous(global_normals, is_point=False).unsqueeze(1).expand(-1, frame_count, -1, -1)
        all_local_normals = torch.matmul(tiled_global_normals, local2world)
        local_normals = torch.gather(all_local_normals, gather_dim, top_indices.unsqueeze(-1).expand(-1, -1, -1, 4))[..., :3]
        local_normals = torch.nn.functional.normalize(local_normals, dim=-1)

    local_features = None
    if global_features is not None:
        local_features = torch.gather(global_features, gather_dim, top_indices.unsqueeze(-1).expand(-1, -1, -1, global_features.shape[-1]))

    return local_points, local_normals, local_features, points_valid

# Example usage:
# global_points = torch.rand(batch_size, global_point_count, 3)
# world2local = torch.rand(batch_size, frame_count, 4, 4)
# local_point_count = 1000
# local_views = local_views_of_shape(global_points, world2local, local_point_count)



# Example parameters
batch_size = 4         # Number of samples in a batch
global_point_count = 1000 # Number of points in the global point cloud
node_num = 10       # Number of local frames



local_point_count = 100 # Number of points in each local view

# Generate synthetic global points
global_points = torch.rand(batch_size, global_point_count, 3)*5   # Random points in a 10x10x10 space


print("Global Points Shape:", global_points[0][0])


point_cloud_1 = np.array(global_points[0])
point_cloud_2 = np.array(global_points[1])
point_cloud_3 = np.array(global_points[2])
point_cloud_4 = np.array(global_points[3])

point_cloud_1_1 = o3d.geometry.PointCloud()
point_cloud_1_1.points = o3d.utility.Vector3dVector(point_cloud_1)
point_cloud_1_1.paint_uniform_color([1, 0.706, 0])

point_cloud_2_2 = o3d.geometry.PointCloud()
point_cloud_2_2.points = o3d.utility.Vector3dVector(point_cloud_2)
point_cloud_2_2.paint_uniform_color([0, 0.651, 0.929])

point_cloud_3_3 = o3d.geometry.PointCloud()
point_cloud_3_3.points = o3d.utility.Vector3dVector(point_cloud_3)
point_cloud_3_3.paint_uniform_color([0.992, 0.153, 0.043])

point_cloud_4_4 = o3d.geometry.PointCloud()
point_cloud_4_4.points = o3d.utility.Vector3dVector(point_cloud_4)
point_cloud_4_4.paint_uniform_color([0.82, 0.373, 0.607])




print("Global Points Shape:", global_points.shape)

# Generate random transformation matrices for world to local space conversion
# For simplicity, these are random matrices. In a real scenario, these would be meaningful transformations.
world2local = torch.diag(torch.tensor([1,1,1,1])).float()
world2local = world2local.unsqueeze(0).unsqueeze(1).expand(4,1,-1,-1)
#world2local[:, :, -1] = torch.tensor([0, 0, 0, 1])  # Ensuring the last row is [0, 0, 0, 1] for valid transformation matrices


# visualize the world2local






# Call the function
local_points, local_normals, local_features, points_valid = local_views_of_shape(
    global_points, 
    world2local, 
    local_point_count
)

# local_points now contains the local views
print("Local Points Shape:", local_points.shape)


# visualize the local points

print("Local Points Shape:", local_points[0][0])

local_points_1 = np.array(local_points[0][0])

local_points_1_1 = o3d.geometry.PointCloud()
local_points_1_1.points = o3d.utility.Vector3dVector(local_points_1)
local_points_1_1.paint_uniform_color([1, 0.706, 0])

o3d.visualization.draw_geometries([point_cloud_1_1,point_cloud_2_2,point_cloud_3_3,point_cloud_4_4,local_points_1_1])




# visualize the local points, global points and local frames
