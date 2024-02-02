import open3d as o3d
import numpy as np
'''
num_points =300
# Create a single point with normal vector
points = np.random.rand(num_points, 3)

# Generate random normal vectors (unit vectors)
normals = np.random.rand(num_points, 3)
normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize to unit length

# Create Open3D PointCloud
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)
cloud.normals = o3d.utility.Vector3dVector(normals)


# Function to create a sphere
def create_sphere_at_position(position, radius=1.0, resolution=30):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    mesh.translate(position)
    return mesh

# Create spheres at each specified point
spheres = []
for point in points:
    sphere = create_sphere_at_position(point, radius=0.1, resolution=10)
    spheres.append(sphere)

# Visualize the sphere
o3d.visualization.draw_geometries([sphere])

# Visualize the point with normal vector
o3d.visualization.draw_geometries([cloud], point_show_normal=True)
'''
import torch

def _get_euler_angles_from_rotation_matrix(rotation_matrix):
    # ZYX rotation (YPR)
    if np.abs(rotation_matrix[2,0]) != 1.:
        beta = -np.arcsin(rotation_matrix[2,0])
        return np.array([np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2]), beta, np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])])

		# Solution not unique, this is the second one
		#const float beta = PI+asin(R.m31); const float cosBeta = cos(beta);
		#return make_float3(atan2(R.m32/cosBeta, R.m33/cosBeta), beta, atan2(R.m21/cosBeta, R.m11/cosBeta));
    else:
        if rotation_matrix[2,0] == -1.0:
            return np.array([np.arctan2(rotation_matrix[0,1], rotation_matrix[0,2]), np.pi/2., 0.])
        else:
            return np.array([np.arctan2(-rotation_matrix[0,1], -rotation_matrix[0,2]), -np.pi/2., 0.])


def _get_rotation_from_euler(euler_angles):
    roll,yaw,pitch = euler_angles
    """
    Convert Euler angles to a rotation matrix.

    Parameters:
    - roll (tensor): Roll angle in radians.
    - pitch (tensor): Pitch angle in radians.
    - yaw (tensor): Yaw angle in radians.

    Returns:
    - rotation_matrix (tensor): 3x3 rotation matrix.
    """
    # Calculate individual rotation matrices
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(roll), -np.sin(roll)],
                           [0, np.sin(roll), np.cos(roll)]])

    rotation_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                           [0, 1, 0],
                           [-np.sin(pitch), 0, np.cos(pitch)]])

    rotation_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0],
                           [0, 0, 1]])

    # Combine the individual rotation matrices
    rotation_matrix = np.dot(rotation_x, np.dot(rotation_y, rotation_z))


    return rotation_matrix

    pass


rotation = np.array([[0,1,0],
                         [0,0,1],
                         [1,0,0]])

euler_angles = _get_euler_angles_from_rotation_matrix(rotation)
rotation_recover = _get_rotation_from_euler(euler_angles)
print(rotation)
print(rotation_recover)
