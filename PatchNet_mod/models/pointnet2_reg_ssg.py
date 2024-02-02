import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer,mini_PointNetEncoder
import pyrender



class PatchNetDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_layers=3, hidden_dim=128, dropout_prob=0.2):
        super().__init__()

        # Define model
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Define layers
        self.decoder_former = self._build_layers('former')
        self.decoder_latter = self._build_layers('latter')

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

    def _build_layers(self,pos):
        layers = []
        if pos == 'former':
            in_dim = self.latent_dim + 3    #259
        else:
            in_dim = self.hidden_dim + self.latent_dim +3
        for idx in range(self.num_layers):
            layers.append(nn.Sequential(
                #nn.utils.weight_norm()
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU()
            ))
            in_dim = self.hidden_dim

        return nn.Sequential(*layers)

    def forward(self, x_in):
        # Former layers
        x_con_in = self.decoder_former(x_in)

        # Latter layers with skip connection
        x = torch.cat([x_con_in, x_in], dim=-1)
        x = self.decoder_latter(x)

        # Output layer
        x = self.output_layer(x)

        return x
def vis_data(pts):
    import pyrender
    xyz = pts[0,:, 0:3]
    sdf = pts[0,:, 3]
    colors = np.zeros(xyz.shape)
    colors[sdf <= 0.00, 2] = 1
    colors[sdf > 0.00, 0] = 1
    cloud = pyrender.Mesh.from_points(xyz, colors)

    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=3)


class get_model(nn.Module):
    # template_prediction
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.normal_channel = normal_channel
        self.feat = PointNetEncoder(global_feat=True, spatial_transform=False,feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        #self.dropout = nn.Dropout(p=0.4)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.embed = nn.Parameter(torch.randn(num_class))
        #'''

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1).contiguous() # batch, 6, num_pts
        #self.draw_point_cloud(xyz.permute(0, 2, 1).contiguous()[0].cpu().detach(),norm.permute(0, 2, 1).contiguous()[0].cpu().detach())
        #batch = xyz.shape[0]
        #return self.embed[None].expand(batch,-1),None
        x, trans_feat = self.feat(xyz)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = F.log_softmax(x, -1)

        return x, trans_feat

    def draw_point_cloud(self,points,colors):
        cloud = pyrender.Mesh.from_points(points,colors)
        scene = pyrender.Scene()
        scene.add(cloud)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=3)



class mini_pointnet(nn.Module):
    # template_prediction
    def __init__(self,latent_dim,normal_channel=True):
        super().__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.normal_channel = normal_channel
        self.feat = mini_PointNetEncoder(global_feat=True,spatial_transform=False, feature_transform=False, channel=channel)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, latent_dim)
        #self.dropout = nn.Dropout(p=0.4)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        #self.embed = nn.Parameter(torch.randn(num_class))
        #'''

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1).contiguous() # batch, 6, num_pts
        #self.draw_point_cloud(xyz.permute(0, 2, 1).contiguous()[0].cpu().detach(),norm.permute(0, 2, 1).contiguous()[0].cpu().detach())
        #batch = xyz.shape[0]
        #return self.embed[None].expand(batch,-1),None
        x,trans_feat = self.feat(xyz)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = F.log_softmax(x, -1)

        return x

    def draw_point_cloud(self,points,colors):
        cloud = pyrender.Mesh.from_points(points,colors)
        scene = pyrender.Scene()
        scene.add(cloud)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=3)

import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self,num_nodes,embed_dim,dim,num_layers):
        super().__init__()
        self.num_nodes = num_nodes

        # Define the first layer
        self.layers = nn.ModuleList([nn.Linear(embed_dim, dim)])

        # Define hidden layers
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(dim, dim))

        # Define the output layer
        self.output_layer = nn.Linear(dim, 1)

        self.output_bias=0.5
        if cfg.supervision == 'sdf':
            self.output_bias=0.0

    # declariation of network component


    def forward(self,x):
        #computation of x
        for layer in self.layers:
            x = F.relu(layer(x))
        #print(self.output_bias)
        x= self.output_layer(x)+self.output_bias
        return x

class MultiMLP_archiv(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        # declariation of network component
        MLP_list = []
        for i in range(num_nodes):
            MLP_list.append(MLP())


    def forward(self,x,T):
        # Compute Gaussians: G(x,theta): Batch, num_pts
        # Evaluate MLP:
            # x.shape = batch, num_pts, 3
            # local_x.shape = batch * num_pts, num_nodes, 3    (local_xi = T-1 * global_x)
            # feed local_x to MLP: R3->R
            # output of MLP: batch * num_pts, num_nodes, 1
        # Sum:
            # sdf: batch, num_pts

        # Weightening: LDIF(x)=G(x,theta) * SDF(x)
            # sdf: batch, num_pts

        return x


    def Analytic_shape_function(self,x,Embedding):
        # input:  x    x.shape = batch, num_pts, 3
        #         T    T.shape = batch, num_nodes, #params_node_i
        # return: G(x,T)  G.shape = batch, num_pts

        return 0


from models.node_proc import sample_rbf_weights,global_to_local,convert_embedding_to_explicit_params,global_to_local_with_normal,local_to_global_with_normal
from models.embedder import Embedder,RBFLayer

class MultiMLP(nn.Module):
    def __init__(self, num_nodes, point_dim, hidden_dim=256, output_dim=1, num_layers=4):
        super(MultiMLP, self).__init__()
        self.num_nodes = num_nodes

        embed_dim = 3
        if cfg.use_ff:
            if cfg.embed_type == 'PE':
            #FFmaping for 3d and 4d
                embed_dim = 6*cfg.ff_sigma +3
                self.embedder = Embedder(
                    include_input=True,
                    input_dims=3,
                    max_freq_log2=cfg.ff_sigma-1,
                    num_freqs=cfg.ff_sigma,
                    log_sampling=True,
                    periodic_fns=[torch.sin, torch.cos],
                )
            elif cfg.embed_type == 'RBF':
                embed_dim = cfg.gaussians_center
                self.rbf_layer = RBFLayer(in_features=3, out_features=cfg.gaussians_center)

        embed_dim = embed_dim+ cfg.latent_dim # latent
        # Define the first layer
        self.MLPs= nn.ModuleList()
        for i in range(num_nodes):
            self.MLPs.append(MLP(num_nodes,embed_dim,hidden_dim,num_layers))


        '''
        self.layers = nn.ModuleList(
            [nn.Conv1d(num_nodes * embed_dim, (hidden_dim+1) * num_nodes, kernel_size=1, groups=num_nodes)])

        # Define hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                nn.Conv1d(hidden_dim * num_nodes, hidden_dim * num_nodes, kernel_size=1, groups=num_nodes))

        # Define the output layer
        self.output_layer = nn.Conv1d(hidden_dim * num_nodes, output_dim * num_nodes, kernel_size=1, groups=num_nodes)
    
        '''


    def forward(self, X, latent_list,constants, scales, rotations, centers, use_constants):
        # X: (batch, num_pts, 3)
        # Theta: (batch, num_nodes, 10)
        # Z: (batch, num_nodes, dim_Z)

        # sample_rbf_weights(points, constants, scales, centers, use_constants):
        # (bs, num_points, num_nodes, 3) --->return : (batch_size, num_points, num_nodes)

        #   return:
        #   LDIF: (batch, num_pts, 1)

        batch_size = X.shape[0]
        num_points = X.shape[1]

        #num_nodes = constants.shape[1]
        #assert X.shape[2] == 3

        #local_X = X.reshape(batch_size, num_points, 1, 3).expand(-1, -1, self.num_nodes, -1)  # (bs, num_points, num_nodes, 3)
        local_X = global_to_local(X,scales,rotations,centers)# (bs, num_points, num_nodes, 3)

        #local_X = self.embedder.embed(local_X)  # bs,num_points,num_nodes,3
        if cfg.use_ff:
            if cfg.embed_type == 'RBF':
                local_X = self.rbf_layer(local_X.reshape(-1,3))
                local_X =local_X.reshape(batch_size,num_points,self.num_nodes,-1)
            if cfg.embed_type == 'PE':
                local_X = self.embedder.embed(local_X)
        # now concatenate them with latents
        # Use torch.split to split the tensor along the specified dimension
        split_X = list(torch.split(local_X, split_size_or_sections=1, dim=2))

        input_list = []
        for i,element in enumerate(split_X):
            input_i = torch.cat([element, (latent_list[i][:, None, :].expand(-1, num_points, -1)[:, :, None, :])], dim=-1)
            input_list.append(self.MLPs[i](input_i.squeeze(2)))
            #input_list.append(self.MLPs[i](element.squeeze(2)))

        pred_sdf = torch.stack(input_list,dim=2)
        '''
        #input = local_X
        # convert X into conv1d input
        #input = input.permute(0,2,3,1).reshape(batch_size,-1,num_points)

        #batch_size, num_points, num_nodes, _ = X.shape

        # Apply the layers
        for layer in self.layers:
            input = F.relu(layer(input))

        #output : batch_size, num_nodes*3 ,num_points

        # Apply the output layer and reshape to original format
        pred_sdf = self.output_layer(input).reshape(batch_size,self.num_nodes,1,num_points).permute(0,3,1,2)# (batch_size, num_points, num_nodes, 1)

        #pred_sdf = pred_sdf.reshape(batch_size, num_points, self.num_nodes, -1)  # (batch_size, num_points, num_nodes, 1)
        # points_sdf = points_sdf+1 # ??? value stabliize?
        '''
        gaussianWeigths = sample_rbf_weights(X, constants, scales,rotations, centers, use_constants).unsqueeze(-1)  # shape: batch_size, num_points, num_nodes,1

        weighted_points_sdf = gaussianWeigths * (-1)
        #+pred_sdf)

        weighted_Gaussians = gaussianWeigths * (-1)
        from models.node_proc import sample_rbf_surface, compute_inverse_occupancy

        # Sum across the num_nodes dimension
        #sdf_pred = sample_rbf_surface(X, constants, scales, rotations, centers, cfg.use_constants,
        #                              cfg.aggregate_coverage_with_max)

        class_pred = compute_inverse_occupancy(torch.sum(weighted_points_sdf,dim=2), cfg.soft_transfer_scale, cfg.level_set)

        #ldif_values = torch.where(class_pred <=0.5,torch.sum(weighted_points_sdf,dim=2) ,1e-6*torch.ones_like(torch.sum(weighted_points_sdf,dim=2)))
        ldif_values = torch.sum(weighted_points_sdf,dim=2)
        #ldif_values = weighted_points_sdf

        ldif_gaussian = (-cfg.level_set + torch.sum(weighted_Gaussians, dim=2))

        return ldif_values,class_pred  # (batch_size, num_points, 1)





import numpy as np
import plyfile
from models.sdf_meshing import create_meshes
import models.config as cfg
class LDIF(nn.Module):
    def __init__(self, num_nodes,type):
        super().__init__()
        self.num_nodes =num_nodes
        self.tempalte_predictor = get_model(num_class=11*num_nodes,normal_channel=True)
        if cfg.use_mirror:
            num_nodes = num_nodes*2

        self.PointNet_mini = nn.ModuleList([])
        for i in range(num_nodes):
            self.PointNet_mini.append(mini_pointnet(latent_dim=cfg.latent_dim,normal_channel=True))
        self.multi_mlp = MultiMLP(num_nodes=num_nodes,point_dim=3,hidden_dim=cfg.hidden_dim)
        self.supervision = type
        self.truncate_radius = 1.0



    def forward(self, on_surface_pts, coords):
        # xs: batch_size,num_pts,pts_dim(7)
        batch = on_surface_pts.shape[0]
        num_pts = on_surface_pts.shape[1]
        oriented_pc = on_surface_pts[:, 0:(num_pts // 2),[0,1,2,4,5,6] ]

        #data check
        #vis_data(on_surface_pts[:,0:(num_pts // 2),:].detach().cpu().numpy())

        embedding,_ = self.tempalte_predictor(oriented_pc) # batch,num_nodes*11
        constants, scales, rotations, centers = convert_embedding_to_explicit_params(embedding, self.num_nodes) # batch_size, num_nodes ( () ,(3), (3,3), (3))
        if cfg.use_mirror:
            constants, scales, rotations, centers = self.symmetry_mirror(constants, scales, rotations, centers)
        #'''
        # use Gaussian balls to get local feature:
        # global_to_local

        xyz_normal = global_to_local_with_normal(oriented_pc,scales,rotations,centers) # batch, num_pts,num_nodes,6

        threshold = 2.0

        # truncate - TODO:might have to use FPS sampling
        distances = torch.norm(xyz_normal[..., :3], p=2, dim=-1) # batch,num_pts,num_nodes
        probabilities = torch.rand(distances.shape, device=xyz_normal.device)
        is_valid = distances < threshold
        #print(is_valid.mean())

        sample_order = torch.where(is_valid, probabilities, -distances)
        _, top_indices = torch.topk(sample_order, k=100, dim=1, largest=True, sorted=False)


        xyz_normal_select = torch.gather(xyz_normal,dim=1,index=top_indices[...,None].expand(-1,-1,-1,6))  # batch, num_pts,num_nodes,6
        # batch,num_pts_sel,num_balls,6

        #data check
        xyz_normal_select_vis = local_to_global_with_normal(xyz_normal_select, scales, rotations, centers)
        #batch, num_pts ,num_balls,6

        xyz_normal_list = list(torch.split(xyz_normal_select, split_size_or_sections=1, dim=2))
        xyz_normal_list = [xyz_normal.squeeze(2) for xyz_normal in xyz_normal_list]




        latent_list=[]
        for i in range(scales.shape[1]):
            latent_list.append(self.PointNet_mini[i](xyz_normal_list[i]))  #batch, hidden_dim
        
        #'''
        #latent_list = []

        # depend on supervison, feed in different coordinate

        ldif,ldif_gaussian = self.multi_mlp(coords,latent_list, constants, scales, rotations, centers, use_constants=cfg.use_constants)
        return ldif,embedding,latent_list,ldif_gaussian, xyz_normal_select_vis

    def symmetry_mirror(self,constants, scales, rotations, centers):
        flip_axis=0
        constants_dup = constants.clone()
        scales_dup = scales.clone()
        #scales_dup[:,:,flip_axis] = -scales_dup[:,:,flip_axis]
        rotations_dup = rotations.clone()
        rotations_dup[:,:,flip_axis,:] = -rotations_dup[:,:,flip_axis,:]
        centers_dup=centers.clone()
        centers_dup[:,:,flip_axis] = -centers_dup[:,:,flip_axis]

        constants = torch.cat([constants,constants_dup],dim=1)
        scales = torch.cat([scales,scales_dup],dim=1)
        rotations = torch.cat([rotations,rotations_dup],dim=1)
        centers = torch.cat([centers,centers_dup],dim=1)

        return constants, scales, rotations, centers



    def get_mesh(self,embedding,latent_list):
        constants, scales, rotations, centers = convert_embedding_to_explicit_params(embedding, self.num_nodes)
        if cfg.use_mirror:
            embedding = self.symmetry_mirror(constants, scales, rotations, centers)
        else:
            embedding =constants, scales, rotations, centers
        #meshes = create_meshes(
        #    self,embedding, latent_list,level=-0.07, N=128,type='ldif'
        #)
        meshes = create_meshes(
            self,embedding, latent_list,level=0.5, N=128,type='gaussian'
        )

        return meshes


    def reconstruct_shape(self,meshes,epoch,it=0,mode='train'):
        for k in range(len(meshes)):
            # try writing to the ply file
            verts = meshes[k]['vertices']
            faces = meshes[k]['faces']
            voxel_grid_origin = [-0.5] * 3
            mesh_points = np.zeros_like(verts)
            mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
            mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
            mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

            num_verts = verts.shape[0]
            num_faces = faces.shape[0]

            verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

            for i in range(0, num_verts):
                verts_tuple[i] = tuple(mesh_points[i, :])

            faces_building = []
            for i in range(0, num_faces):
                faces_building.append(((faces[i, :].tolist(),)))
            faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

            el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
            el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

            ply_data = plyfile.PlyData([el_verts, el_faces])
            # logging.debug("saving mesh to %s" % (ply_filename_out))
            ply_data.write("./ply/" + str(epoch) + "_" +str(mode)+"_"+ str(it*len(meshes)+k) + "_poly.ply")


from models.siren_pytorch import Siren_Decoder, SirenNet

class PatchNet(nn.Module):
    def __init__(self, num_nodes, latent_dim, type='sdf'):
        super().__init__()
        self.num_nodes = num_nodes
        #self.tempalte_predictor = get_model(num_class=11 * num_nodes, normal_channel=True)
        #if cfg.use_mirror:
        #    num_nodes = num_nodes * 2

        #self.PointNet_mini = nn.ModuleList([])
        #for i in range(num_nodes):
        #    self.PointNet_mini.append(mini_pointnet(latent_dim=cfg.latent_dim, normal_channel=True))
        #self.patchnet_decoder = PatchNetDecoder()
        net = SirenNet(
            dim_in=3,  # input dimension, ex. 2d coor
            dim_hidden=256,  # hidden dimension
            dim_out=1,  # output dimension, ex. rgb value
            num_layers=6,  # number of layers
            final_activation=torch.nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
            w0=30,
            w0_initial=30. # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )
        self.siren_decoder = Siren_Decoder(net,latent_dim)
        self.supervision = type
        self.truncate_radius = 1.0

    def sample_uniform_points_in_unit_sphere(self,batch,amount):
        unit_sphere_points = np.random.uniform(-1, 1, size=(batch*(amount * 2 + 2000), 3))
        unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]
        unit_sphere_points = unit_sphere_points[:amount*batch,:]
        return unit_sphere_points.reshape(batch,amount,3) #batch, amount, 3


    def forward(self, on_surface_sample, extrinsics,latent_codes,type='train'):
        # on_surface_sample: batch_size,num_pts,pts_dim(7)
        # extrinsics   # batch, num_nodes, ext_size
        # latent_codes # batch, num_nodes, latent_size
        # sdf #batch num_nodes, 1
        pass
        '''prepare Input'''
        batch = on_surface_sample.shape[0]
        num_pts = on_surface_sample.shape[1]
        dim = on_surface_sample.shape[2] #6


        # data check
        # vis_data(on_surface_pts[:,0:(num_pts // 2),:].detach().cpu().numpy())
        '''convert embeddings into extrinsics'''
        constants, scales, rotations, centers = convert_embedding_to_explicit_params(extrinsics,self.num_nodes)  # batch_size, num_nodes ( () ,(3), (3,3), (3))
        num_patch = constants.shape[1]

        '''convert global into local Gaussians coordinates'''
        coords_local = global_to_local_with_normal(on_surface_sample, scales, rotations, centers)  # batch, num_pts,num_nodes,6
        #xyz_normal = oriented_pc[:,:,None,:].expand(-1,-1,num_patch,-1)
        threshold = 1.0

        ''' truncate points within the surface using distances metrics ''' # only for visualization now
        # truncate - TODO:might have to use FPS sampling
        if type == 'train':
            num_pts_sel = 1024

            distances = torch.norm(coords_local[:,:,:,:3], p=2, dim=-1)  # batch,num_pts,num_nodes

            probabilities = torch.rand(distances.shape, device=coords_local.device)
            is_valid = distances < threshold
            #print(is_valid.mean())
            sample_order = torch.where(is_valid, probabilities, -distances)
            _, top_indices = torch.topk(sample_order, k=num_pts_sel, dim=1, largest=True, sorted=False) # batch,num_pts_sel ,num_balls


            coords_select = torch.gather(coords_local, dim=1, index=top_indices[..., None].expand(-1, -1, -1, 6))  # batch, num_pts_sel,num_nodes,6

            '''create off sdf samples'''
            coords_sel_off = torch.from_numpy(self.sample_uniform_points_in_unit_sphere(batch, num_pts_sel))[:, :, None, :]\
                .expand(-1, -1,num_patch,-1).to(coords_select.device) # batch, num_pts_sel, num_nodes, 3
            coords_sel_off = torch.cat([coords_sel_off, coords_sel_off], dim=-1)# batch, num_pts_sel, num_nodes, 6

            coords_select = torch.cat([coords_select, coords_sel_off], dim=1)  # batch, 2*num_pts_sel, num_nodes, 6


            coords_select_vis = local_to_global_with_normal(coords_select, scales, rotations, centers)  # batch,2*num_pts_sel,num_balls,6

            coords_input = coords_select[:,:,:,:3].clone().detach()
            coords_input.requires_grad = True
        else:
            coords_select = 0
            coords_input = coords_local
            coords_select_vis = local_to_global_with_normal(coords_input, scales, rotations, centers)
        '''select gt_sdf'''

        #xyz_normal_select_vis = 0
        '''cat latent_codes && extrinsics as input for network'''
        #xyz = xyz_normal[:,:,:,:3] # batch,num_pts,num_nodes,3
        #latent_codes = latent_codes[:,None,:,:].expand(-1,num_pts,-1,-1) #batch, num_pts,num_nodes,latent_size
        #input = torch.cat([xyz,latent_codes],dim=-1) #batch, num_pts, num_nodes, 3+latent_size
        '''for testing extrinsics loss'''
        patch_sdfs = 0
        weighted_sdf = 0
        #patch_weight, scaled_distance = sample_rbf_weights(on_surface_sample[:, :, :3][:, :,None,:].expand(-1, -1, num_patch, -1), constants, scales,
        #                                                     rotations, centers, use_constants=cfg.use_constants)
        ext = scales[:,:,0]
        '''Network Prediction'''
        #patch_sdfs = self.patchnet_decoder(input.float()).squeeze(-1) #batch, num_pts,num_nodes
        if True:
            patch_sdfs = self.siren_decoder(coords_input.float(),latent_codes.float()).squeeze(-1) #batch, num_pts,num_nodes


            '''blended surface reconstruction''' #TODO: make rbf weights zero at the boundary
            #coords_sel = coords[:,:,None,:].expand(-1,-1,num_patch,-1)
            #coords_sel = torch.gather(coords_sel, dim=1, index=top_indices[..., None].expand(-1, -1, -1, 3))

            patch_weight,scaled_distance = sample_rbf_weights(coords_select_vis[:,:,:,:3], constants, scales, rotations, centers, use_constants=cfg.use_constants)
            # shape: batch_size, num_points, num_nodes
            patch_weight_normalization = torch.sum(patch_weight,dim=2) #shape: batch_size, num_pts

            patch_weight_norm_mask = patch_weight_normalization == 0. #shape:batch_size, num_pts
            patch_weight[patch_weight_norm_mask,:] = 0.0 #for all points outside balls assign zero weights
            patch_weight[~patch_weight_norm_mask,:]= patch_weight[~patch_weight_norm_mask,:] / patch_weight_normalization[~patch_weight_norm_mask].unsqueeze(-1)
            #patch_weight = patch_weight.unsqueeze(-1) #shape: batch_size, num_points, num_nodes,1


            weighted_sdf = patch_weight * patch_sdfs #batch, num_pts,num_nodes
            weighted_sdf = torch.sum(weighted_sdf,dim=2) #batch, num_pts

            # experimental..
            default_sdf_value = 1.0
            weighted_sdf[patch_weight_norm_mask] = default_sdf_value
            #weighted_sdf[~patch_weight_norm_mask] = weighted_sdf[~patch_weight_norm_mask] / patch_weight_normalization[~patch_weight_norm_mask] # what is this??

            weighted_sdf = weighted_sdf.unsqueeze(-1) #batch,num_pts,1


        return weighted_sdf, coords_select_vis, coords_select,coords_input,patch_weight,patch_sdfs,scaled_distance,ext,centers


    def get_mesh(self, batch_latent, batch_ext):


        # meshes = create_meshes(
        #    self,embedding, latent_list,level=0.07, N=128,type='ldif'
        # )
        meshes = create_meshes(
            self, batch_latent, batch_ext, level=0.0, N=256, type='ldif'
        )
        #meshes += (create_meshes(
        #    self, batch_latent, batch_ext, level=0.0000001, N=128, type='guassians'
        #))

        return meshes

    def reconstruct_shape(self, meshes, epoch,savepath, it=0, mode='train'):
        for k in range(len(meshes)):
            # try writing to the ply file
            verts = meshes[k]['vertices']
            faces = meshes[k]['faces']
            voxel_grid_origin = [0] * 3
            mesh_points = np.zeros_like(verts)
            mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
            mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
            mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

            num_verts = verts.shape[0]
            num_faces = faces.shape[0]

            output_file = (savepath +'obj/'+ str(epoch) + "_" + str(mode) + "_" + str(k) + ".obj")

            def write_obj(vertices, faces, output_file):
                """
                Write vertices and faces to an .obj file.

                Parameters:
                - vertices: numpy array of shape (num_vertices, 3)
                - faces: numpy array of shape (num_faces, 3) specifying vertex indices for each face
                - output_file: path to the output .obj file
                """
                with open(output_file, 'w') as f:
                    # Write vertices
                    for vertex in vertices:
                        f.write(f"v {' '.join(map(str, vertex))}\n")

                    # Write faces
                    for face in faces:
                        # Increment indices by 1 (OBJ format uses 1-based indices)
                        face_indices = face + 1
                        f.write(f"f {' '.join(map(str, face_indices))}\n")


            write_obj(mesh_points, faces,output_file)

