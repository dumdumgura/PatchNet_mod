"""
Author: Benny
Date: Nov 2019
"""
import pyrender

import os
import sys
import torch
import numpy as np
import models.config as cfg
import datetime
import logging
import provider
import importlib
import shutil
import argparse
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from PIL import Image

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from torch.utils.tensorboard import SummaryWriter
from logger import  init_tb_logger,MessageLogger
import json
import open3d as o3d
import scipy
import torchvision.transforms as transforms
NORMALIZATION_EPS = 1e-8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
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
    torch.tensor([0, 0, 0]),  # Black
    torch.tensor([128, 0, 0]),  # Dark Red
    torch.tensor([0, 128, 0]),  # Dark Green
    torch.tensor([0, 0, 128]),  # Dark Blue
    torch.tensor([128, 128, 0]),  # Dark Yellow
    torch.tensor([128, 0, 128]),  # Dark Magenta
    torch.tensor([0, 128, 128]),  # Dark Cyan
    torch.tensor([64, 64, 64]),  # Dark Grey
    torch.tensor([20, 20, 20]),  # Very Dark Grey
    torch.tensor([100, 50, 50]),  # Dark Maroon
    torch.tensor([50, 100, 50]),  # Dark Forest Green
    torch.tensor([50, 50, 100]),  # Dark Navy Blue
    torch.tensor([100, 100, 50]),  # Dark Olive
    torch.tensor([100, 50, 100]),  # Dark Purple
    torch.tensor([50, 100, 100]),  # Dark Teal
    torch.tensor([80, 80, 80]),  # Darker Grey
    torch.tensor([100, 80, 80]),  # Darker Maroon
    torch.tensor([80, 100, 80]),  # Darker Forest Green
    torch.tensor([80, 80, 100]),  # Darker Navy Blue
    torch.tensor([70, 70, 0]),  # Dark Olive Green
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
    torch.tensor([0, 0, 0]),  # Black
    torch.tensor([128, 0, 0]),  # Dark Red
    torch.tensor([0, 128, 0]),  # Dark Green
    torch.tensor([0, 0, 128]),  # Dark Blue
    torch.tensor([128, 128, 0]),  # Dark Yellow
    torch.tensor([128, 0, 128]),  # Dark Magenta
    torch.tensor([0, 128, 128]),  # Dark Cyan
    torch.tensor([64, 64, 64]),  # Dark Grey
    torch.tensor([20, 20, 20]),  # Very Dark Grey
    torch.tensor([100, 50, 50]),  # Dark Maroon
    torch.tensor([50, 100, 50]),  # Dark Forest Green
    torch.tensor([50, 50, 100]),  # Dark Navy Blue
    torch.tensor([100, 100, 50]),  # Dark Olive
    torch.tensor([100, 50, 100]),  # Dark Purple
    torch.tensor([50, 100, 100]),  # Dark Teal
    torch.tensor([80, 80, 80]),  # Darker Grey
    torch.tensor([100, 80, 80]),  # Darker Maroon
    torch.tensor([80, 100, 80]),  # Darker Forest Green
    torch.tensor([80, 80, 100]),  # Darker Navy Blue
    torch.tensor([70, 70, 0]),  # Dark Olive Green
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
    torch.tensor([0, 0, 0]),  # Black
    torch.tensor([128, 0, 0]),  # Dark Red
    torch.tensor([0, 128, 0]),  # Dark Green
    torch.tensor([0, 0, 128]),  # Dark Blue
    torch.tensor([128, 128, 0]),  # Dark Yellow
    torch.tensor([128, 0, 128]),  # Dark Magenta
    torch.tensor([0, 128, 128]),  # Dark Cyan
    torch.tensor([64, 64, 64]),  # Dark Grey
    torch.tensor([20, 20, 20]),  # Very Dark Grey
    torch.tensor([100, 50, 50]),  # Dark Maroon
    torch.tensor([50, 100, 50]),  # Dark Forest Green
    torch.tensor([50, 50, 100]),  # Dark Navy Blue
    torch.tensor([100, 100, 50]),  # Dark Olive
    torch.tensor([100, 50, 100]),  # Dark Purple
    torch.tensor([50, 100, 100]),  # Dark Teal
    torch.tensor([80, 80, 80]),  # Darker Grey
    torch.tensor([100, 80, 80]),  # Darker Maroon
    torch.tensor([80, 100, 80]),  # Darker Forest Green
    torch.tensor([80, 80, 100]),  # Darker Navy Blue
    torch.tensor([70, 70, 0]),  # Dark Olive Green
]
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=20, type=int, choices=[1, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=20000, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=0, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()

def _init_latent_vectors(sdf_samples_object,specs,load_init_path=None):
        logging.info("There are {} scenes".format(sdf_samples_object.__len__()))
        lat_vecs = []
        initial_patch_latent = None
        for i, npyfile in enumerate(sdf_samples_object.npyfiles):
            if i % 1000 == 0:
                print("initializing: " + str(i) + " / " + str(sdf_samples_object.__len__()), flush=True)


            filename = os.path.join(sdf_samples_object.data_source, npyfile)
            #filename = os.path.join(sdf_samples_object.data_source, ws.sdf_samples_subdir, npyfile)
            if load_init_path is None:
                print('init from point cloud')
                vec = initialize_mixture_latent_vector(specs, sdf_filename=filename,
                                                       overwrite_init_file=specs["overwrite_init_files"],
                                                       use_precomputed_init=specs["use_precomputed_init"],
                                                       initial_patch_latent=initial_patch_latent).cuda()
            else:
                print('using precomputed init')
                initialization_file = "_init_" + str(specs['num_patches']) + "_" + str(
                    specs['PatchCodeLength']) + "_" + npyfile
                path = load_init_path +'/'+initialization_file
                vec = torch.load(path)
                vec = torch.tensor(vec).cuda()
            # vec:Tensor: num_patches, latent_size+7
            vec.requires_grad = True
            lat_vecs.append(torch.nn.Embedding.from_pretrained(vec))
        return lat_vecs


def get_settings_dictionary():
    dict = {}

    dict["default_specs_file"] = "specs.json"
    dict["root_folder"] = "some_folder_for_experiments"

    return dict


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    experienment_name = 'PatchNet_overfit_9'

    init_latent_path = 'logs/PatchNet/PatchNet_overfit_9/latent_init'
    #init_latent_path = None
    model_path = 'logs/PatchNet/PatchNet_overfit_9/ckpt/1000.pth'
    #model_path =None

    '''HYPER PARAMETER'''
    #
    print(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", int(args.gpu))

    specs_file = None
    code_folder = os.path.dirname(os.path.realpath(__file__))
    system_specific_settings = get_settings_dictionary()

    if specs_file is None:
        specs_file = code_folder + "/" + system_specific_settings["default_specs_file"]

    with open(specs_file) as specs:
        specs = "\n".join([line for line in specs.readlines() if line.strip()[:2] != "//"]) # remove comment lines
        specs = json.loads(specs)



    '''CREATE DIR'''
    exp_dir = Path('./log/')



    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    #logger.setLevel(logging.INFO)
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    #file_handler.setLevel(logging.INFO)
    #file_handler.setFormatter(formatter)
    #logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    from data_utils.mydatasets import ShapeNet
    folder='/home/umaru/praktikum/changed_version/Pointnet_Pointnet2_pytorch_current/data/sdf_test'
    split = 'val'
    sdf_dataset = ShapeNet(folder, split=split, type='sdf', subsample=1024*30)  #has to be larger for 3000, higher the better, since each patch only get few
    trainDataLoader = torch.utils.data.DataLoader(sdf_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)


    '''Latent Initialization'''
    train_lat_vecs = _init_latent_vectors(sdf_dataset,specs,init_latent_path) # list of (patch_num, latent_size+10) Embedding!  and they are on gpu now!
    embedding_parameters = []
    for embedding in train_lat_vecs:
        embedding_parameters.extend(embedding.parameters()) #list of (patch_num, latent_size+10) Tensor!



    '''check data'''
    idx=0
    lat_ext = embedding_parameters[0][:,-7:] #patch_num,10
    num_patch = lat_ext.shape[0]
    constant, scale, rotation, center = convert_embedding_to_explicit_params(embedding=lat_ext[None],num_nodes=num_patch)
    ellipsoids = createBatchEllipsoids(n=num_patch, scale_constants=scale[idx],semi_axes=scale[idx], centers=center[idx], rotation_matrices=rotation[idx])


    initialization_file = sdf_dataset.npyfiles[idx]
    pts = 6 * 10000
    point_cloud = np.load(os.path.join(folder,split,initialization_file))
    on_surface_pts = point_cloud[:pts // 3, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(on_surface_pts[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(on_surface_pts[:, 3:6])
    ellipsoids.append(pcd)
    #o3d.visualization.draw_geometries(ellipsoids,point_show_normal=True)


    '''MODEL LOADING'''
    from models.pointnet2_reg_ssg import PatchNet
    from models.loss import SamplerLoss
    #model = PatchNet(num_nodes=num_patch)
    '''
    from models.siren_pytorch import SirenNet
    model = SirenNet(
        dim_in=3,  # input dimension, ex. 2d coor
        dim_hidden=256,  # hidden dimension
        dim_out=1,  # output dimension, ex. rgb value
        num_layers=5,  # number of layers
        final_activation=torch.nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
        w0=30,
        w0_initial=30.  # different signals may require different omega_0 in the first layer - this is a hyperparameter
    )
    '''
    model = PatchNet(num_nodes=num_patch,latent_dim=specs['PatchCodeLength'])
    '''GPU SETTINGS'''
    if torch.cuda.is_available():
        model = model.to(device)

    '''Optimizier LOADING'''
    # TODO: make an if branch for deciding which part of embeddings should be sent for optimizer
    embedding_parameters = []
    for embedding in train_lat_vecs:
        for tensor in embedding.parameters():
            tensor.requires_grad =True
        embedding_parameters.extend(embedding.parameters())
    optimizer_parameters = [
            {"params": model.parameters(), "lr":   0.00001}, #has to be 1e-5 or lower for siren tsdf, 1e-4 for embedding
            {"params": embedding_parameters, "lr": 0.0001} #0.001
        ]

    optimizer_train = torch.optim.Adam(optimizer_parameters,betas=(0.9, 0.999),eps=1e-08,)

    '''Loss criterion'''
    from models.loss_patchNet import PatchNetLoss
    from models.loss import ReconstructionLoss,CovLoss,SurLoss,NodeSparsityLoss
    #criterion = torch.nn.L1Loss() #TODO: Initialization of the Loss
    criterion = ReconstructionLoss(type='ssdf')
    criterion_CovLoss= CovLoss()
    criterion_SurLoss= SurLoss()
    criterion_Spars =  NodeSparsityLoss()


    if model_path != None:
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_train.load_state_dict(checkpoint['optimizer_state_dict'])
        log_string('Use pretrain model')
    else:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_train, step_size=1000, gamma=0.1)
    global_epoch = 0
    global_step = 0





    writer = SummaryWriter("logs/PatchNet/"+experienment_name)
    exp_dir = Path('logs/PatchNet/'+experienment_name+'/ckpt')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = Path('logs/PatchNet/'+experienment_name+'/latent_init')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = Path('logs/PatchNet/'+experienment_name+'/obj')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = Path('logs/PatchNet/'+experienment_name+'/pic')
    exp_dir.mkdir(exist_ok=True)



    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        model = model.train()
        pbar = tqdm(enumerate(trainDataLoader), total=len(trainDataLoader))
        #scheduler.step()
        #print(scheduler.get_last_lr())

        epoch_batch_ext=[]
        epoch_xyz=[]
        epoch_batch_latent=[]

        for it,xt in pbar:
            loss_record = {}
            '''take in batch data'''
            batch_indices_to_optimize = []
            #near_surface_pts = xt['near_surface_pts'].to(device)  ## used for sdf reconstruction!
            on_surface_pts = xt['on_surface_pts'].to(device)
            '''decide supervision form'''
            num_pts = on_surface_pts.shape[1]
            on_surface_samples = on_surface_pts[:,:,:] #only on surface samples with normal
            num_pts = on_surface_samples.shape[1]
            #coords.requires_grad = True
            #grid = xt['grid'].to(device)
            indices= xt['idx']

            '''take in batch latents'''
            latent_inputs = []
            for ind in indices.numpy():
                batch_indices_to_optimize.append(ind)
                latent_ind = embedding_parameters[ind]  # num_patch,latent_size+10
                latent_inputs.append(latent_ind)
            latent_inputs = torch.stack(latent_inputs, 0) #batch,num_patch,latent_size+10

            '''split batch latents into batch extrinsics and batch latent codes'''
            batch_ext = latent_inputs[:,:,-7:]
            epoch_batch_ext.append(batch_ext)
            batch_latent = latent_inputs[:,:,:-7] #batch,num_patch,latent_size
            epoch_batch_latent.append(batch_latent)


            '''zero grads of model and optimizier'''
            model.zero_grad()
            optimizer_train.zero_grad()


            '''Model Prediction'''
            pred_sdf,xyz_normal_sel_vis,gt_select,coords_input,patch_weight,patch_sdfs,scaled_distance,ext,center = model(on_surface_samples,batch_ext,batch_latent,type='train') # pred.shape = Batch,num_ball*params
            epoch_xyz.append(xyz_normal_sel_vis)
            #pred_sdf, xyz_normal_sel_vis, patch_weight, patch_sdfs = model(coords, batch_ext, batch_latent)  # pred.shape = Batch,num_ball*params
            #pred_sdf = model(near_surface_pts[:,:,:3],batch_ext,batch_latent)  #batch, num_pts,1
            '''visualize Batch Point with color'''


            '''Loss Calculation'''
            #input={}
            #input["xyz"] = near_surface_pts
            #input["mixture_latent_vectors"] =latent_inputs # batch x mixture_latent_size
            #gt_sdf = near_surface_pts[:,:,3][:,:,None]
            #gt_sdf = gt_sdf.clamp(-0.1,0.1)
            #gt_sdf_sel = gt_sdf[:,:,None,:].expand(-1,-1,num_patch,-1)

            #loss = criterion(pred_sdf,gt_sdf)



            '''loss on patch directly'''
            #'''
            patch_mask = patch_weight==0 #batch,num_pts,num_nodes
            # l1 loss
            #patch_recon = patch_sdfs - gt_sdf #batch, num_pts,num_nodes,1
            #patch_recon = torch.abs(patch_recon)
            # siren loss
            patch_recon,loss_record_recon = criterion(patch_sdfs,gt_select,coords_input)
            #print(loss_record[1])

            patch_recon[patch_mask] = 0
            patch_recon = patch_recon/ (torch.sum((~patch_mask),dim=1)+NORMALIZATION_EPS) #batch, num_pts, num_nodes
            direct_patch_loss = torch.sum(patch_recon, dim=1).mean()
            loss = direct_patch_loss
            #'''

            '''loss on latent code'''
            #'''
            latent_code_reg_loss = torch.mean(batch_latent.pow(2))
            latent_mean = torch.mean(batch_latent)
            #print(latent_code_reg_loss)
            loss += 0.00001*latent_code_reg_loss  #does not change much?
            #'''

            '''loss on extrinsics'''
            '''
            patch_scaling = ext
            loss_sur = criterion_SurLoss.forward(scaled_distance,patch_scaling)

            loss_cov = criterion_CovLoss.forward(scaled_distance,patch_scaling)

            loss_small = torch.mean(patch_scaling ** 2)

            loss_spars = criterion_Spars(center)
            variances = torch.var(patch_scaling, dim=-1, unbiased=False)
            loss_var = torch.mean(variances)
            #loss_
            loss = 5.0*loss_sur+ 200*loss_cov + loss_small+ 10*loss_spars
            if epoch>1000:
                loss = 5.0*loss_sur+ 200*loss_cov
            # + 20*loss_small + 0.01*loss_var
            print(loss_sur)
            print(loss_cov)
            print(loss_small)
            print(loss_spars)
            #print(loss_small)
            #print(loss_var)
            '''
            '''BP'''
            loss.backward()
            optimizer_train.step()
            global_step += 1

            '''logging for training curve and stats'''
            loss_record['loss_reg'] = latent_code_reg_loss
            loss_record['loss_patch_loss'] = direct_patch_loss
            loss_record['loss_recon_item'] = loss_record_recon
            loss_record['loss_total'] = loss
            loss_record['laten_var'] = latent_code_reg_loss
            loss_record['laten_mean'] = latent_mean

            '''log into tensorboard'''
            writer.add_scalar('train/loss/total_loss', loss_record['loss_total'], global_step)
            writer.add_scalar('train/loss/latent_reg', loss_record['loss_reg'], global_step)
            writer.add_scalar('train/loss/direct_patch', loss_record['loss_patch_loss'], global_step)
            writer.add_scalar('train/loss/dp_onsurface', loss_record['loss_recon_item'][0], global_step)
            writer.add_scalar('train/loss/dp_offsurface', loss_record['loss_recon_item'][1], global_step)
            writer.add_scalar('train/loss/dp_normal', loss_record['loss_recon_item'][2], global_step)
            writer.add_scalar('train/loss/dp_grad', loss_record['loss_recon_item'][3], global_step)

            writer.add_scalar('train/latent/latent_var', loss_record['laten_var'], global_step)
            writer.add_scalar('train/latent/latent_mean', loss_record['laten_mean'], global_step)


        '''Logging'''
        #writer.add_scalar('loss_total', loss, global_step)
        #for key in loss_dict:
        #    writer.add_scalar(key, loss_dict[key], global_step)
        log_string('Train loss: %f' % loss)



        if epoch%1000==0 and epoch>=0:
            # eval process
            savepath = 'logs/PatchNet/'+experienment_name+'/'

            meshes=[]
            epoch_batch_latent=torch.cat(epoch_batch_latent,dim=0)
            epoch_batch_ext =torch.cat(epoch_batch_ext,dim=0)
            epoch_xyz = torch.cat(epoch_xyz,dim=0)

            for idx in range(epoch_batch_latent.shape[0]):
                meshes += model.get_mesh(epoch_batch_ext[idx][None],epoch_batch_latent[idx][None])
                vis_geo(epoch_xyz[idx][None],epoch_batch_ext[idx][None],epoch,num_patch,savepath,idx)

                #vis_geo_loss(on_surface_samples[idx,:,:3][None], scaled_distance[idx][None], batch_ext[idx][None], epoch, num_patch)
                if idx>=5:
                    break

            '''save visualization'''
            logger.info('Saving ply...')
            savepath = 'logs/PatchNet/'+experienment_name+'/'
            model.reconstruct_shape(meshes,epoch,savepath)
            img_gt_list = []
            img_recon_list = []

            for idx in len(meshes):
                vis_mesh(savepath, epoch, idx)
                img_gt_list.append(savepath+"pic/vis_geo_"+str(idx)+".png")
                img_recon_list.append(savepath + "pic/vis_recon_" + str(idx) + ".png")
            image_paths = img_gt_list+img_recon_list
            min_width, min_height = calculate_min_common_size(image_paths)
            center_crop = transforms.CenterCrop((min_height, min_width))
            images = [transforms.ToTensor()(center_crop(Image.open(path))) for path in image_paths]
            # Concatenate images into a grid (2 rows, n columns)
            grid_image = make_grid(images, nrow=len(images) // 2)
            writer.add_image('vis/recon', grid_image, global_step)


            '''save latent'''
            logger.info('Saving latent...')
            save_latent(sdf_dataset,embedding_parameters,specs,experienment_name)

            '''save model'''
            savepath = 'logs/PatchNet/'+experienment_name+'/ckpt/' + str(epoch) + '.pth'
            logger.info('Save model...')
            log_string('Saving at %s' % savepath)
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_train.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, savepath)
        #train_instance_acc = np.mean(mean_correct)


        global_epoch += 1
    logger.info('End of training...')


def save_latent(sdf_samples_object,embedding_parameters,specs,experienment_name):
    assert  len(sdf_samples_object.npyfiles) == len(embedding_parameters)
    for i, npyfile in enumerate(sdf_samples_object.npyfiles):
        latent = embedding_parameters[i].clone().detach().cpu().numpy()
        #print(latent.shape)
        initialization_file = "_init_" + str(specs['num_patches']) + "_" + str(specs['PatchCodeLength']) + "_"  + npyfile
        savepath = 'logs/PatchNet/' + experienment_name + '/latent_init/' +initialization_file

        torch.save(latent,savepath)

def calculate_min_common_size(image_paths):
    min_width, min_height = float('inf'), float('inf')
    max_width, max_height = 0, 0

    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
            min_width = min(min_width, width)
            min_height = min(min_height, height)
            max_width = max(max_width, width)
            max_height = max(max_height, height)

    return min(min_width, max_width), min(min_height, max_height)

def createBatchEllipsoids(n, scale_constants, centers, semi_axes, rotation_matrices):
    centers = centers.clone().detach().cpu().numpy()
    semi_axes = semi_axes.clone().detach().cpu().numpy()
    rotation_matrices = rotation_matrices.clone().detach().cpu().numpy()

    resolution = 25  # Number of points on each axis

    # Create points on the ellipsoid's surface
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)
    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_v, sin_v = np.cos(v), np.sin(v)

    # Adjust for broadcasting
    cos_u, sin_u = cos_u[np.newaxis, np.newaxis, ...], sin_u[np.newaxis, np.newaxis, ...]
    cos_v, sin_v = cos_v[np.newaxis, np.newaxis, ...], sin_v[np.newaxis, np.newaxis, ...]

    # For numerical stability we use at least eps.
    #scales = np.where(scale_constants > NORMALIZATION_EPS, scale_constants,
    #                  NORMALIZATION_EPS * np.ones_like(scale_constants))  # (bs, num_nodes, 3)
    #inv_scales = 1.0 / scales
    # Scale and semi-axes adjustment
    semi_axes = np.array(semi_axes)*0.5
    semi_axes = semi_axes[:, :, np.newaxis, np.newaxis]

    # Create ellipsoids
    x = semi_axes[:, 0] * cos_u * sin_v
    y = semi_axes[:, 1] * sin_u * sin_v
    z = semi_axes[:, 2] * cos_v

    # Reshape for points
    points = np.stack([x, y, z], axis=-1)
    points = points.reshape(n, -1, 3)  # Flatten resolution dimensions

    # Apply rotation and translation
    points = np.einsum('nij,nkj->nki', rotation_matrices, points)  # Apply rotation
    points += centers[:, np.newaxis, :]  # Translation

    # Apply rotation to normal
    normals = np.repeat(np.array([0,0,1])[None],repeats=n,axis=0).reshape(n,1,3)
    normals = np.einsum('nij,nkj->nki', rotation_matrices, normals)  # Apply rotation

    normals = normals.reshape(n, 3)

    # Create meshes for each ellipsoid
    meshes = []
    import random
    random.seed(128)
    color = [[0.2 + 0.5 * random.randint(0, 1) for _ in range(3)] for _ in range(n)]
    for i in range(n):
        # Create triangles for each ellipsoid
        triangles = []
        for j in range(resolution - 1):
            for k in range(resolution - 1):
                p1 = j * resolution + k
                p2 = p1 + 1
                p3 = (j + 1) * resolution + k
                p4 = p3 + 1
                triangles.append([p1, p2, p3])
                triangles.append([p2, p4, p3])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points[i])
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        wireFrame = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wireFrame.paint_uniform_color(colors_rgb[i]/255)  # Color the wireframe
        #meshes.append(wireFrame)

    #print(normals)
    pc = o3d.geometry.PointCloud()
    pc.points =  o3d.utility.Vector3dVector(centers.reshape(n,3))
    pc.normals =  o3d.utility.Vector3dVector(normals.reshape(n,3))
    meshes.append(pc)

    return meshes

def farthest_point_sampling(points, K):
    # greedily sample K farthest points from "points" (N x 3)
    num_points = points.shape[0]
    if num_points < K:
        print("too few points for farthest point sampling. will returned repeated indices.")
        indices = np.tile(np.arange(num_points), int(K // num_points) + 1)
        indices = indices[:K]
        return points[indices,:], None

    # compute all pairwise distances
    import scipy.spatial
    pairwise_distances = scipy.spatial.distance.cdist(points, points, metric="euclidean") # points x points
    farthest_points_mask = np.zeros(num_points).astype(bool)
    farthest_points_mask[0] = True
    index_helper = np.arange(num_points)
    for k in range(K-1):
        relevant_distances = pairwise_distances[np.ix_(index_helper[~farthest_points_mask], index_helper[farthest_points_mask])] # distances from not-yet-sampled points to farthest points
        relevant_minimums = np.min(relevant_distances, axis=1)
        new_farthest_point_index = np.argmax(relevant_minimums) # new_farthest_point_index indexes "1" entries in ~farthest_points_mask, not num_points
        new_farthest_point_index = index_helper[~farthest_points_mask][new_farthest_point_index]
        farthest_points_mask[new_farthest_point_index] = True

    return points[farthest_points_mask,:], farthest_points_mask # numpy array K x 3, Boolean mask of size "num_points"

def _normalized_vector(a):
    return a / np.linalg.norm(a)

def _angle_between_vectors(a, b):
    an = _normalized_vector(a)
    bn = _normalized_vector(b)
    return np.arccos(np.clip(np.dot(an, bn), -1.0, 1.0))

def _get_euler_angles_from_rotation_matrix(rotation_matrix):
    # rotation_sequence: Z-Y-X in local coordinate -> R =RxRyRz
    if np.abs(rotation_matrix[2,0]) != 1.:
        beta = -np.arcsin(rotation_matrix[2,0])
        cosBeta = np.cos(beta)
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
    rotation_x = torch.tensor([[1, 0, 0],
                              [0, torch.cos(roll), -torch.sin(roll)],
                              [0, torch.sin(roll), torch.cos(roll)]])

    rotation_y = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                              [0, 1, 0],
                              [-torch.sin(pitch), 0, torch.cos(pitch)]])

    rotation_z = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                              [torch.sin(yaw), torch.cos(yaw), 0],
                              [0, 0, 1]])

    # Combine the individual rotation matrices
    rotation_matrix = torch.matmul(rotation_x, torch.matmul(rotation_y, rotation_z)).to(euler_angles.device)



    return rotation_matrix

def _get_rotation_from_normal(normal):
    # normal: numpy array of size (3,)

    # left-handed coordinate system with x-y plane and z-axis as height (x=thumb, y=middle, z=index). want to rotate the normal such that it aligns with the z-axis. apply that rotation to the patch as a whole
    # first, rotate coordinate system around y-axis such that the normal lies in the y-z plane.
    projected_z_axis = np.array([0., 1.]) # in x-z plane
    projected_normal = np.array([normal[0], normal[2]])
    if np.linalg.norm(projected_normal) < 0.000001:
        y_angle = 0.
    else:
        y_angle = _angle_between_vectors(projected_z_axis, projected_normal)
    if normal[0] > 0.:
        y_angle *= -1.
    y_rotation = scipy.spatial.transform.Rotation.from_euler("y", y_angle)
    rotated_normal = y_rotation.apply(normal)
    # then, rotate around x-axis to align the normal with the z-axis
    z_axis = np.array([0., 0., 1.])
    x_angle = _angle_between_vectors(z_axis, rotated_normal)
    if rotated_normal[1] <= 0.:
        x_angle *= -1.
    x_rotation = scipy.spatial.transform.Rotation.from_euler("x", x_angle)

    # converts global coordinates into local coordinates. the normal is in global coordinates. after multiplication with the rotation matrix, we get the local z-axis [0,0,1].
    rotation = x_rotation * y_rotation
    rotation = rotation.as_matrix() # 3x3 numpy array
    # we use local-to-global rotations in the model. so we need to take the inverse.
    rotation = rotation.transpose()

    euler_angles = _get_euler_angles_from_rotation_matrix(rotation)

    if np.any(np.isnan(euler_angles)):
        euler_angles = np.zeros(3)

    return euler_angles # numpy array of size (3,)

def initial_metadata_from_sdf_samples(surface_samples, normals, num_patches, num_samples_for_computation=10000, final_scaling_increase_factor=1.3):
    # sdf_samples: numpy array, num_points x 4
    # normals: numpy array, num_points x 3


    # return:
    # patch_center:    num_nodes,3
    # patch_constant:  num_nodes,1
    # patch_scale:     num_nodes,3
    # patch_rot:       num_nodes,4

    num_surface_samples = surface_samples.shape[0]
    if num_surface_samples < num_patches:
        raise RuntimeError("not enough surface SDF samples found")

    # DeepSDF preprocessing generates ~500k points. Considering all of them is very expensive. Instead, only consider at most num_samples_for_computation many of them.
    if num_surface_samples > num_samples_for_computation:
        indices = np.linspace(0, num_surface_samples, num=num_samples_for_computation, endpoint=False, dtype=int)
        surface_samples = surface_samples[indices,:]
        normals = normals[indices,:]
        num_surface_samples = num_samples_for_computation

    # patch centers
    patch_centers, patch_center_indices = farthest_point_sampling(surface_samples[:,:3], K=num_patches) # patch_centers: num_patches , 3

    normals = normals[patch_center_indices,:]
    #print(normals)
    # patch rotations
    patch_rotations = np.array([_get_rotation_from_normal(normal) for normal in normals]) # num_patches , 3

    # patch scales
    index_helper = np.arange(num_surface_samples)
    distances_to_patches = scipy.spatial.distance.cdist(surface_samples[:,:3], patch_centers, metric="euclidean") # num_surface_samples x num_patches
    closest_patches = np.argmin(distances_to_patches, axis=1) # shape: num_surface_samples
    filtered_distances = np.zeros((num_surface_samples, num_patches), dtype=np.float32)
    filtered_distances[index_helper,closest_patches] = distances_to_patches[index_helper,closest_patches]
    patch_scales = np.max(filtered_distances, axis=0) # shape: num_patches
    patch_scales *= final_scaling_increase_factor
    patch_scales = patch_scales[:,None]

    # patch constants
    patch_constants = np.ones((num_patches,1))
    return patch_centers, patch_rotations, patch_scales, patch_constants # num_patches x 3, num_patches x 3, num_patchesx3, num_patchesx1






def initialize_mixture_latent_vector(specs, sdf_samples_with_normals=None, sdf_filename=None, overwrite_init_file=False,
                                     use_precomputed_init=None, initial_patch_latent=None):
    pass

    '''param settings'''
    patch_latent_size = specs["PatchCodeLength"]
    num_patches = specs["NetworkSpecs"]["num_patches"]
    final_scaling_increase_factor = 1.
    num_samples_for_computation = 30000

    '''read in pointcloud with normal'''
    if sdf_samples_with_normals is None:
        if use_precomputed_init:
            initialization_file = sdf_filename + "_init_" + str(patch_latent_size) + "_" + str(num_patches) + "_" + str(final_scaling_increase_factor)  + ".npy"
            return torch.from_numpy(np.load(initialization_file))
        else:
            initialization_file = sdf_filename
            pts = 6 * 10000
            point_cloud = np.load(initialization_file)
            on_surface_pts = point_cloud[:pts // 3, :]

        sdf_samples_with_normals = on_surface_pts

    '''Initialize the extrinsics from data'''
    patch_centers, patch_rotations, patch_scales, patch_constants = initial_metadata_from_sdf_samples(sdf_samples_with_normals[:, 0:3],
                                                                                     sdf_samples_with_normals[:, 3:6],
                                                                                     num_patches=num_patches,
                                                                                     num_samples_for_computation=num_samples_for_computation,
                                                                                     final_scaling_increase_factor=final_scaling_increase_factor)

    patch_extrinsics = np.concatenate([patch_centers,patch_rotations,patch_scales],axis=1) # num_patch,10
    # patch_center:    b,num_nodes,3
    # patch_rot:       b,num_nodes,3  #ZYX
    # patch_scale:     b,num_nodes,3
    # patch_constant:  b,num_nodes,1  #not using for now
    '''random initialization'''
    if False:
        print("using completely random initialization. are you sure?")
        patch_centers = np.random.uniform(-1,1,(num_patches,3))
        patch_rot = np.ones((num_patches,3))
        patch_scale = 0.15*np.ones((num_patches,1))
        patch_constants = np.ones((num_patches,1))
        patch_extrinsics = np.concatenate([patch_centers, patch_rot, patch_scale],
                                          axis=1)  # num_patch,7

    '''Initialize the latent'''
    patch_latents = [np.random.normal(0,1,patch_latent_size)  for _ in range(num_patches)] # [(latent_size,) ... ],num_ptch
    #patch_latents= [np.random.uniform(-1, 1 , patch_latent_size) for _ in range(num_patches)]
    patch_latents = np.concatenate([patch_latents],0).astype(np.float32) #num_patch,latent_size


    '''save the latent'''
    latent = np.concatenate([patch_latents,patch_extrinsics],axis=1)
    if sdf_filename is not None and (not os.path.exists(initialization_file) or overwrite_init_file):
        np.save(initialization_file, latent)

    return torch.from_numpy(latent)  # num_patch, latent_size+10





def convert_embedding_to_explicit_params(embedding, num_nodes, scaling_type='isotropic', max_blob_radius=cfg.max_blob,center_scale=1.0):
    # def convert_embedding_to_explicit_params(embedding, rotated2gaps, num_nodes, scaling_type, max_blob_radius=0.05, center_scale=0.5, max_constant=1.0):
    batch_size = embedding.shape[0]   # batch, num_pred(num_nodes*10)
    embedding = embedding.view(batch_size, num_nodes, 7)

    center = embedding[:, :, 0:3]
    rotation = embedding[:, :, 3:6]   #angle-axis            #can use
    scale = embedding[:, :, 6][:,:,None]                            #only one needed

    constant = embedding[:, :, 0][:,:,None]    #importance   #no longer used...


    if scaling_type == "anisotropic":
        #scale = torch.sigmoid(scale) * max_blob_radius
        scale = 0.7*scale
        # TODO: Doesn't support general augmentation, not reliable!
    elif scaling_type == "isotropic":
        scale = scale[:, :, 0].view(batch_size, num_nodes, 1).expand(-1, -1, 3)
        #scale = torch.sigmoid(scale[:, :, 0].view(batch_size, num_nodes, 1).expand(-1, -1, 3))*0.2
        #scale = torch.sigmoid(scale) * max_blob_radius
    else:
        scale = torch.ones_like(scale) * max_blob_radius

    #constant = -torch.abs(constant)
    max_constant =1
    # constant = -torch.min(torch.abs(constant), max_constant * torch.ones_like(constant))
    constant = torch.sigmoid(constant) * max_constant
    center = center * center_scale
    center = center.view(batch_size, num_nodes, 3)

    # We represent rotations in axis-angle notation.
    #rotation = kornia.angle_axis_to_rotation_matrix(rotation.view(batch_size * num_nodes, 3))
    eulers = list(torch.split(rotation.reshape(-1,3),split_size_or_sections=1,dim=0))  # B* num_node, 3
    rotation = [_get_rotation_from_euler(euler.squeeze(0)) for euler in eulers]# list of (3,3) len=B*num_nodes
    rotation =torch.stack(rotation,dim=0) #B*num_nodes, 3,3
    rotation = rotation.view(batch_size, num_nodes, 3, 3) # B,num_node,3,3

    return constant, scale, rotation, center




def vis_geo(xyz_normal_sel_vis,embedding,epoch,num_node,save_path,k):
    B = xyz_normal_sel_vis.shape[0]
    # view_omegas = extract_view_omegas_from_embedding(embedding, cfg.num_nodes)
    constants, scales, rotations, centers = convert_embedding_to_explicit_params(embedding, num_node)  # batch_size, num_nodes ( () ,(3), (3,3), (3))
    xyz_normal_sel_vis = xyz_normal_sel_vis.permute(0, 2, 1, 3)  # b,num_balls,num_point,6
    if True:
        for i in range(B):
            ellipsoids = createBatchEllipsoids(num_node, constants[i],
                                                    centers[i, :, :],
                                                    scales[i],
                                                    rotations[i])

            # Create an Open3D PointCloud object

            # '''

            for idx, xyz in enumerate(xyz_normal_sel_vis[i].cpu().detach().numpy()):
                pcd = o3d.geometry.PointCloud()
                num_pts = xyz.shape[0]
                pcd.points = o3d.utility.Vector3dVector(xyz[:num_pts//2, 0:3])
                pcd.colors = o3d.utility.Vector3dVector(colors_rgb[idx][None].expand(num_pts//2, -1) / 255)

                ellipsoids.append(pcd)


            #o3d.visualization.draw_geometries(ellipsoids)
            #'''
            # Create a visualizer object
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            ctr = vis.get_view_control()
            for geometry in ellipsoids:
                vis.add_geometry(geometry)
                vis.update_geometry(geometry)


            parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-02-02-18-56-46.json")
            #print(parameters.extrinsic)
            ctr.convert_from_pinhole_camera_parameters(parameters,True)
            vis.poll_events()
            vis.update_renderer()
            vis.get_render_option().load_from_json("RenderOption.json")

            #vis.update_renderer()
            #vis.draw_geometries(ellipsoids)
            #vis.run()
            vis.capture_screen_image(save_path+"pic/vis_geo_"+str(k)+".png",do_render=True)
            vis.destroy_window()
            #'''

def vis_mesh(save_path,epoch,k,mode='train'):
        #o3d.visualization.draw_geometries(ellipsoids)
        #'''
        # Create a visualizer object
        mesh_file_path = (save_path +'obj/'+ str(epoch) + "_" + str(mode) + "_" + str(k) + ".obj")
        mesh = o3d.io.read_triangle_mesh(mesh_file_path)
        mesh.compute_vertex_normals()

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        ctr = vis.get_view_control()

        vis.add_geometry(mesh)
        vis.update_geometry(mesh)


        parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-02-02-18-56-46.json")
        #print(parameters.extrinsic)
        ctr.convert_from_pinhole_camera_parameters(parameters,True)
        vis.poll_events()
        vis.update_renderer()
        vis.get_render_option().load_from_json("RenderOption.json")

        #vis.update_renderer()
        #vis.draw_geometries(ellipsoids)
        #vis.run()
        vis.capture_screen_image(save_path+"pic/vis_recon_"+str(k)+".png",do_render=True)
        vis.destroy_window()
        #'''


def vis_geo_loss(coords,scaled_center_distances,embedding,epoch,num_node):
    B = coords.shape[0]
    # view_omegas = extract_view_omegas_from_embedding(embedding, cfg.num_nodes)
    constants, scales, rotations, centers = convert_embedding_to_explicit_params(embedding,
                                                                                 num_node)  # batch_size, num_nodes ( () ,(3), (3,3), (3))
    num_pts = coords.shape[1]
    #os_pts = coords[:, :,None,:].expand(-1, -1, num_node, -1)
    k=scales[:,:,0].unsqueeze(1).expand(-1,num_pts,-1)
    #k=0.15
    uncoverd_pts_mask = scaled_center_distances > k# if  cover than cover!
    uncoverd_pts_mask,_ = torch.min(uncoverd_pts_mask,dim=-1)
    os_pts = coords.reshape(-1, 3)
    uncoverd_pts_mask = uncoverd_pts_mask.reshape(-1)
    colors = torch.zeros_like(os_pts)
    #colors = colors.reshape(-1, 3)

    colors[uncoverd_pts_mask,2] = 1
    colors[~uncoverd_pts_mask,0] = 1
    os_pts = os_pts.reshape(B,-1,3).clone().detach().cpu().numpy()
    colors = colors.reshape(B,-1,3).clone().detach().cpu().numpy()


    if True:
        for i in range(B):
            ellipsoids = createBatchEllipsoids(num_node, constants[i],
                                                    centers[i, :, :],
                                                    scales[i],
                                                    rotations[i])

            # Create an Open3D PointCloud object

            # '''
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(os_pts[i])
            pcd.colors = o3d.utility.Vector3dVector(colors[i])

            ellipsoids.append(pcd)

            # '''
            o3d.visualization.draw_geometries(ellipsoids)





if __name__ == '__main__':
    args = parse_args()
    main(args)
