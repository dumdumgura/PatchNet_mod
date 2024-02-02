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

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from torch.utils.tensorboard import SummaryWriter
from logger import  init_tb_logger,MessageLogger

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
    parser.add_argument('--batch_size', type=int, default=2, help='batch size in training')
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


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    experienment_name = 'SIF_overfit8'


    '''HYPER PARAMETER'''
    #
    print(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", int(args.gpu))


    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('template_regression')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    #train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    #test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    from data_utils.mydatasets import ShapeNet
    folder='/home/umaru/praktikum/changed_version/Pointnet_Pointnet2_pytorch_current/data/sdf_test'
    train_dataset = ShapeNet(folder, split="train", type='sdf')
    test_dataset = ShapeNet(folder, split="val", type='sdf')

    trainDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
    #testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    '''MODEL LOADING'''
    num_class = args.num_category
    #model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_regression.py', str(exp_dir))

    from models.pointnet2_reg_ssg import LDIF
    from models.loss import SamplerLoss
    model = LDIF(num_class,cfg.supervision)
    criterion = SamplerLoss()
    #model.apply(inplace_relu)

    if not args.use_cpu:
        if torch.cuda.is_available():

            model = model.to(device)
            criterion = criterion.to(device)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        params = [
            # Adjust the learning rate as needed
            {'params': model.multi_mlp.parameters(), 'lr': cfg.MLP_LR},  # shared layer
            # Set a smaller learning rate for the transformer
            # {'params': model.weight_groups.parameters(), 'lr': 0.00001}, #shared initialization
        ]
        optimizer_MLP = torch.optim.Adam(
            params,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        params = [
            # Adjust the learning rate as needed
            {'params': model.PointNet_mini.parameters(), 'lr': cfg.MiniPointNet_LR},  # shared layer
            # Set a smaller learning rate for the transformer
            # {'params': model.weight_groups.parameters(), 'lr': 0.00001}, #shared initialization
        ]
        optimizer_PointNetMini = torch.optim.Adam(
            params,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )

        params = [
            # Adjust the learning rate as needed
            {'params': model.tempalte_predictor.parameters(), 'lr': cfg.PointNet_LR},  # pointnet2
            # Set a smaller learning rate for the transformer
            # {'params': model.weight_groups.parameters(), 'lr': 0.00001}, #shared initialization
        ]
        optimizer_PointNet= torch.optim.Adam(
            params,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer_PointNet, step_size=2000, gamma=0)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0


    writer = SummaryWriter("logs/LDIF/"+experienment_name)

    exp_dir = Path('logs/LDIF/'+experienment_name+'/ckpt')
    exp_dir.mkdir(exist_ok=True)

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        model = model.train()

        #scheduler.step()
        #for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        pbar = tqdm(enumerate(trainDataLoader), total=len(trainDataLoader))
        for it,xt in pbar:
            # take in data

            near_surface_pts = xt['near_surface_pts'].to(device)
            on_surface_pts = xt['on_surface_pts'].to(device)
            grid = xt['grid'].to(device)

            if cfg.supervision == 'sdf':
                coords = near_surface_pts[:, :, :3]
            else:
                coords = on_surface_pts[:, :, :3]
                coords.requires_grad = True


            #zero grads of model and optimizier
            model.zero_grad()
            optimizer_MLP.zero_grad()
            optimizer_PointNet.zero_grad()
            optimizer_PointNetMini.zero_grad()
            #batch_size,num_pts,_ = coords_ns.shape


            # data augmentation
            #points = points.data.numpy()
            #points = provider.random_point_dropout(points)
            #points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            #points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            #points = torch.Tensor(points)
            #points = points.transpose(2, 1)

            #if not args.use_cpu:
            #    points, target = points.cuda(), target.cuda()

            pred_sdf,embedding,latent_list,ldif_gaussian,xyz_normal_sel_vis = model(on_surface_pts,coords) # pred.shape = Batch,num_ball*params




            loss,loss_dict = criterion(embedding,pred_sdf,ldif_gaussian,xyz_normal_sel_vis,near_surface_pts,on_surface_pts,coords,grid,epoch)

            #pred_choice = pred.data.max(1)[1]

            #correct = pred_choice.eq(target.long().data).cpu().sum()
            #mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer_MLP.step()
            #if epoch <=1000:
            optimizer_PointNet.step()
            optimizer_PointNetMini.step()

            writer.add_scalar('loss_total', loss, global_step)
            for key in loss_dict:
                writer.add_scalar(key, loss_dict[key], global_step)


            global_step += 1

        if epoch % cfg.vis_epoch == 0:
            # eval process
            meshes = model.get_mesh(embedding, latent_list)
            model.reconstruct_shape(meshes, epoch)
            vis_geo(near_surface_pts,
                    xyz_normal_sel_vis,
                    embedding,
                    epoch
                    )

            logger.info('Save model...')

            savepath = 'logs/LDIF/'+experienment_name+'/ckpt/'+str(epoch)+'.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        #train_instance_acc = np.mean(mean_correct)
        log_string('Train pred loss: %f' % loss)

        '''
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
        '''
        global_epoch += 1
    logger.info('End of training...')


import open3d as o3d

NORMALIZATION_EPS = 1e-8
def createBatchEllipsoids(n, scale_constants, centers, semi_axes, rotation_matrices):
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
    scales = np.where(scale_constants > NORMALIZATION_EPS, scale_constants,
                      NORMALIZATION_EPS * np.ones_like(scale_constants))  # (bs, num_nodes, 3)
    inv_scales = 1.0 / scales
    # Scale and semi-axes adjustment
    semi_axes = np.array(semi_axes)
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
        wireFrame.paint_uniform_color(colors_rgb[i] / 255)  # Color the wireframe
        meshes.append(wireFrame)

    return meshes

from models.node_proc import convert_embedding_to_explicit_params
def vis_geo(near_surface_pts,
            xyz_normal_sel_vis,
            embedding,
            epoch):
    B, num_pts, _ = near_surface_pts.shape
    num_node = embedding.shape[1] // 11
    # view_omegas = extract_view_omegas_from_embedding(embedding, cfg.num_nodes)
    constants, scales, rotations, centers = convert_embedding_to_explicit_params(embedding,
                                                                                 num_node)  # batch_size, num_nodes ( () ,(3), (3,3), (3))

    if cfg.use_mirror:
        flip_axis = 0
        constants_dup = constants[:,:num_node//2].clone()
        scales_dup = scales[:,:num_node//2,:].clone()
        # scales_dup[:,:,flip_axis] = -scales_dup[:,:,flip_axis]
        rotations_dup = rotations[:,:num_node//2,:].clone()
        centers_dup = centers[:,:num_node//2,:].clone()

        rotations_dup[:, :, flip_axis, :] = -rotations_dup[:, :, flip_axis, :]
        centers_dup[:, :, flip_axis] = -centers_dup[:, :, flip_axis]

        constants = torch.cat([constants, constants_dup], dim=1)
        scales = torch.cat([scales, scales_dup], dim=1)
        rotations = torch.cat([rotations, rotations_dup], dim=1)
        centers = torch.cat([centers, centers_dup], dim=1)

        num_node = num_node + num_node//2
    #uniform_samples = near_surface_pts[:, -num_pts // 2:, 0:4]
    near_surface_samples = near_surface_pts[:, 0:num_pts // 2, 0:4]

    xyz_normal_sel_vis = xyz_normal_sel_vis.permute(0, 2, 1, 3)  # b,num_balls,num_point,6
    if epoch % cfg.vis_epoch == 0 and epoch > -1:
        for i in range(B):

            ellipsoids = createBatchEllipsoids(num_node, constants[i][:, None].cpu().detach().numpy(),
                                                    centers[i, :, :].cpu().detach().numpy(),
                                                    scales[i].cpu().detach().numpy(),
                                                    rotations[i].cpu().detach().numpy())

            # Create an Open3D PointCloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(near_surface_samples[i, :, 0:3].cpu().detach().numpy())
            pcd.colors = o3d.utility.Vector3dVector(
                0.5 * np.ones_like(near_surface_samples[i, :, 0:3].cpu().detach().numpy()))
            #o3d.visualization.draw_geometries([pcd])
            # Visualize all ellipsoids
            ellipsoids.append(pcd)

            # '''

            for idx, xyz in enumerate(xyz_normal_sel_vis[i].cpu().detach().numpy()):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz[:, 0:3])
                pcd.colors = o3d.utility.Vector3dVector(colors_rgb[idx][None].expand(xyz.shape[0], -1) / 255)

                ellipsoids.append(pcd)

            # '''
            o3d.visualization.draw_geometries(ellipsoids)


if __name__ == '__main__':
    args = parse_args()
    main(args)
