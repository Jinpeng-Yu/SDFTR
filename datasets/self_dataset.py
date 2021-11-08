from numpy.core.fromnumeric import choose
from numpy.lib.shape_base import split
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from plyfile import PlyData
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor,
                            generalized_box3d_iou)
import utils.pc_util as pc_util
from utils.pc_util import scale_points, shift_scale_points
from utils.random_cuboid import RandomCuboid

DATASET_ROOT_DIR = '/p300/SDFTransformer/datasets/selfdata'
DATASET_METADATA_DIR = '/p300/SDFTransformer/datasets/meta_data'
MAX_NUM_POINT = 50000

class SelfdataDatasetConfig(object):
    def __init__(self):
        # I Need
        self.num_angle_bin = 12
        self.max_num_obj = 16

        self.num_semcls = 10
        self.type2class = {
            "bed": 0,
            "table": 1,
            "sofa": 2,
            "chair": 3,
            "toilet": 4,
            "desk": 5,
            "dresser": 6,
            "night_stand": 7,
            "bookshelf": 8,
            "bathtub": 9,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.type2onehotclass = {
            "bed": 0,
            "table": 1,
            "sofa": 2,
            "chair": 3,
            "toilet": 4,
            "desk": 5,
            "dresser": 6,
            "night_stand": 7,
            "bookshelf": 8,
            "bathtub": 9,
        }

    def angle2class(self, angle):
        """Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        returns class [0,1,...,N-1] and a residual number such that
            class*(2pi/N) + number = angle
        """
        num_class = self.num_angle_bin
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (
            class_id * angle_per_class + angle_per_class / 2
        )
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """Inverse function to angle2class"""
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle
    
    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle_batch(pred_cls, residual, to_label_format)

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    def my_compute_box_3d(self, center, size, heading_angle):
        # for x-forward, y-right, z-up coordinate, means the cube rotate clockwise along the z axis
        R = pc_util.rotz(-1 * heading_angle)
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)

    # @staticmethod
    # def rotate_aligned_boxes(input_boxes, rot_mat):
    #     centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
    #     new_centers = np.dot(centers, np.transpose(rot_mat))

    #     dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
    #     new_x = np.zeros((dx.shape[0], 4))
    #     new_y = np.zeros((dx.shape[0], 4))

    #     for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
    #         crnrs = np.zeros((dx.shape[0], 3))
    #         crnrs[:, 0] = crnr[0] * dx
    #         crnrs[:, 1] = crnr[1] * dy
    #         crnrs = np.dot(crnrs, np.transpose(rot_mat))
    #         new_x[:, i] = crnrs[:, 0]
    #         new_y[:, i] = crnrs[:, 1]

    #     new_dx = 2.0 * np.max(new_x, 1)
    #     new_dy = 2.0 * np.max(new_y, 1)
    #     new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

    #     return np.concatenate([new_centers, new_lengths], axis=1)

class SelfdataDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set='train',
        root_dir=None,
        meta_data_dir = None,
        num_points=20000,
        use_color=False,
        use_height=False,
        augment=False,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
    ):

        self.dataset_config = dataset_config
        assert split_set in ['train', 'val']
        if root_dir is None:
            root_dir = DATASET_ROOT_DIR

        if meta_data_dir is None:
            meta_data_dir = DATASET_METADATA_DIR

        self.data_path = root_dir
        all_scene_names = list(
            set(
                [
                    os.path.basename(x)[0:9]
                    for x in os.listdir(self.data_path)
                    if x.startswith('scene')
                ]
            )
        )
        if split_set == 'all':
            self.scene_names = all_scene_names
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join(meta_data_dir, f'selfdata_{split_set}.txt')
            with open(split_filenames, 'r') as f:
                self.scene_names = f.read().splitlines()
            # remove unavailiable scans
            num_scenes = len(self.scene_names)
            self.scene_names = [
                sname for sname in self.scene_names if sname in all_scene_names
            ]
            print(f"kept {len(self.scene_names)} scenes out of {num_scenes}")
        else:
            raise ValueError(f'Unknown split name {split_set}')
        
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1,3), dtype=np.float32),
            np.ones((1,3), dtype=np.float32),
        ]
        # self.center_normalizing_range = [
        #     -np.ones((1,3), dtype=np.float32),
        #     np.ones((1,3), dtype=np.float32),
        # ]


        self.max_num_obj = 16

    def scene_params(self, bg_num, fg_num):
        # params array of background cube number
        # x0, y0, z0 [-1.0, 1.0)
        delta = np.random.rand(bg_num, 3) * 2 - 1.0
        # scale_x, scale_y, scale_z [0.5,2.0)
        scale = np.random.rand(bg_num, 3) * 1.5 + 0.5
        # theta [0,2*pi)
        theta = np.random.rand(bg_num, 1) * 2 * np.pi

        bg_params = np.concatenate((delta, scale, theta), axis=1)

        # params array of foreground cube number
        # x0, y0, z0 [-1.0, 1.0)
        delta = np.random.rand(fg_num, 3) * 2 - 1.0
        # scale_x, scale_y, scale_z [0.5,2.0)
        scale = np.random.rand(fg_num, 3) * 1.5 + 0.5
        # theta [0,2*pi)
        theta = np.random.rand(fg_num, 1) * 2 * np.pi

        fg_params = np.concatenate((delta, scale, theta), axis=1)

        # params array of target
        # tg_params = np.concatenate((bg_params, fg_params), axis = 0)

        return bg_params, fg_params

    def read_mesh_vertices(self, filename):
        """ read XYZ for each vertex.
        """
        assert os.path.isfile(filename)
        with open(filename, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
            vertices[:,0] = plydata['vertex'].data['x']
            vertices[:,1] = plydata['vertex'].data['y']
            vertices[:,2] = plydata['vertex'].data['z']
        return vertices

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):
        self.augment = False
        scene_name = self.scene_names[idx]

        # # random seed is OK
        # print(np.random.ranf([3,3]))
        # print(np.random.randint(2, size=(3,3)))
        # exit()

        # bg and fg cubes number 16
        bg_num , fg_num = 16, 16
        bg_params, fg_params = self.scene_params(bg_num, fg_num)
        # print(f'bg_params: {bg_params}')
        # print(f'fg_params: {fg_params}')
        # exit()

        bboxes = fg_params # K x 7 different from sunrgbd K x 8 because of non 'class'
        bboxes[:, 3:6] = bboxes[:, 3:6] * 0.1 # basic cube sdf has semi_x = 0.1
        # ------------------------------- LABELS ------------------------------
        angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
        angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_angles = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_sizes = np.zeros((self.max_num_obj, 3), dtype=np.float32)
        label_mask = np.zeros((self.max_num_obj))
        label_mask[0 : bboxes.shape[0]] = 1
        max_bboxes = np.zeros((self.max_num_obj, 8))
        max_bboxes[0 : bboxes.shape[0], :] = bboxes

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((self.max_num_obj, 6))

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            # semantic_class = bbox[7]         # without semantic class
            raw_angles[i] = bbox[6] % 2 * np.pi
            box3d_size = bbox[3:6] * 2
            raw_sizes[i, :] = box3d_size
            angle_class, angle_residual = self.dataset_config.angle2class(bbox[6])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            corners_3d = self.dataset_config.my_compute_box_3d(
                bbox[0:3], bbox[3:6], bbox[6]
            )
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                    zmax - zmin,
                ]
            )
            target_bboxes[i, :] = target_bbox
        
        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        gt_decision = np.ones(fg_num)
        print(gt_decision.shape)
        for i in range(fg_num):
            for j in range(i):
                iou = generalized_box3d_iou
                gt_decision[i] = ((iou < 0.75) | (~gt_decision[j].astype(bool))).astype(float) * gt_decision[i]
        
        print(gt_decision)
        exit()

        # bg_mesh_vertices = self.read_mesh_vertices(os.path.join(self.data_path, scene_name, scene_name)+'_bg.ply')
        # tg_mesh_vertices = self.read_mesh_vertices(os.path.join(self.data_path, scene_name, scene_name)+'_target.ply')
        # fg_mesh_vertices = self.read_mesh_vertices(os.path.join(self.data_path, scene_name, scene_name)+'_fg.ply')

        # point_cloud_dims_min = tg_mesh_vertices.min(axis=0)
        # point_cloud_dims_max = tg_mesh_vertices.max(axis=0)
        
        # # all scene in [-1, 1]
        # point_cloud_dims_min = -np.ones_like(point_cloud_dims_min)
        # point_cloud_dims_max = np.ones_like(point_cloud_dims_max)

        # N = bg_mesh_vertices.shape[0]
        # if N > MAX_NUM_POINT:
        #     choice = np.random.choice(N, MAX_NUM_POINT, replace=False)
        #     bg_mesh_vertices = bg_mesh_vertices[choice, :]
        #     tg_mesh_vertices = tg_mesh_vertices[choice, :]
        #     fg_mesh_vertices = fg_mesh_vertices[choice, :]
            
        # instance_bboxes = np.load(os.path.join(self.data_path,scene_name,scene_name)+'_fg.npy')

        # bg_point_cloud = bg_mesh_vertices[:, 0:3]
        # tg_point_cloud = tg_mesh_vertices[:, 0:3]
        # fg_point_cloud = fg_mesh_vertices[:, 0:3] # do not use color for now

        # # --------------------------------- LABELS ---------------------------------
        # MAX_NUM_OBJ = self.dataset_config.max_num_obj
        # target_bboxes = np.zeros((MAX_NUM_OBJ, 9), dtype=np.float32)
        # target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        # angle_classes = np.zeros((MAX_NUM_OBJ,3), dtype= np.int64)
        # angle_residuals = np.zeros((MAX_NUM_OBJ,3), dtype=np.float32)
        # raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        # raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)

        # # if self.augment and self.use_random_cuboid:
        # #     (
        # #         point_cloud,
        # #         instance_bboxes,
        # #         per_point_labels,
        # #     ) = self.random_cuboid_augmentor(
        # #         point_cloud, instance_bboxes
        # #     )

        # bg_point_cloud, choices = pc_util.random_sampling(
        #     bg_point_cloud, self.num_points, return_choices=True
        # )

        # tg_point_cloud, choices = pc_util.random_sampling(
        #     tg_point_cloud, self.num_points, return_choices=True
        # )
        
        # target_bboxes_mask[0 : instance_bboxes.shape[0]] = 1
        # target_bboxes[0 : instance_bboxes.shape[0], :] = instance_bboxes[:, 0:9]

        # ## ---------------------------- DATA AUGMENTATION ----------------------------
        # # if self.augment:

        # #     if np.random.random() > 0.5:
        # #         # Flipping along the YZ plane
        # #         point_cloud[:, 0] = -1 * point_cloud[:, 0]
        # #         target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

        # #     if np.random.random() > 0.5:
        # #         # Flipping along the XZ plane
        # #         point_cloud[:, 1] = -1 * point_cloud[:, 1]
        # #         target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

        # #     # Rotation along up-axis/Z-axis
        # #     rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
        # #     rot_mat = pc_util.rotz(rot_angle)
        # #     point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
        # #     target_bboxes = self.dataset_config.rotate_aligned_boxes(
        # #         target_bboxes, rot_mat
        # #     )

        # # raw_sizes = target_bboxes[:, 3:6]
        # # Original cube semi_edge is 0.1
        # raw_sizes = np.ones_like(target_bboxes[:, 3:6]).astype(np.float32) / 10 \
        #             * target_bboxes[:, 3:6] * 2

        # # angle size [0, 360)
        # # angle_classes = ((target_bboxes[:, 6:9] + 15) // 30 % 12).astype(np.int64)
        # # angle_residuals = ((((target_bboxes[:, 6:9] + 15) % 360 - angle_classes * 30.) \
        # #                     / 15. * np.pi / 12.) - np.pi / 12).astype(np.float32)
        
        # # angle size [0, 2*pi)
        # angle_classes = ((target_bboxes[:, 6:9] + np.pi / 12) / (np.pi / 6) % 12).astype(np.int64)
        # angle_residuals = (((target_bboxes[:, 6:9] + np.pi / 12 + 1e-15) % (np.pi * 2) \
        #                         - angle_classes * np.pi / 6) - np.pi / 12).astype(np.float32)

        # # box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        # # original cube center is [0., 0., 0.]
        # box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        # box_centers_normalized = shift_scale_points(
        #     box_centers[None, ...],
        #     src_range=[
        #         point_cloud_dims_min[None, ...],
        #         point_cloud_dims_max[None, ...],
        #     ],
        #     dst_range=self.center_normalizing_range,
        # )
        # box_centers_normalized = box_centers_normalized.squeeze(0)
        # box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]

        # mult_factor = point_cloud_dims_max - point_cloud_dims_min
        # box_sizes_normalized = scale_points(
        #     raw_sizes.astype(np.float32)[None, ...],
        #     mult_factor=1.0 / mult_factor[None, ...],
        # )
        # box_sizes_normalized = box_sizes_normalized.squeeze(0)

        # # box_corners = self.dataset_config.box_parametrization_to_corners_np(
        # #     box_centers[None, ...],
        # #     raw_sizes.astype(np.float32)[None, ...],
        # #     raw_angles.astype(np.float32)[None, ...],
        # # )
        # # box_corners = box_corners.squeeze(0)

        # # print(box_centers)
        # # print(box_centers_normalized)
        # # print(f"fg_min:{fg_mesh_vertices.min(axis=0)}")
        # # print(f"bg_min:{bg_mesh_vertices.min(axis=0)}")
        # # print(f"tg_min:{tg_mesh_vertices.min(axis=0)}")
        # # print(f"fg_max:{fg_mesh_vertices.max(axis=0)}")
        # # print(f"bg_max:{bg_mesh_vertices.max(axis=0)}")
        # # print(f"tg_max:{tg_mesh_vertices.max(axis=0)}")
        # # print(raw_sizes)
        # # print(box_sizes_normalized)
        # # print(point_cloud_dims_min)
        # # print(point_cloud_dims_max)
        # # exit()

        # ret_dict = {}
        # ret_dict['bg_point_clouds'] = bg_point_cloud.astype(np.float32)
        # ret_dict['tg_point_clouds'] = tg_point_cloud.astype(np.float32)
        # ret_dict['fg_point_clouds'] = fg_point_cloud.astype(np.float32)
        # ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        # ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
        #     np.float32
        # )
        # ret_dict['gt_angle_class_label'] = angle_classes.astype(np.int64)
        # ret_dict["gt_angle_residual_label"] = angle_residuals.astype(np.float32)
        # ret_dict['gt_box_present'] = target_bboxes_mask.astype(np.float32)
        # ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        # ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        # ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        # ret_dict['point_cloud_dims_min'] = point_cloud_dims_min.astype(np.float32)
        # ret_dict['point_cloud_dims_max'] = point_cloud_dims_max.astype(np.float32)
        # return ret_dict