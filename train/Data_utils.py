import os
import pickle
import torch
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from PIL import Image
from tqdm import tqdm
import open3d
import numpy as np


def normalize_point_cloud(point_cloud):
    """
    Normalize a point cloud to fit within the range [-1, 1].
    """
    if torch.all(point_cloud == 0):
        return point_cloud

    centroid = point_cloud.mean(dim=0)
    point_cloud_centered = point_cloud - centroid

    max_distance = point_cloud_centered.abs().max()
    normalized_point_cloud = point_cloud_centered / max_distance

    return normalized_point_cloud


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape

    if N == 0:
        return np.zeros((npoint, 3))

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def process_batch_pointclouds(points_batch, npoint, n_jobs=-1):
    """
    Processes a batch of point clouds in parallel using farthest point sampling.

    Input:
        points_batch: list or array of point clouds, each of shape [N, D]
        npoint: number of points to sample for each point cloud
        n_jobs: number of CPU cores to use (default: -1, which uses all available cores)
    Return:
        sampled_points_batch: list of sampled point clouds
    """
    # Parallelize over each point cloud in the batch
    with tqdm_joblib(tqdm(desc="Processing Point Clouds", total=len(points_batch))) as progress_bar:
        sampled_points_batch = Parallel(n_jobs=n_jobs)(
            delayed(farthest_point_sample)(points, npoint) for points in points_batch
        )
    return sampled_points_batch


# Divide point cloud into four parts
def divide_pc(points):

    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    mid_x = (min_bound[0] + max_bound[0]) / 2
    mid_y = (min_bound[1] + max_bound[1]) / 2

    part1 = points[(points[:, 0] < mid_x) & (points[:, 1] < mid_y)]
    part2 = points[(points[:, 0] >= mid_x) & (points[:, 1] < mid_y)]

    part3 = points[(points[:, 0] < mid_x) & (points[:, 1] >= mid_y)]
    part4 = points[(points[:, 0] >= mid_x) & (points[:, 1] >= mid_y)]

    return part1, part2, part3, part4


class DescriptionPointCloud(Dataset):
    def __init__(self, data_description_path, data_scene_path, data_pc_path, num_pc, mem=True, fps=False,
                 customized_margin=False):
        self.description_path = data_description_path
        self.data_pov_path = data_scene_path
        self.data_pc_path = data_pc_path
        available_data = open('data/available_data.txt', 'r')
        self.samples = [x[:-1] for x in available_data.readlines()]
        self.mem = mem
        self.fps = fps
        self.margin_needed = customized_margin
        if self.mem:
            print('Data Loading ...')

            print('Loading descriptions ...')
            if os.path.exists('./data/descs.pkl'):
                pickle_file = open('./data/descs.pkl', 'rb')
                self.descs = pickle.load(pickle_file)
                pickle_file.close()
            else:
                self.descs = []
                for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                    self.descs.append(torch.load(self.description_path + os.sep + s + '.pt'))
                pickle_file = open('./data/descs.pkl', 'wb')
                pickle.dump(self.descs, pickle_file)
                pickle_file.close()

            print('Loading POVs ...')
            if self.data_pov_path is not None:
                if os.path.exists('./data/pov_images.pkl'):
                    pickle_file = open('./data/pov_images.pkl', 'rb')
                    self.pov_images = pickle.load(pickle_file)
                    pickle_file.close()
                else:
                    self.pov_images = []
                    for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                        self.pov_images.append(torch.load(self.data_pov_path + os.sep + s + '.pt'))
                    pickle_file = open('./data/pov_images.pkl', 'wb')
                    pickle.dump(self.pov_images, pickle_file)
                    pickle_file.close()

            print('Loading Point Clouds ...')
            if self.data_pc_path is not None:
                data_file = f'./data/pc_{num_pc}_fps_div.pkl' if self.fps else f'./data/pc_{num_pc}_uniform_div.pkl'
                if os.path.exists(data_file):
                    pickle_file = open(data_file, 'rb')
                    self.pc = pickle.load(pickle_file)
                    pickle_file.close()
                else:
                    self.pc = []
                    for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                        pcd = open3d.io.read_point_cloud(self.data_pc_path + os.sep + s + '.ply')
                        pcd_numpy = np.asarray(pcd.points)
                        pcd_numpy = pcd_numpy[~np.isnan(pcd_numpy).any(axis=1)]

                        if not self.fps:
                            if num_pc < len(pcd.points):
                                pcd = pcd.uniform_down_sample(int(pcd_numpy.shape[0] / num_pc))
                                pcd_tensor = torch.from_numpy(np.asarray(pcd.points))
                            else:
                                pcd_tensor = torch.from_numpy(pcd_numpy)
                            if torch.isnan(pcd_tensor).any():
                                valid_rows_mask = ~torch.isnan(pcd_tensor).any(dim=1)
                                pcd_tensor = pcd_tensor[valid_rows_mask]

                            self.pc.append(normalize_point_cloud(point_cloud=pcd_tensor))

                        else:
                            self.pc.append(pcd_numpy)

                    if self.fps:
                        total_cores = os.cpu_count()
                        print(f"Total CPU cores available: {total_cores}")
                        pc_div = [item for x in self.pc for item in divide_pc(x)]

                        sampled_points_batch = process_batch_pointclouds(pc_div, num_pc // 4, n_jobs=-1)
                        self.pc = [normalize_point_cloud(torch.from_numpy(x)) for x in sampled_points_batch]

                        tmp_pc = []
                        for i in range(0, len(self.pc), 4):
                            tmp_pc.append(torch.stack((self.pc[i], self.pc[i+1], self.pc[i+2], self.pc[i+3]), dim=1))

                        count_prob = 0
                        for i in range(len(tmp_pc)):
                            if torch.nonzero(torch.isnan(tmp_pc[i]), as_tuple=False).shape[0] >= 1:
                                nan_indices = torch.nonzero(torch.isnan(tmp_pc[i]), as_tuple=False)
                                print('i'*10, i)
                                print('nan indices', nan_indices)
                                count_prob += 1
                        print(count_prob)

                        self.pc = tmp_pc
                        pickle_file = open(f'./data/pc_{num_pc}_fps_div.pkl', 'wb')
                    else:
                        pickle_file = open(f'./data/pc_{num_pc}_uniform_div.pkl', 'wb')

                    pickle.dump(self.pc, pickle_file)
                    pickle_file.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mem:
            desc_tensor = self.descs[index]
            scene_img_tensor = self.pov_images[index]
            pc_tensor = self.pc[index]
        else:
            desc_tensor = torch.load(self.description_path + os.sep + self.samples[index] + '.pt')
            scene_img_tensor = torch.load(self.data_pov_path + os.sep + self.samples[index] + '.pt')
            pcd = open3d.io.read_point_cloud(self.data_pc_path + os.sep + self.samples[index] + '.ply')
            pc_tensor = torch.from_numpy(np.asarray(pcd.points))
        return desc_tensor, scene_img_tensor, pc_tensor, index


class DescriptionScene(Dataset):
    '''
    fps: use Farthest Point Sampling
    '''
    def __init__(self, data_description_path, data_scene_path, data_pc_path, num_pc, mem=True, fps=False,
                 customized_margin=False):
        self.description_path = data_description_path
        self.data_pov_path = data_scene_path
        self.data_pc_path = data_pc_path
        available_data = open('data/available_data.txt', 'r')
        self.samples = [x[:-1] for x in available_data.readlines()]
        self.mem = mem
        self.fps = fps
        self.margin_needed = customized_margin
        if self.mem:
            print('Data Loading ...')

            print('Loading descriptions ...')
            if os.path.exists('./data/descs.pkl'):
                pickle_file = open('./data/descs.pkl', 'rb')
                self.descs = pickle.load(pickle_file)
                pickle_file.close()
            else:
                self.descs = []
                for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                    self.descs.append(torch.load(self.description_path + os.sep + s + '.pt'))
                pickle_file = open('./data/descs.pkl', 'wb')
                pickle.dump(self.descs, pickle_file)
                pickle_file.close()

            print('Loading POVs ...')
            if self.data_pov_path is not None:
                if os.path.exists('./data/pov_images.pkl'):
                    pickle_file = open('./data/pov_images.pkl', 'rb')
                    self.pov_images = pickle.load(pickle_file)
                    pickle_file.close()
                else:
                    self.pov_images = []
                    for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                        self.pov_images.append(torch.load(self.data_pov_path + os.sep + s + '.pt'))
                    pickle_file = open('./data/pov_images.pkl', 'wb')
                    pickle.dump(self.pov_images, pickle_file)
                    pickle_file.close()

            print('Loading Point Clouds ...')
            if self.data_pc_path is not None:
                data_file = f'./data/pc_{num_pc}_fps.pkl' if self.fps else f'./data/pc_{num_pc}_uniform.pkl'
                if os.path.exists(data_file):
                    pickle_file = open(data_file, 'rb')
                    self.pc = pickle.load(pickle_file)
                    pickle_file.close()
                else:
                    self.pc = []
                    for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                        pcd = open3d.io.read_point_cloud(self.data_pc_path + os.sep + s + '.ply')
                        pcd_numpy = np.asarray(pcd.points)
                        pcd_numpy = pcd_numpy[~np.isnan(pcd_numpy).any(axis=1)]

                        if not self.fps:
                            if num_pc < len(pcd.points):
                                pcd = pcd.uniform_down_sample(int(pcd_numpy.shape[0] / num_pc))
                                pcd_tensor = torch.from_numpy(np.asarray(pcd.points))
                            else:
                                pcd_tensor = torch.from_numpy(pcd_numpy)
                            if torch.isnan(pcd_tensor).any():
                                valid_rows_mask = ~torch.isnan(pcd_tensor).any(dim=1)
                                pcd_tensor = pcd_tensor[valid_rows_mask]

                            self.pc.append(normalize_point_cloud(point_cloud=pcd_tensor))

                        else:
                            self.pc.append(pcd_numpy)

                    if self.fps:
                        total_cores = os.cpu_count()
                        print(f"Total CPU cores available: {total_cores}")
                        sampled_points_batch = process_batch_pointclouds(self.pc, num_pc, n_jobs=-1)
                        self.pc = [normalize_point_cloud(torch.from_numpy(x)) for x in sampled_points_batch]
                        pickle_file = open(f'./data/pc_{num_pc}_fps.pkl', 'wb')
                    else:
                        pickle_file = open(f'./data/pc_{num_pc}_uniform.pkl', 'wb')

                    pickle.dump(self.pc, pickle_file)
                    pickle_file.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mem:
            desc_tensor = self.descs[index]
            scene_img_tensor = self.pov_images[index]
            pc_tensor = self.pc[index]
        else:
            desc_tensor = torch.load(self.description_path + os.sep + self.samples[index] + '.pt')
            scene_img_tensor = torch.load(self.data_pov_path + os.sep + self.samples[index] + '.pt')
            pcd = open3d.io.read_point_cloud(self.data_pc_path + os.sep + self.samples[index] + '.ply')
            pc_tensor = torch.from_numpy(np.asarray(pcd.points))
        return desc_tensor, scene_img_tensor, pc_tensor, index


class VAEDataset(Dataset):
    def __init__(self, data_floor_plan_path, transform, mem=False):
        counter = 0
        self.floor_plan_path = data_floor_plan_path
        self.transform = transform
        self.samples = os.listdir(self.floor_plan_path)
        self.mem = mem
        if self.mem:
            print('Data Loading ...')
            if os.path.exists('./data/images.pt'):
                self.images = torch.load('./data/images.pt')
            else:
                self.images = torch.empty((len(self.samples), 3, 224, 224), dtype=torch.float32)
                for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                    try:
                        image = Image.open(self.floor_plan_path + os.sep + s).convert('RGB')
                    except:
                        counter += 1
                        image = Image.new("RGB", (224, 224), (0, 0, 0))
                        print('counter = ', counter)
                        print('name = ', s)
                    self.images[idx] = self.transform(image)
                torch.save(self.images, './data/images.pt')
            print('counter = ', counter)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mem:
            return self.images[idx]
        else:
            try:
                image = Image.open(self.floor_plan_path + os.sep + self.samples[idx] + '.png').convert('RGB')
            except:
                image = Image.new("RGB", (224, 224), (0, 0, 0))
            img_tensor = self.transform(image)
            return img_tensor
