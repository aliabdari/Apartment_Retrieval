'''
Train Pointnet ++ using pointnet pointnet++ using pytorch repository
'''

import pickle
from DNNs import GRUNet, OneDimensionalCNN
from pointnet2_cls_ssg import classifier_pointnet2_ssg
from pointnet2_cls_msg import classifier_pointnet2_msg
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import StepLR
import train_utility
from Data_utils import DescriptionScene
from tqdm import tqdm
import Constants
import argparse
import provider
import time, wandb
import random
import numpy as np


def set_seed(seed_num):
    np.random.seed(seed_num)
    random.seed(0)
    torch.use_deterministic_algorithms(True)

    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(data):
    # desc
    tmp_description_povs = [x[0] for x in data]
    tmp = pad_sequence(tmp_description_povs, batch_first=True)
    descs_pov = pack_padded_sequence(tmp,
                                     torch.tensor([len(x) for x in tmp_description_povs]),
                                     batch_first=True,
                                     enforce_sorted=False)
    # pov image
    tmp_pov = [x[1] for x in data]
    padded_pov = pad_sequence(tmp_pov, batch_first=True)
    padded_pov = torch.transpose(padded_pov, 1, 2)

    # point clouds
    tmp_pc = [x[2] for x in data]
    padded_pc = pad_sequence(tmp_pc, batch_first=True)
    padded_pc = torch.transpose(padded_pc, 1, 2)

    indexes = [x[3] for x in data]
    return descs_pov, padded_pov, padded_pc, indexes


def start_train(args):
    seed_num = 42
    set_seed(seed_num=seed_num)
    wandb.init(
        project="pointnet++",

        config={
            "learning_rate": args.lr,
            "architecture": "PointNet++",
            "dataset": "3DFRONT",
            "epochs": args.epochs,
        }
    )

    approach_name = Constants.pc_approach + '_' + str(int(time.time()))
    output_feature_size = 256

    is_customized_margin = False

    is_bidirectional = True
    model_desc = GRUNet(hidden_size=output_feature_size, num_features=512, is_bidirectional=is_bidirectional)
    model_pov = OneDimensionalCNN(in_channels=512, out_channels=512, kernel_size=5, feature_size=output_feature_size)
    # model_pc = PointNetBackbone(num_points=args.npc, num_global_feats=output_feature_size, local_feat=True)
    model_pc = classifier_pointnet2_ssg()

    cont_loss = train_utility.LossContrastive(name=approach_name, patience=25, delta=0.0001)

    num_epochs = args.epochs
    batch_size = args.batch_size

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device = ', device)
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    model_desc.to(device=device)
    model_pov.to(device=device)
    model_pc.to(device=device)

    #     data section
    train_indices, val_indices, test_indices = train_utility.retrieve_indices()

    descriptions_path, pov_path, pc_path = train_utility.get_entire_data()

    dataset = DescriptionScene(data_description_path=descriptions_path, data_scene_path=pov_path, data_pc_path=pc_path,
                               mem=True, num_pc=args.npc, fps=args.fps, customized_margin=is_customized_margin)
    train_subset = Subset(dataset, list(train_indices))
    val_subset = Subset(dataset, list(val_indices))
    test_subset = Subset(dataset, list(test_indices))

    g = torch.Generator()
    g.manual_seed(seed_num)

    train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4, worker_init_fn=seed_worker,generator=g)
    # train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, worker_init_fn=seed_worker,generator=g)
    # val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, worker_init_fn=seed_worker,generator=g)
    # test_loader = DataLoader(test_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)

    '''Train Procedure'''
    # params = list(model_desc.parameters()) + list(model_pov.parameters())
    params = list(model_desc.parameters()) + list(model_pc.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    step_size = 27
    gamma = 0.75
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_r10 = 0
    print('Train procedure ...')
    for _ in tqdm(range(num_epochs)):

        if not cont_loss.is_val_improving():
            print('Early Stopping !!!')
            break

        total_loss_train = 0
        total_loss_val = 0
        num_batches_train = 0
        num_batches_val = 0

        output_description_val = torch.empty(len(val_indices), output_feature_size)
        output_pov_val = torch.empty(len(val_indices), output_feature_size)

        # output_description_train = torch.empty(len(train_indices), output_feature_size)
        # output_pov_train = torch.empty(len(train_indices), output_feature_size)

        max_loss = 0
        for i, (data_desc, data_pov, data_pc, indexes) in enumerate(tqdm(train_loader, total=(len(train_indices) // batch_size))):
            data_desc = data_desc.to(device)
            # data_pov = data_pov.to(device)
            data_pc = data_pc.to(device)

            optimizer.zero_grad()

            if torch.isnan(data_pc).any() or torch.isinf(data_pc).any():
                print("Invalid input detected")

            output_desc = model_desc(data_desc)

            data_pc = data_pc.transpose(2, 1)
            points = data_pc.cpu().numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            data_pc = points.transpose(2, 1)
            data_pc = data_pc.to(device=device)
            output_pc, trans_feat = model_pc(data_pc)

            # initial_index = i * batch_size
            # final_index = (i + 1) * batch_size
            # if final_index > len(train_indices):
            #     final_index = len(train_indices)

            multiplication_dp = train_utility.cosine_sim(output_desc, output_pc)

            # output_description_train[initial_index:final_index, :] = output_desc.to('cpu')
            # output_pov_train[initial_index:final_index, :] = output_pc.to('cpu')

            loss_contrastive = cont_loss.calculate_loss(multiplication_dp)

            if loss_contrastive.item() > max_loss:
                max_loss = loss_contrastive.item()

            loss_contrastive.backward()

            optimizer.step()
            torch.cuda.empty_cache()

            total_loss_train += loss_contrastive.detach().item()
            num_batches_train += 1

        # ds1, ds5, ds10, sd1, sd5, sd10, ndgc_10, ndcg, ds_medr, sd_medr = train_utility.evaluate(output_description=output_description_train,
        #                                                                     output_scene=output_pov_train, section='train')
        # wandb.log({"ds10_train": ds10, "median_r_train": ds_medr})

        scheduler.step()
        print(scheduler.get_last_lr())
        print('total_loss_train', total_loss_train)
        epoch_loss_train = total_loss_train / num_batches_train

        # Validation Procedure
        model_desc.eval()
        model_pov.eval()
        model_pc.eval()
        with torch.no_grad():
            for j, (data_desc, data_pov, data_pc, indexes) in enumerate(val_loader):

                data_desc = data_desc.to(device)
                # data_pov = data_pov.to(device)
                data_pc = data_pc.to(device)

                if torch.isnan(data_pc).any():
                    print('Data_PC has NAN')
                    print(indexes)

                output_desc = model_desc(data_desc)
                # output_pov = model_pov(data_pov)
                output_pc, trans_feat = model_pc(data_pc)

                initial_index = j * batch_size
                final_index = (j + 1) * batch_size
                if final_index > len(val_indices):
                    final_index = len(val_indices)

                output_description_val[initial_index:final_index, :] = output_desc.to('cpu')
                output_pov_val[initial_index:final_index, :] = output_pc.to('cpu')

                multiplication_dp = train_utility.cosine_sim(output_desc, output_pc)

                loss_contrastive = cont_loss.calculate_loss(multiplication_dp)

                total_loss_val += loss_contrastive.detach().item()
                num_batches_val += 1

            epoch_loss_val = total_loss_val / num_batches_val

            print('Loss Train', epoch_loss_train)
            cont_loss.on_epoch_end(epoch_loss_train, train=True)
            print('Loss Val', epoch_loss_val)
            cont_loss.on_epoch_end(epoch_loss_val, train=False)

        ds1, ds5, ds10, sd1, sd5, sd10, ndgc_10, ndcg, ds_medr, sd_medr = train_utility.evaluate(output_description=output_description_val,
                                                                  output_scene=output_pov_val, section='val')

        wandb.log({"ds10_val": ds10, "median_r_val": ds_medr})

        model_desc.train()
        model_pov.train()
        model_pc.train()

        if ds10 > best_r10:
            best_r10 = ds10
            train_utility.save_best_model(approach_name, model_pc.state_dict(), model_desc.state_dict())

    bm_pc, bm_desc_pov = train_utility.load_best_model(approach_name)
    model_pc.load_state_dict(bm_pc)
    model_desc.load_state_dict(bm_desc_pov)

    model_pc.eval()
    model_desc.eval()
    output_description_test = torch.empty(len(test_indices), output_feature_size)
    output_pov_test = torch.empty(len(test_indices), output_feature_size)
    # Evaluate test set
    with torch.no_grad():
        for j, (data_desc, data_pov, data_pc, indexes) in enumerate(test_loader):

            data_desc = data_desc.to(device)
            # data_pov = data_pov.to(device)
            data_pc = data_pc.to(device)

            output_desc = model_desc(data_desc)
            # output_pov = model_pov(data_pov)
            output_pc, trans_feat = model_pc(data_pc)

            initial_index = j * batch_size
            final_index = (j + 1) * batch_size
            if final_index > len(test_indices):
                final_index = len(test_indices)
            output_description_test[initial_index:final_index, :] = output_desc
            output_pov_test[initial_index:final_index, :] = output_pc
    ds1, ds5, ds10, sd1, sd5, sd10, ndgc_10, ndcg, ds_medr, sd_medr = train_utility.evaluate(
        output_description=output_description_test,
        output_scene=output_pov_test,
        section="test")
    wandb.log({"ds10_test": ds10, "median_r_test": ds_medr})
    # train_utility.save_results(margins, [ds1, ds5, ds10, sd1, sd5, sd10, ndgc_10, ndcg, ds_medr, sd_medr])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--npc", type=int, help="num of points",
                        default=5000,
                        required=False)
    parser.add_argument("--batch_size", type=int, help="num of points",
                        default=64,
                        required=False)
    parser.add_argument("--epochs", type=int, help="num of epochs",
                        default=50,
                        required=False)
    parser.add_argument("--lr", type=float, help="Learning Rate",
                        default=0.0008,
                        required=False)
    parser.add_argument("--fps", type=bool, help="Use Farthest PointSampler",
                        default=True,
                        required=False)
    args = parser.parse_args()
    start_train(args)
