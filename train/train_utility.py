import torch
import os
import pickle
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


class ErrorIndices(Exception):
    pass


def retrieve_indices():
    if os.path.exists('indices/indices.pkl'):
        indices_pickle = open('indices/indices.pkl', "rb")
        indices_pickle = pickle.load(indices_pickle)
        train_indices = indices_pickle["train"]
        val_indices = indices_pickle["val"]
        test_indices = indices_pickle["test"]
    else:
        print('ERROR: Indices do not exist')
        exit()
        # raise ErorrIndices('Indices Not Existed')
        # with open('../houses_data/houses_data.pkl', 'rb') as f:
        #     pickle_file = pickle.load(f)
        # data_size = len(pickle_file)
        # train_ratio = .7
        # val_ratio = .15
        # perm = torch.randperm(data_size)
        # train_indices = perm[:int(data_size * train_ratio)]
        # val_indices = perm[int(data_size * train_ratio):int(data_size * (val_ratio + train_ratio))]
        # test_indices = perm[int(data_size * (val_ratio + train_ratio)):]
        # indices_pickle = {"train": train_indices, "val": val_indices, "test": test_indices}
        # with open('indices/indices.pkl', 'wb') as f:
        #     pickle.dump(indices_pickle, f)
    return train_indices, val_indices, test_indices


def cosine_sim(im, s):
    '''cosine similarity between all the image and sentence pairs
    '''
    inner_prod = im.mm(s.t())
    im_norm = torch.sqrt((im ** 2).sum(1).view(-1, 1) + 1e-18)
    s_norm = torch.sqrt((s ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)
    return sim


def create_rank(result, entire_descriptor, desired_output_index):
    similarity = torch.nn.functional.cosine_similarity(entire_descriptor, result, dim=1)
    similarity = similarity.squeeze()
    sorted_indices = torch.argsort(similarity, descending=True)
    position = torch.where(sorted_indices == desired_output_index)
    return position[0].item(), sorted_indices


def evaluate(output_description, output_scene, section):
    avg_rank_scene = 0
    ranks_scene = []
    avg_rank_description = 0
    ranks_description = []

    ndcg_10_list = []
    ndcg_entire_list = []

    for j, i in enumerate(output_scene):
        rank, sorted_list = create_rank(i, output_description, j)
        avg_rank_scene += rank
        ranks_scene.append(rank)

    for j, i in enumerate(output_description):
        rank, sorted_list = create_rank(i, output_scene, j)
        avg_rank_description += rank
        ranks_description.append(rank)

    ranks_scene = np.array(ranks_scene)
    ranks_description = np.array(ranks_description)

    n_q = len(output_scene)
    sd_r1 = 100 * len(np.where(ranks_scene < 1)[0]) / n_q
    sd_r5 = 100 * len(np.where(ranks_scene < 5)[0]) / n_q
    sd_r10 = 100 * len(np.where(ranks_scene < 10)[0]) / n_q
    sd_medr = np.median(ranks_scene) + 1
    sd_meanr = ranks_scene.mean() + 1

    n_q = len(output_description)
    ds_r1 = 100 * len(np.where(ranks_description < 1)[0]) / n_q
    ds_r5 = 100 * len(np.where(ranks_description < 5)[0]) / n_q
    ds_r10 = 100 * len(np.where(ranks_description < 10)[0]) / n_q
    ds_medr = np.median(ranks_description) + 1
    ds_meanr = ranks_description.mean() + 1

    ds_out, sc_out = "", ""
    for mn, mv in [["R@1", ds_r1],
                   ["R@5", ds_r5],
                   ["R@10", ds_r10],
                   ["median rank", ds_medr],
                   ["mean rank", ds_meanr],
                   ]:
        ds_out += f"{mn}: {mv:.4f}   "

    for mn, mv in [("R@1", sd_r1),
                   ("R@5", sd_r5),
                   ("R@10", sd_r10),
                   ("median rank", sd_medr),
                   ("mean rank", sd_meanr),
                   ]:
        sc_out += f"{mn}: {mv:.4f}   "

    print(section + " data: ")
    print("Scenes ranking: " + ds_out)
    print("Descriptions ranking: " + sc_out)
    if section == "test" and len(ndcg_10_list) > 0:
        avg_ndcg_10_entire = 100 * sum(ndcg_10_list) / len(ndcg_10_list)
        avg_ndcg_entire = 100 * sum(ndcg_entire_list) / len(ndcg_entire_list)
    else:
        avg_ndcg_10_entire = -1
        avg_ndcg_entire = -1

    return ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, avg_ndcg_10_entire, avg_ndcg_entire, ds_medr, sd_medr


def get_entire_data():
    descriptions_path = '../../features/open_clip_features/descriptions_ViT-B-32_laion2b_s34b_b79k/sentences'
    scene_path = '../../features/open_clip_features/images_ViT-B-32_laion2b_s34b_b79k'
    pcs_path = '../../3dscene_dataset/colored_point_clouds'
    return descriptions_path, scene_path, pcs_path


def save_best_model(model_name, *args):
    model_path = "models"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path = model_path + os.sep + model_name + '.pt'
    new_dict = dict()
    for i, bm in enumerate(args):
        new_dict[f'best_model_{str(i)}'] = bm
    torch.save(new_dict, model_path)


def load_best_model(model_name):
    model_path = "models"
    model_path = model_path + os.sep + model_name + '.pt'
    check_point = torch.load(model_path)
    bm_list = [check_point[bm] for bm in check_point.keys()]
    return bm_list


def get_transform_fp():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def get_margin_tensor(indexes, relevance_info, margins, status, thresholds=None, sent_similarity=False):
    if thresholds:
        if status == 0:
            t = thresholds
            low_, high_ = margins['margin_low'], margins['margin_high']
        elif status == 1:
            t_l, t_u = thresholds[0], thresholds[1]
            low_, mid_, high_ = margins['margin_low'], margins['margin_mid'], margins['margin_high']
        elif status == 2:
            t_l, t_m, t_u = thresholds[0], thresholds[1], thresholds[2]
            low_, mids_, midl_, high_ = margins['margin_low'], margins['margin_mids'], margins['margin_midl'], margins['margin_high']
    else:
        t_l, t_u = 0.25, 0.75
        low_, mid_, high_ = margins['margin_low'], margins['margin_mid'], margins['margin_high']
    # print('THRESHOLDS:', str(t_l), '  ', str(t_u))
    margin_tensors = torch.zeros((len(indexes), len(indexes)))

    margin_tensors.diagonal().fill_(0.25)

    for i in range(len(indexes)):
        for j in range(i+1, len(indexes)):
            if sent_similarity:
                rel = relevance_info[(indexes[i], indexes[j])] if indexes[i] < indexes[j] else relevance_info[(indexes[j], indexes[i])]
            else:
                rel = relevance_info[(indexes[i], indexes[j])]['rooms'] if indexes[i] < indexes[j] else relevance_info[(indexes[j], indexes[i])]['rooms']
            if status == 0:
                if rel < t:
                    margin_tensors[i, j] = margin_tensors[j, i] = high_
                else:
                    margin_tensors[i, j] = margin_tensors[j, i] = low_
            elif status == 1:
                if rel < t_l:
                    margin_tensors[i, j] = margin_tensors[j, i] = high_
                elif t_l <= rel < t_u:
                    margin_tensors[i, j] = margin_tensors[j, i] = mid_
                else:
                    margin_tensors[i, j] = margin_tensors[j, i] = low_
            elif status == 2:
                if rel < t_l:
                    margin_tensors[i, j] = margin_tensors[j, i] = high_
                elif t_l <= rel < t_m:
                    margin_tensors[i, j] = margin_tensors[j, i] = midl_
                elif t_m <= rel < t_u:
                    margin_tensors[i, j] = margin_tensors[j, i] = mids_
                else:
                    margin_tensors[i, j] = margin_tensors[j, i] = low_

    return margin_tensors


class LossContrastive:
    def __init__(self, name, patience=15, delta=.001, verbose=True):
        self.train_losses = []
        self.validation_losses = []
        self.name = name
        self.counter_patience = 0
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.verbose = verbose

    def on_epoch_end(self, loss, train=True):
        if train:
            self.train_losses.append(loss)
        else:
            self.validation_losses.append(loss)

    def get_loss_trend(self):
        return self.train_losses, self.validation_losses

    def calculate_loss(self, pairwise_distances, margin=.25, margin_tensor=None):
        batch_size = pairwise_distances.shape[0]
        diag = pairwise_distances.diag().view(batch_size, 1)
        pos_masks = torch.eye(batch_size).bool().to(pairwise_distances.device)
        d1 = diag.expand_as(pairwise_distances)
        if margin_tensor is not None:
            margin_tensor = margin_tensor.to(pairwise_distances.device)
            cost_s = (margin_tensor + pairwise_distances - d1).clamp(min=0)
        else:
            cost_s = (margin + pairwise_distances - d1).clamp(min=0)
        cost_s = cost_s.masked_fill(pos_masks, 0)
        # print('5', cost_s.shape)
        cost_s = cost_s / (batch_size * (batch_size - 1))
        cost_s = cost_s.sum()

        d2 = diag.t().expand_as(pairwise_distances)
        if margin_tensor is not None:
            margin_tensor = margin_tensor.to(pairwise_distances.device)
            cost_d = (margin_tensor + pairwise_distances - d2).clamp(min=0)
        else:
            cost_d = (margin + pairwise_distances - d2).clamp(min=0)
        cost_d = cost_d.masked_fill(pos_masks, 0)
        cost_d = cost_d / (batch_size * (batch_size - 1))
        cost_d = cost_d.sum()

        return (cost_s + cost_d) / 2

    def is_val_improving(self):
        score = -self.validation_losses[-1] if self.validation_losses else None

        if score and self.best_score and self.verbose:
            print('epoch:', len(self.validation_losses), ' score:', -score, ' best_score:', -self.best_score, ' counter:', self.counter_patience)

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter_patience += 1
            if self.counter_patience >= self.patience:
                return False
        else:
            self.best_score = score
            self.counter_patience = 0
        return True

    def save_plots(self):
        save_path = f'models/{self.name}.png'
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.validation_losses, label='Val Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Trend')

        plt.legend()

        plt.savefig(save_path)


def save_results(margins, results, thres=None, file_name=None):
    titles_results = ['ds1: ', ' ds5: ', ' ds10: ', ' sd1: ', ' sd5: ', ' sd10: ', ' ndgc_10: ', ' ndcg: ', ' ds_medr: ', ' sd_medr: ']
    titles_margins = list(margins.keys())
    if thres:
        if isinstance(thres, tuple):
            if len(thres) == 2:
                thres_list = ['lower_thres ', ' upper_thres ']
            elif len(thres) == 3:
                thres_list = ['lower_thres ', ' mid_thres ', ' upper_thres ']

    file_path = 'results' + os.sep + (file_name if file_name else 'results.txt')
    with open(file_path, 'a') as file:
        if thres:
            if isinstance(thres, tuple):
                for i in range(len(thres)):
                    file.write(thres_list[i] + str(thres[i]) + ' ')
            elif isinstance(thres, float):
                file.write('Threshold ' + str(thres) + ' ')
            file.write('\n')
        for i in range(len(margins)):
            file.write(titles_margins[i] + ': ' + str(margins[titles_margins[i]]) + ' ')
        file.write('\n')
        for i in range(len(results)):
            file.write(titles_results[i] + str(results[i]))
        file.write('\n')
        file.write('*'*100)
        file.write('\n')

