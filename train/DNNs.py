import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Type
import logging
# from layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
#     create_grouper, furthest_point_sample, random_sample, three_interpolation, get_aggregation_feautres


class Combiner(nn.Module):
    def __init__(self, input_features=256, output_features=256, num_inputs=4, hidden_dim=256):
        super(Combiner, self).__init__()
        self.total_input_features = input_features * num_inputs
        self.fc1 = nn.Linear(self.total_input_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_features)
        self.activation = nn.ReLU()

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=-1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x


class Tnet(nn.Module):
    ''' T-Net learns a Transformation matrix with a specified dimension '''

    def __init__(self, dim, num_points=2500):
        super(Tnet, self).__init__()

        # dimensions for transform matrix
        self.dim = dim

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim ** 2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x):
        bs = x.shape[0]

        # pass through shared MLP layers (conv1d)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))

        # max pool over num points
        x = self.max_pool(x).view(bs, -1)

        # pass through MLP
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)

        # initialize identity matrix
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.dim, self.dim) + iden

        return x


class PointNetBackbone(nn.Module):
    '''
    This is the main portion of Point Net before the classification and segmentation heads.
    The main function of this network is to obtain the local and global point features,
    which can then be passed to each of the heads to perform either classification or
    segmentation. The forward pass through the backbone includes both T-nets and their
    transformations, the shared MLPs, and the max pool layer to obtain the global features.

    The forward function either returns the global or combined (local and global features)
    along with the critical point index locations and the feature transformation matrix. The
    feature transformation matrix is used for a regularization term that will help it become
    orthogonal. (i.e. a rigid body transformation is an orthogonal transform and we would like
    to maintain orthogonality in high dimensional space). "An orthogonal transformations preserves
    the lengths of vectors and angles between them"
    '''

    def __init__(self, num_points=2500, num_global_feats=1024, local_feat=True):
        ''' Initializers:
                num_points - number of points in point cloud
                num_global_feats - number of Global Features for the main
                                   Max Pooling layer
                local_feat - if True, forward() returns the concatenation
                             of the local and global features
            '''
        super(PointNetBackbone, self).__init__()

        # if true concat local and global features
        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat

        # Spatial Transformer Networks (T-nets)
        self.tnet1 = Tnet(dim=3, num_points=num_points)
        self.tnet2 = Tnet(dim=64, num_points=num_points)

        # shared MLP 1
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # shared MLP 2
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)

        # batch norms for both shared MLPs
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

        # max pool to get the global features
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)

    def show_memory(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        free_memory = torch.cuda.get_device_properties(device).total_memory - reserved_memory

        allocated_memory_mb = allocated_memory / (1024 ** 2)
        reserved_memory_mb = reserved_memory / (1024 ** 2)
        free_memory_mb = free_memory / (1024 ** 2)

        print(f"Allocated Memory: {allocated_memory_mb:.2f} MB")
        print(f"Reserved Memory: {reserved_memory_mb:.2f} MB")
        print(f"Free Memory: {free_memory_mb:.2f} MB")

    def forward(self, x):
        x = x.float()

        bs = x.shape[0]

        A_input = self.tnet1(x)

        # perform first transformation across each point in the batch
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        A_feat = self.tnet2(x)

        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)

        local_features = x.clone()

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        if self.local_feat:
            features = torch.cat((local_features,
                                  global_features.unsqueeze(-1).repeat(1, 1, self.num_points)),
                                 dim=1)
            return features, critical_indexes, A_feat

        else:
            return global_features, critical_indexes, A_feat


class OneDimensionalCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_size):
        super(OneDimensionalCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(out_channels * ((input_size - kernel_size + 1) // 2), feature_size)
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, list_length=None):
        x = x.to(torch.float32)
        x1 = self.conv(x)
        # remove the effect of the padding
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, :, list_length[item_idx]:] = 0
        x1 = self.relu(x1)
        x1 = self.adaptive_pool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)
        return x1


class OneDimensionalCNNVClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_size, feature_size):
        super(OneDimensionalCNNVClassifier, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(out_channels * ((input_size - kernel_size + 1) // 2), feature_size)
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, list_length=None):
        x = x.to(torch.float32)
        x1 = self.conv(x)
        # remove the effect of the padding
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, :, list_length[item_idx]:] = 0
        x2 = self.relu(x1)
        x3 = self.adaptive_pool(x2)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc(x3)
        return x3, x1


class FCNetClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCNetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=.1)
        self.fc2 = nn.Linear(512, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=.1)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        output = self.softmax(x)
        return output


class OneDimensionalCNNSmall(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_size, feature_size):
        super(OneDimensionalCNNSmall, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(out_channels * ((input_size - kernel_size + 1) // 2), feature_size)
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, list_length=None):
        x = x.to(torch.float32)
        x1 = self.conv(x)
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, :, list_length[item_idx]:] = 0
        return x1


class FCNet(nn.Module):
    def __init__(self, input_size, feature_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, feature_size)

    def forward(self, out, skip=False):
        out = out.to(torch.float32)
        if not skip:
            out = out.view(out.size(0), -1)
        out = self.relu1(self.fc1(out))
        out = self.dropout1(out)
        out = self.bn1(out)
        out = self.relu2(self.fc2(out))
        out = self.dropout2(out)
        out = self.bn2(out)
        out = self.fc3(out)
        return out


class GRUNet(nn.Module):
    def __init__(self, hidden_size, num_features, is_bidirectional=False):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True,
                          bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional

    def forward(self, x):
        x = x.to(torch.float32)
        _, h_n = self.gru(x)
        if self.is_bidirectional:
            return h_n.mean(0)
        return h_n.squeeze(0)


def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


# class LocalAggregation(nn.Module):
#     """Local aggregation layer for a set
#     Set abstraction layer abstracts features from a larger set to a smaller set
#     Local aggregation layer aggregates features from the same set
#     """
#
#     def __init__(self,
#                  channels: List[int],
#                  norm_args={'norm': 'bn1d'},
#                  act_args={'act': 'relu'},
#                  group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
#                  conv_args=None,
#                  feature_type='dp_fj',
#                  reduction='max',
#                  last_act=True,
#                  **kwargs
#                  ):
#         super().__init__()
#         if kwargs:
#             logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
#         channels[0] = CHANNEL_MAP[feature_type](channels[0])
#         convs = []
#         for i in range(len(channels) - 1):  # #layers in each blocks
#             convs.append(create_convblock2d(channels[i], channels[i + 1],
#                                             norm_args=norm_args,
#                                             act_args=None if i == (
#                                                     len(channels) - 2) and not last_act else act_args,
#                                             **conv_args)
#                          )
#         self.convs = nn.Sequential(*convs)
#         self.grouper = create_grouper(group_args)
#         self.reduction = reduction.lower()
#         self.pool = get_reduction_fn(self.reduction)
#         self.feature_type = feature_type
#
#     def forward(self, pf) -> torch.Tensor:
#         # p: position, f: feature
#         p, f = pf
#         # neighborhood_features
#         dp, fj = self.grouper(p, p, f)
#         fj = get_aggregation_feautres(p, dp, f, fj, self.feature_type)
#         f = self.pool(self.convs(fj))
#         """ DEBUG neighbor numbers.
#         if f.shape[-1] != 1:
#             query_xyz, support_xyz = p, p
#             radius = self.grouper.radius
#             dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
#             points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
#             logging.info(
#                 f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
#         DEBUG end """
#         return f


# class InvResMLP(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  norm_args=None,
#                  act_args=None,
#                  aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
#                  group_args={'NAME': 'ballquery'},
#                  conv_args=None,
#                  expansion=1,
#                  use_res=True,
#                  num_posconvs=2,
#                  less_act=False,
#                  **kwargs
#                  ):
#         super().__init__()
#         self.use_res = use_res
#         mid_channels = int(in_channels * expansion)
#         self.convs = LocalAggregation([in_channels, in_channels],
#                                       norm_args=norm_args, act_args=act_args if num_posconvs > 0 else None,
#                                       group_args=group_args, conv_args=conv_args,
#                                       **aggr_args, **kwargs)
#         if num_posconvs < 1:
#             channels = []
#         elif num_posconvs == 1:
#             channels = [in_channels, in_channels]
#         else:
#             channels = [in_channels, mid_channels, in_channels]
#         pwconv = []
#         # point wise after depth wise conv (without last layer)
#         for i in range(len(channels) - 1):
#             pwconv.append(create_convblock1d(channels[i], channels[i + 1],
#                                              norm_args=norm_args,
#                                              act_args=act_args if
#                                              (i != len(channels) - 2) and not less_act else None,
#                                              **conv_args)
#                           )
#         self.pwconv = nn.Sequential(*pwconv)
#         self.act = create_act(act_args)
#
#     def forward(self, pf):
#         p, f = pf
#         identity = f
#         f = self.convs([p, f])
#         f = self.pwconv(f)
#         if f.shape[-1] == identity.shape[-1] and self.use_res:
#             f += identity
#         f = self.act(f)
#         return [p, f]


# class SetAbstraction(nn.Module):
#     """The modified set abstraction module in PointNet++ with residual connection support
#     """
#
#     def __init__(self,
#                  in_channels, out_channels,
#                  layers=1,
#                  stride=1,
#                  group_args={'NAME': 'ballquery',
#                              'radius': 0.1, 'nsample': 16},
#                  norm_args={'norm': 'bn1d'},
#                  act_args={'act': 'relu'},
#                  conv_args=None,
#                  sampler='fps',
#                  feature_type='dp_fj',
#                  use_res=False,
#                  is_head=False,
#                  **kwargs,
#                  ):
#         super().__init__()
#         self.stride = stride
#         self.is_head = is_head
#         self.all_aggr = not is_head and stride == 1
#         self.use_res = use_res and not self.all_aggr and not self.is_head
#         self.feature_type = feature_type
#
#         mid_channel = out_channels // 2 if stride > 1 else out_channels
#         channels = [in_channels] + [mid_channel] * \
#                    (layers - 1) + [out_channels]
#         channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](channels[0])
#
#         if self.use_res:
#             self.skipconv = create_convblock1d(
#                 in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
#                 -1] else nn.Identity()
#             self.act = create_act(act_args)
#
#         # actually, one can use local aggregation layer to replace the following
#         create_conv = create_convblock1d if is_head else create_convblock2d
#         convs = []
#         for i in range(len(channels) - 1):
#             convs.append(create_conv(channels[i], channels[i + 1],
#                                      norm_args=norm_args if not is_head else None,
#                                      act_args=None if i == len(channels) - 2
#                                                       and (self.use_res or is_head) else act_args,
#                                      **conv_args)
#                          )
#         self.convs = nn.Sequential(*convs)
#         if not is_head:
#             if self.all_aggr:
#                 group_args.nsample = None
#                 group_args.radius = None
#             self.grouper = create_grouper(group_args)
#             self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
#             if sampler.lower() == 'fps':
#                 self.sample_fn = furthest_point_sample
#             elif sampler.lower() == 'random':
#                 self.sample_fn = random_sample
#
#     def forward(self, pf):
#         p, f = pf
#         if self.is_head:
#             f = self.convs(f)  # (n, c)
#         else:
#             if not self.all_aggr:
#                 idx = self.sample_fn(p, p.shape[1] // self.stride).long()
#                 new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
#             else:
#                 new_p = p
#             """ DEBUG neighbor numbers.
#             query_xyz, support_xyz = new_p, p
#             radius = self.grouper.radius
#             dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
#             points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
#             logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
#             DEBUG end """
#             if self.use_res or 'df' in self.feature_type:
#                 fi = torch.gather(
#                     f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
#                 if self.use_res:
#                     identity = self.skipconv(fi)
#             else:
#                 fi = None
#             dp, fj = self.grouper(new_p, p, f)
#             fj = get_aggregation_feautres(new_p, dp, fi, fj, feature_type=self.feature_type)
#             f = self.pool(self.convs(fj))
#             if self.use_res:
#                 f = self.act(f + identity)
#             p = new_p
#         return p, f


# class PointNextEncoder(nn.Module):
#     r"""The Encoder for PointNext
#     `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
#     <https://arxiv.org/abs/2206.04670>`_.
#     .. note::
#         For an example of using :obj:`PointNextEncoder`, see
#         `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
#     Args:
#         in_channels (int, optional): input channels . Defaults to 4.
#         width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
#         blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
#         strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
#         block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
#         nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
#         radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
#         aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
#         group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
#         norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
#         act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
#         expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
#         sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
#         sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S.
#     """
#
#     def __init__(self,
#                  in_channels: int = 4,
#                  width: int = 32,
#                  blocks: List[int] = [1, 4, 7, 4, 4],
#                  strides: List[int] = [4, 4, 4, 4],
#                  block: str or Type[InvResMLP] = 'InvResMLP',
#                  nsample: int or List[int] = 32,
#                  radius: float or List[float] = 0.1,
#                  aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
#                  group_args: dict = {'NAME': 'ballquery'},
#                  sa_layers: int = 1,
#                  sa_use_res: bool = False,
#                  **kwargs
#                  ):
#         super().__init__()
#         if isinstance(block, str):
#             block = eval(block)
#         self.blocks = blocks
#         self.strides = strides
#         self.in_channels = in_channels
#         self.aggr_args = aggr_args
#         self.norm_args = kwargs.get('norm_args', {'norm': 'bn'})
#         self.act_args = kwargs.get('act_args', {'act': 'relu'})
#         self.conv_args = kwargs.get('conv_args', None)
#         self.sampler = kwargs.get('sampler', 'fps')
#         self.expansion = kwargs.get('expansion', 4)
#         self.sa_layers = sa_layers
#         self.sa_use_res = sa_use_res
#         self.use_res = kwargs.get('use_res', True)
#         radius_scaling = kwargs.get('radius_scaling', 2)
#         nsample_scaling = kwargs.get('nsample_scaling', 1)
#
#         self.radii = self._to_full_list(radius, radius_scaling)
#         self.nsample = self._to_full_list(nsample, nsample_scaling)
#         logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')
#
#         # double width after downsampling.
#         channels = []
#         for stride in strides:
#             if stride != 1:
#                 width *= 2
#             channels.append(width)
#         encoder = []
#         for i in range(len(blocks)):
#             group_args.radius = self.radii[i]
#             group_args.nsample = self.nsample[i]
#             encoder.append(self._make_enc(
#                 block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
#                 is_head=i == 0 and strides[i] == 1
#             ))
#         self.encoder = nn.Sequential(*encoder)
#         self.out_channels = channels[-1]
#         self.channel_list = channels
#
#     def _to_full_list(self, param, param_scaling=1):
#         # param can be: radius, nsample
#         param_list = []
#         if isinstance(param, List):
#             # make param a full list
#             for i, value in enumerate(param):
#                 value = [value] if not isinstance(value, List) else value
#                 if len(value) != self.blocks[i]:
#                     value += [value[-1]] * (self.blocks[i] - len(value))
#                 param_list.append(value)
#         else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
#             for i, stride in enumerate(self.strides):
#                 if stride == 1:
#                     param_list.append([param] * self.blocks[i])
#                 else:
#                     param_list.append(
#                         [param] + [param * param_scaling] * (self.blocks[i] - 1))
#                     param *= param_scaling
#         return param_list
#
#     def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
#         layers = []
#         radii = group_args.radius
#         nsample = group_args.nsample
#         group_args.radius = radii[0]
#         group_args.nsample = nsample[0]
#         layers.append(SetAbstraction(self.in_channels, channels,
#                                      self.sa_layers if not is_head else 1, stride,
#                                      group_args=group_args,
#                                      sampler=self.sampler,
#                                      norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
#                                      is_head=is_head, use_res=self.sa_use_res, **self.aggr_args
#                                      ))
#         self.in_channels = channels
#         for i in range(1, blocks):
#             group_args.radius = radii[i]
#             group_args.nsample = nsample[i]
#             layers.append(block(self.in_channels,
#                                 aggr_args=self.aggr_args,
#                                 norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
#                                 conv_args=self.conv_args, expansion=self.expansion,
#                                 use_res=self.use_res
#                                 ))
#         return nn.Sequential(*layers)
#
#     def forward_cls_feat(self, p0, f0=None):
#         if hasattr(p0, 'keys'):
#             p0, f0 = p0['pos'], p0.get('x', None)
#         if f0 is None:
#             f0 = p0.clone().transpose(1, 2).contiguous()
#         for i in range(0, len(self.encoder)):
#             p0, f0 = self.encoder[i]([p0, f0])
#         return f0.squeeze(-1)
#
#     def forward_seg_feat(self, p0, f0=None):
#         if hasattr(p0, 'keys'):
#             p0, f0 = p0['pos'], p0.get('x', None)
#         if f0 is None:
#             f0 = p0.clone().transpose(1, 2).contiguous()
#         p, f = [p0], [f0]
#         for i in range(0, len(self.encoder)):
#             _p, _f = self.encoder[i]([p[-1], f[-1]])
#             p.append(_p)
#             f.append(_f)
#         return p, f
#
#     def forward(self, p0, f0=None):
#         return self.forward_seg_feat(p0, f0)
