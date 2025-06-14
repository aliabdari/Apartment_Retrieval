import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class classifier_pointnet2_msg(nn.Module):
    def __init__(self, num_class=10, normal_channel=False):
        super(classifier_pointnet2_msg, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 128],
                                             in_channel=in_channel, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        xyz = xyz.float()
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # time_ = str(int(time.time()))
        # torch.save(xyz, 'xyz.pt')
        # print('30', xyz.shape)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        # print('l1_xyz.shape', l1_xyz.shape, 'l1_points.shape', l1_points.shape)
        # torch.save(l1_xyz, 'l1_xyz.pt')
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # torch.save(l2_xyz, 'l2_xyz.pt')
        # print('l2_xyz.shape', l2_xyz.shape, 'l2_points.shape', l2_points.shape)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # torch.save(l3_xyz, 'l3_xyz.pt')
        # print('l3_xyz.shape', l3_xyz.shape, 'l3_points.shape', l3_points.shape)
        x = l3_points.view(B, 1024)
        # print('37', x.shape)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # print('39', x.shape)
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # print('41', x.shape)
        # x = self.fc3(x)
        # print('43', x.shape)
        # exit(0)
        # x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
