# Apartment_Retrieval

This Repository is for Learning Joint embedding Space for the 3D Point Clouds Apartments Retrieval Task. We have a dataset containing 3d point clouds of apartments with their corresponding detailed textual descriptions. 
We try to train a retrieval system, able to receive a textual query and find the most similar 3d structures based on that.

## Dataset

The dataset can be obtained through this [repository](https://github.com/aliabdari/Point_Clouds_Generation). It will become available to the public in a few weeks.

## Train and Evaluation

In this work, we utilize [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) architecture to represent the Point Clouds information, while we use a bidirectional GRU network to represent the textual part.

To run the training and evaluation procedure, you can use [train_pointnet_v2.py](https://github.com/aliabdari/Apartment_Retrieval/blob/main/train/train_pointnet_v2.py) module with a command like:
```
python3.10 train_pointnet_v2.py
```

### Arguments
There are some arguments which can be set at the time of training:

- --npc: It allows us to set how many points from each point cloud sample we want to process. The default number is 5000.
- --batch_size: To set the batch size, the default of which is 64.
- --epochs: To set the number of epochs, the default of which is 50.
- --lr: To set the learning rate, the default of which is 0.0008.
- --fps: To set the method to reduce the number of points used in the training procedure. It can be True or False, and the default value is True. When it is True it uses the Farthest Point Sampling (FPS) method, while witht he Flase switch it uses uniform sampling.  

### Environment
To run the Experiments, we have used Python 3.10, with torch 2.5.0+cu124, with an NVIDIA A100 GPU.
