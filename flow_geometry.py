import numpy as np
import torch
import torch.nn as nn
from monodepth2.networks.layers import PixToFlow, FlowToPix

"""
Src: https://github.com/Huangying-Zhan/DF-VO/blob/master/libs/geometry/backprojection.py
"""
class Backprojection(nn.Module):
    """Layer to backproject a depth image given the camera intrinsics
    """
    
    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(Backprojection, self).__init__()

        self.height = height
        self.width = width

        # generate regular grid
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.tensor(id_coords)

        # generate homogeneous pixel coordinates
        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                 requires_grad=False)
        self.xy = torch.unsqueeze(
                        torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0)
                        , 0)
        self.xy = torch.cat([self.xy, self.ones], 1)
        self.xy = nn.Parameter(self.xy, requires_grad=False)

    def forward(self, depth, inv_K, img_like_out=False):
        """Forward pass
        Args:
            depth (tensor, [Nx1xHxW]): depth map 
            inv_K (tensor, [Nx4x4]): inverse camera intrinsics
            img_like_out (bool):if True, the output shape is Nx4xHxW; else Nx4x(HxW)
        
        Returns:
            points (tensor, [Nx4x(HxW) or Nx4xHxW]): 3D points in homogeneous coordinates
        """
        depth = depth.contiguous()

        xy = self.xy.repeat(depth.shape[0], 1, 1)
        ones = self.ones.repeat(depth.shape[0],1,1)
        
        points = torch.matmul(inv_K[:, :3, :3], xy)
        points = depth.view(depth.shape[0], 1, -1) * points
        points = torch.cat([points, ones], 1)

        if img_like_out:
            points = points.reshape(depth.shape[0], 4, self.height, self.width)
        return points
  
"""
Src: https://github.com/Huangying-Zhan/DF-VO/blob/master/libs/geometry/backprojection.py
"""
class Projection(nn.Module):
    """Layer to project 3D points into a camera view given camera intrinsics
    """
    def __init__(self, height, width, eps=1e-7):
        """
        Args:
            height (int): image height
            width (int): image width
            eps (float): small number to prevent division of zero
        """
        super(Projection, self).__init__()

        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points3d, K, normalized=True):
        """Forward pass
        Args:
            points3d (tensor, [Nx4x(HxW)]): 3D points in homogeneous coordinates
            K (tensor, [Nx4x4]): camera intrinsics
            normalized (bool): 
                
                - True: normalized to [-1, 1]
                - False: [0, W-1] and [0, H-1]
        
        Returns:
            xy (tensor, [NxHxWx2]): pixel coordinates
        """
        # projection
        points2d = torch.matmul(K[:, :3, :], points3d)

        # convert from homogeneous coordinates
        xy = points2d[:, :2, :] / (points2d[:, 2:3, :] + self.eps)
        xy = xy.view(points3d.shape[0], 2, self.height, self.width)
        xy = xy.permute(0, 2, 3, 1)

        # normalization
        if normalized:
            xy[..., 0] /= self.width - 1
            xy[..., 1] /= self.height - 1
            xy = (xy - 0.5) * 2
        return xy
    
class Transformation3D(nn.Module):
    """Layer to transform 3D points given transformation matrice
    """
    def __init__(self):
        super(Transformation3D, self).__init__()

    def forward(self, points, T):
        """Forward pass
        
        Args:
            points (tensor, [Nx4x(HxW)]): 3D points in homogeneous coordinates
            T (tensor, [Nx4x4]): transformation matrice
        
        Returns:
            transformed_points (tensor, [Nx4x(HxW)]): 3D points in homogeneous coordinates
        """
        transformed_points = torch.matmul(T, points)
        return transformed_points
    
class Reprojection(nn.Module):
    """Layer to transform pixel coordinates from one view to another view via
    backprojection, transformation in 3D, and projection
    """
    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(Reprojection, self).__init__()

        # layers
        self.backproj = Backprojection(height, width)
        self.transform = Transformation3D()
        self.project = Projection(height, width)

    def forward(self, depth, T, K, inv_K, normalized=True):
        """Forward pass
        
        Args:
            depth (tensor, [Nx1xHxW]): depth map 
            T (tensor, [Nx4x4]): transformation matrice
            inv_K (tensor, [Nx4x4]): inverse camera intrinsics
            K (tensor, [Nx4x4]): camera intrinsics
            normalized (bool): 
                
                - True: normalized to [-1, 1]
                - False: [0, W-1] and [0, H-1]
        Returns:
            xy (NxHxWx2): pixel coordinates
        """
        points3d = self.backproj(depth, inv_K)
        points3d_trans = self.transform(points3d, T)
        xy = self.project(points3d_trans, K, normalized) 
        return xy
    
class RigidFlow(nn.Module):
    """Layer to compute rigid flow given depth and camera motion
    """
    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(RigidFlow, self).__init__()
        # basic configuration
        self.height = height
        self.width = width
        self.device = torch.device('cuda')

        # layer setup
        self.pix2flow = PixToFlow(1, self.height, self.width) 
        self.pix2flow.to(self.device)

        self.reprojection = Reprojection(self.height, self.width)

    def forward(self, depth, T, K, inv_K, normalized=True):
        """Forward pass
        
        Args:
            depth (tensor, [Nx1xHxW]): depth map 
            T (tensor, [Nx4x4]): transformation matrice
            inv_K (tensor, [Nx4x4]): inverse camera intrinsics
            K (tensor, [Nx4x4]): camera intrinsics
            normalized (bool): 
                
                - True: normalized to [-1, 1]
                - False: [0, W-1] and [0, H-1]
        Returns:
            flow (NxHxWx2): rigid flow
        """
        xy = self.reprojection(depth, T, K, inv_K, normalized)
        flow = self.pix2flow(xy)

        return flow