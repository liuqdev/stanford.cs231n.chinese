# -*- coding:utf-8 -*-
import matplotlib # 绘图库
import numpy as np # 科学计算库
from scipy.ndimage import uniform_filter # 对于图像的操作


def extract_features(imgs, feature_fns, verbose=False):
  """
  给定多幅数字图像的像素信息，以及多个可以作用于单幅图像上的特征函数，
  将这些特征函数应用于这些图像，连接每幅图像的特征性向量，并将所有图像
  的这些特征向量存储啊在一个特征矩阵中。

  输入参数:
  - imgs: N幅图像的像素矩阵，大小为N x H X W X C
  - feature_fns: 多个特征函数组成的列表。第 i 个特征函数的输入为 
  H x W x D 的数组，并返回一个（一维的）长度为F_i的数组。
  - verbose: 布尔值，若为真则输出中间过程。

  返回:
  一个大小为 (N, F_1 + ... + F_k) 的数组，其中每列都是对于单幅图像
  而言特征的级联。
  """
  num_images = imgs.shape[0]
  if num_images == 0:
    return np.array([])

  # 使用第一幅图像来确定特征的维度数是多少
  feature_dims = []
  first_image_features = []
  for feature_fn in feature_fns:
    feats = feature_fn(imgs[0].squeeze())
    assert len(feats.shape) == 1, 'Feature functions must be one-dimensional 特征函数必须是一维的'
    feature_dims.append(feats.size)
    first_image_features.append(feats)

  # 现在我们知道了这些特征的各维度，我们可以分配一个大数据将所有的特征存储为列
  total_feature_dim = sum(feature_dims)
  imgs_features = np.zeros((num_images, total_feature_dim))
  imgs_features[0] = np.hstack(first_image_features).T

  # 提取剩余图像的特征
  for i in xrange(1, num_images):
    idx = 0
    for feature_fn, feature_dim in zip(feature_fns, feature_dims):
      next_idx = idx + feature_dim
      imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
      idx = next_idx
    if verbose and i % 1000 == 0:
      print 'Done extracting features for %d / %d images' % (i, num_images)

  return imgs_features


def rgb2gray(rgb):
  """将 RGB 图像转化为灰度图像

    参数:
      rgb : RGB 图像

    返回:
      gray : 灰度图像
  
  """
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def hog_feature(im):
  """计算一幅图像的 梯度直方图(Histogram of Gradient (HOG))
  
       Modified from skimage.feature.hog
       http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog
     
     参考:
       Histograms of Oriented Gradients for Human Detection
       Navneet Dalal and Bill Triggs, CVPR 2005
     
    参数:
      im : 一幅灰度或者RGB图像
      
    返回:
      feat: Histogram of Gradient (HOG) 特征
    
  """
  
  # 遇到 RGB 彩图转换为灰度图像
  if im.ndim == 3:
    image = rgb2gray(im)
  else:
    image = np.at_least_2d(im)

  sx, sy = image.shape # 图像大小
  orientations = 9 # 梯度的分组数
  cx, cy = (8, 8) # 每个胞的像素数

  gx = np.zeros(image.shape)
  gy = np.zeros(image.shape)
  gx[:, :-1] = np.diff(image, n=1, axis=1) # 计算 x 方向上的梯度
  gy[:-1, :] = np.diff(image, n=1, axis=0) # 计算 y 方向的梯度
  grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
  grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # 梯度方向

  n_cellsx = int(np.floor(sx / cx))  # x 中胞的个数
  n_cellsy = int(np.floor(sy / cy))  # y 中胞的个数
  # 计算方向上积分图像
  orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
  for i in range(orientations):
    # 在该方向上创建新的积分图像 create new integral image for this orientation
    # 将方向隔离在这些范围内 isolate orientations in this range
    temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                        grad_ori, 0)
    temp_ori = np.where(grad_ori >= 180 / orientations * i,
                        temp_ori, 0)
    # 为这些方向选择大小 select magnitudes for those orientations
    cond2 = temp_ori > 0
    temp_mag = np.where(cond2, grad_mag, 0)
    orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx/2::cx, cy/2::cy].T
  
  return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
  """
  使用色调计算图像的颜色直方图。

  输入:
  - im: H x W x C array of pixel data for an RGB image.
  - nbin: Number of histogram bins. (default: 10)
  - xmin: Minimum pixel value (default: 0)
  - xmax: Maximum pixel value (default: 255)
  - normalized: Whether to normalize the histogram (default: True)

  返回:
    1D vector of length nbin giving the color histogram over the hue of the
    input image.
  """
  ndim = im.ndim
  bins = np.linspace(xmin, xmax, nbin+1)
  hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
  imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
  imhist = imhist * np.diff(bin_edges)

  # return histogram
  return imhist


pass
