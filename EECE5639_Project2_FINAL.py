from numpy import zeros_like, array, float32, ndarray, ones, zeros, sqrt as np_sqrt, arctan2, pi, argwhere, column_stack, copy as np_copy, newaxis, vstack, hstack, linalg, random, where, column_stack
from PIL import Image, ImageChops
from PIL.Image import fromarray, open as pil_open
from os.path import isdir, exists, join
import os
from typing import Callable
from math import ceil
from random import randint
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


'''
    Image Sequence Folder Directories with Image Sequence start photo and end photo
'''

# Default Image Settings
default_img_dirs = {1: {
    'img_base_dir': r'.\DanaHallWay1',
    'filename_prefix': 'DSC_',
    'filename_ext': 'JPG',
    'filename_zfill': 4,
    'image_id_start': 281,
    'image_id_end': 282,
    'block_size': (3, 3),
    'k': 0.04,
    'patch_size': 5,
    'thresh_perc': 0.04,
    'match_corner_thresh': 0.7,
    'match_corner_chunk_size': 250,
    'black_white_patches': False,
    'ransac_threshold': 0.1,
    'ransac_max_iter': 1000,
    'ransac_sample_size': 4},
    2: {
    'img_base_dir': r'.\DanaOffice',
    'filename_prefix': 'DSC_',
    'filename_ext': 'JPG',
    'filename_zfill': 4,
    'image_id_start': 308,
    'image_id_end': 317,
    'block_size': (3, 3),
    'k': 0.04,
    'patch_size': 5,
    'thresh_perc': 0.01,
    'match_corner_thresh': 0.7,
    'match_corner_chunk_size': 250,
    'black_white_patches': False,
    'ransac_threshold': .1,
    'ransac_max_iter': 1000,
    'ransac_sample_size': 4}}

# Testing Image Settings
test_img_dirs = {1: { # 946 best inliner
    'img_base_dir': r'.\DanaHallWay1',
    'filename_prefix': 'DSC_',
    'filename_ext': 'JPG',
    'filename_zfill': 4,
    'image_id_start': 281,
    'image_id_end': 282,
    'block_size': (3, 3),
    'k': 0.04,
    'patch_size': 5,
    'thresh_perc': 0.04,
    'match_corner_thresh': .97,
    'match_corner_chunk_size': 250,
    'black_white_patches': False,
    'ransac_threshold': 9,
    'ransac_max_iter': 12000,
    'ransac_sample_size': 4},
    2: {
    'img_base_dir': r'.\DanaOffice',
    'filename_prefix': 'DSC_',
    'filename_ext': 'JPG',
    'filename_zfill': 4,
    'image_id_start': 308,
    'image_id_end': 309,
    'block_size': (3, 3),
    'k': 0.04,
    'patch_size': 5,
    'thresh_perc': 0.01,
    'match_corner_thresh': 0.7,
    'match_corner_chunk_size': 250,
    'black_white_patches': True,
    'ransac_threshold': .1,
    'ransac_max_iter': 1000,
    'ransac_sample_size': 4}}


# Global Settings
image_setting = test_img_dirs[1]


class ImageMosaic(object):
    """
    Class with all the necessary functions to create an Image Mosaic.
    We decided to use the Corner Harris method, Normalized Cross-Correlation,
    RANSAC, least squares homography, and planar warping with blending
    """

    def __init__(self,
                 img_base_dir: str,
                 filename_prefix: str,
                 filename_ext: str,
                 filename_zfill: int,
                 image_id_start: int,
                 image_id_end: int):
        """
        Initialize Image Mosaic class that controls various functions to create an
        image mosaic

        @param img_base_dir: Base directory for sequence of images dataset
        @type img_base_dir: String
        @param filename_prefix: Prefix name for filename of sequence of images
        @type filename_prefix: String
        @param filename_ext: Extension for filename
        @type filename_ext: String
        @param filename_zfill: Number padding for image ID (i.e. zfill 4 is 0000)
        @type filename_zfill: Integer
        @param image_id_start: Starting ID for image number pad
        @type image_id_start: Integer
        @param image_id_end: Ending ID for image number pad
        @type image_id_end: Integer
        """
        print(img_base_dir)
        # Ensure that img_base_dir is an existing path and valid directory
        assert isdir(img_base_dir) and exists(img_base_dir)

        # Initialize parameters
        self.sobel_x = array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_y = array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.dialate_kernel = ones((3, 3))
        self.img_dir = img_base_dir
        self.filename_prefix = filename_prefix
        self.filename_ext = filename_ext
        self.filename_zfill = filename_zfill
        self.image_id_start = image_id_start
        self.image_id_counter = image_id_start
        self.image_id_end = image_id_end
        self.convolve_sum_func = lambda x, y: (x * y).sum()

    def create_mosaic(self,
                      block_size: tuple = (3, 3),
                      k: float = 0.04,
                      patch_size: int = 5,
                      thresh_perc: float = 0.01,
                      match_corner_thresh: float = 0.7,
                      match_corner_chunk_size: int = 1000,
                      black_white_patches: bool = False,
                      ransac_threshold: float = .1,
                      ransac_max_iter: int = 1000,
                      ransac_sample_size: int = 4):
        """

        @param block_size:
        @type block_size: Tuple
        @param k:
        @type k: Float
        @param patch_size:
        @type patch_size: Integer
        @param thresh_perc:
        @type thresh_perc: Float
        @param match_corner_thresh:
        @type match_corner_thresh:
        @param match_corner_chunk_size:
        @type match_corner_chunk_size:
        @param black_white_patches:
        @type black_white_patches:
        """

        # find corners for first image in sequence
        print('Finding corners for Starting Image: %s' % '{}{}.{}'.format(
                self.filename_prefix,
                str(self.image_id_counter).zfill(
                    self.filename_zfill),
                self.filename_ext))
        prev_image = self.get_next_image()
        prev_corners = self.find_corners(prev_image, block_size, k, thresh_perc)
        prev_corners = column_stack((prev_corners[:, 1], prev_corners[:, 0]))

        # Get patches for first image in sequence
        print('Getting patches for Starting Image: %s' % '{}{}.{}'.format(
                self.filename_prefix,
                str(self.image_id_counter - 1).zfill(
                    self.filename_zfill),
                self.filename_ext))
        prev_patches = self.get_patches(prev_image,
                                        prev_corners,
                                        patch_size,
                                        black_white=black_white_patches)

        # Iterate among image sequence to process mosaic
        for img_id in range(self.image_id_start + 1,
                            self.image_id_end + 1):
            # Find corners for next image in sequence of images
            print('Finding corners for Image: %s' % '{}{}.{}'.format(
                self.filename_prefix,
                str(self.image_id_counter).zfill(
                    self.filename_zfill),
                self.filename_ext))
            curr_image = self.get_next_image()
            curr_corners = self.find_corners(curr_image, block_size, k, thresh_perc)
            curr_corners = column_stack((curr_corners[:, 1], curr_corners[:, 0]))

            # Get patches for current image
            print('Getting patches for Image: %s' % '{}{}.{}'.format(
                self.filename_prefix,
                str(self.image_id_counter - 1).zfill(
                    self.filename_zfill),
                self.filename_ext))
            curr_patches = self.get_patches(curr_image,
                                            curr_corners,
                                            patch_size,
                                            black_white=black_white_patches)

            # Find the normalized cross-correlation between previous image and current image
            print('Finding Best Corners via NCC for Images (%s, %s)' % (
                '{}{}.{}'.format(
                    self.filename_prefix,
                    str(self.image_id_counter - 1).zfill(
                        self.filename_zfill),
                    self.filename_ext),
                '{}{}.{}'.format(
                    self.filename_prefix,
                    str(self.image_id_counter - 2).zfill(
                        self.filename_zfill),
                    self.filename_ext)
            ))
            prev_corners, curr_corners = self.match_corners(prev_patches,
                                                            curr_patches,
                                                            prev_corners,
                                                            curr_corners,
                                                            threshold=match_corner_thresh,
                                                            chunk_size=match_corner_chunk_size,
                                                            black_white=black_white_patches)
            print(prev_corners.shape, curr_corners.shape)

            self.draw_corr_corners(prev_image,
                                   curr_image,
                                   prev_corners,
                                   curr_corners,
                                   filename_prefix='color_corr')

            indices, h, num_inliers = self.ransac(prev_corners,
                                                  curr_corners,
                                                  threshold=ransac_threshold,
                                                  max_iter=ransac_max_iter,
                                                  sample_size=ransac_sample_size
            )
            self.draw_corr_corners(prev_image,
                                   curr_image,
                                   prev_corners[indices],
                                   curr_corners[indices],
                                   filename_prefix='color_ransac')
            print(h)
            
            print('Generating warped image.')
            self.warp_image(prev_image,
                            curr_image,
                            h,
                            filename_prefix='warped_image')
            
            print('Starting Extra Credit')
            #With PIL
            #extracredit1 = pil_open('extra credit image 1.png')
            #extracredit2 = pil_open('extra credit image 2.png')
            
            #With matplotlib
            extracredit1 = 'extra credit image 1.png' 
            extracredit2 = 'extra credit image 2.png'

            self.extra_credit(
                image1 = extracredit1,
                image2 = extracredit2)

            
            

    def get_next_image(self) -> Image:
        """
        Get the next image in the image sequence

        @return: Next image in the Image sequence if it exists
        @rtype: Image
        """

        # assemble file path for next image
        fp = join(self.img_dir, '{}{}.{}'.format(
            self.filename_prefix,
            str(self.image_id_counter).zfill(
                self.filename_zfill),
            self.filename_ext))

        # ensure that image exists
        assert exists(fp)

        # open image as a PIL image and return image
        img = pil_open(fp)
        self.image_id_counter += 1
        return img

    def find_corners(self, image: Image,
                     block_size: tuple = (3, 3),
                     k: float = 0.04,
                     thresh_perc: float = 0.01) -> ndarray:
        """
        Find corners in an image by calculating the r equation from
        harris corner detection.

        @param image: Image that you wish to find corners
        @type image: PIL Image
        @param block_size: block size for neighborhood in calculating Sx2, Sy2, Sxy
        @type block_size: tuple
        @param k: K value for harris corner r equation
        @type k: float
        @param thresh_perc: Percent threshold for r threshold after non-max suppression
        @type thresh_perc: float
        @return: Array of corners from Harris Corner detector
        @rtype: Numpy Array
        """

        # Copy Image
        bw_img = image.copy()

        # Convert to black and white image
        if bw_img.mode != 'L':
            bw_img = bw_img.convert("L")

        # make as numpy array and find the derivatives by applying sobel filters
        bw_img = array(bw_img)
        dx = self.convolve(bw_img, self.sobel_x)
        dy = self.convolve(bw_img, self.sobel_y)

        # Calculate dx2, dy2, and dxy
        dx2, dy2, dxy = dx ** 2, dy ** 2, dx * dy

        # Find Sx2, Sy2, and Sxy
        neighborhood = ones(block_size)
        neigh_dx2 = self.convolve(dx2, neighborhood)
        neigh_dy2 = self.convolve(dy2, neighborhood)
        neigh_dxy = self.convolve(dxy, neighborhood)

        # Calculate R equation
        r = (neigh_dx2 * neigh_dy2) - (neigh_dxy ** 2) - k * ((neigh_dx2 + neigh_dy2) ** 2)

        # Find magnitude and direction by using dx2, dy2, dx, and dy values
        magnitude = np_sqrt(dx2 + dy2)
        direction = arctan2(dy, dx) * 180 / pi

        # Find the non-max suppression for r
        r = self.non_max_suppression(r, magnitude, direction)

        # copy r, save image as non-bright r equation after non-max suppression
        r_bright = np_copy(r)
        fromarray(r).convert("L").save('unbright_corner_{}{}.png'.format(
            self.filename_prefix,
            str(self.image_id_counter-1).zfill(
                self.filename_zfill - 1)))

        # Make a bright version of the r equation after non-max suppression and save img
        r_bright[r > thresh_perc * r.max()] = 255
        r_bright[r <= thresh_perc * r.max()] = 0
        fromarray(r_bright).convert("L").save('bright_corner_{}{}.png'.format(
            self.filename_prefix,
            str(self.image_id_counter-1).zfill(
                self.filename_zfill - 1)))

        # return corners by thresholding r
        return argwhere(r > thresh_perc * r.max())

    @staticmethod
    def non_max_suppression(image: ndarray,
                            magnitude: ndarray,
                            direction: ndarray) -> ndarray:
        """
        Thins corners to get rid of non-max items by suppressing it in the image

        @param image: Array representing a gray picture
        @type image: Numpy Array
        @param magnitude: Magnitude calculated by using sqrt(dx2 + dy2)
        @type magnitude: Numpy Array
        @param direction: Directions calculated by arctan2(dy, dx) * 180 / pi
        @type direction: Numpy Array
        @return: Picture that is non-max suppressed (thinning of corners)
        @rtype: Numpy Array
        """

        # generate list of angles to suppress non-max items
        angles = array(list(range(1, 14, 2))) * (360.0 / 16)

        # iterate image with i, j indices
        for i in range(1, magnitude.shape[0] - 1):
            for j in range(1, magnitude.shape[1] - 1):
                # apply magnitude value to neigh1, neigh2 according to the direction and
                # where that direction points to between the below angles
                if (0 <= direction[i, j] < angles[0]) or (157.5 <= direction[i, j] <= 180):
                    neigh1 = magnitude[i, j + 1]
                    neigh2 = magnitude[i, j - 1]
                elif angles[0] <= direction[i, j] < angles[1]:
                    neigh1 = magnitude[i + 1, j + 1]
                    neigh2 = magnitude[i - 1, j - 1]
                elif angles[1] <= direction[i, j] < angles[2]:
                    neigh1 = magnitude[i + 1, j]
                    neigh2 = magnitude[i - 1, j]
                elif angles[2] <= direction[i, j] < angles[3]:
                    neigh1 = magnitude[i + 1, j - 1]
                    neigh2 = magnitude[i - 1, j + 1]
                elif angles[3] <= direction[i, j] < angles[4]:
                    neigh1 = magnitude[i, j - 1]
                    neigh2 = magnitude[i, j + 1]
                elif angles[4] <= direction[i, j] < angles[5]:
                    neigh1 = magnitude[i - 1, j - 1]
                    neigh2 = magnitude[i + 1, j + 1]
                elif angles[5] <= direction[i, j] < angles[6]:
                    neigh1 = magnitude[i - 1, j]
                    neigh2 = magnitude[i + 1, j]
                elif angles[6] <= direction[i, j] < angles[7]:
                    neigh1 = magnitude[i - 1, j + 1]
                    neigh2 = magnitude[i + 1, j - 1]
                else:
                    neigh1 = 255
                    neigh2 = 255

                # if the two neighbors values are smaller than the image magnitude then...
                if magnitude[i, j] >= neigh1 and magnitude[i, j] >= neigh2:
                    # apply magnitude to image at that location
                    image[i, j] = magnitude[i, j]
                else:
                    # apply 0 at that image location
                    image[i, j] = 0

        # return image
        return image

    def get_patches(self,
                    image: Image,
                    corners: ndarray,
                    patch_size: int = 5,
                    black_white: bool = False) -> ndarray:
        """
        Find patches centered at the corners given a defined patch size

        @param image: Image to get patches from
        @type image: PIL Image
        @param corners: Array of corners to find patches of images
        @type corners: Numpy Array
        @param patch_size: Size of patch for each corner
        @type patch_size: Integer
        @param black_white: (default False) Convert Image to black and white
        @type black_white: Bool
        """

        # Copy Image
        bw_img = image.copy()

        # Calculate padding size
        pad_h, pad_w = patch_size // 2, patch_size // 2

        # Convert to black and white image
        if bw_img.mode != 'L' and black_white:
            bw_img = bw_img.convert("L")

        rgb_img = self.pad(array(bw_img), (pad_h, pad_w))
        patches = list()

        for corner in corners:
            patches.append(rgb_img[pad_h + corner[1] - patch_size // 2:pad_h + corner[1] + patch_size // 2 + 1,
                           pad_w + corner[0] - patch_size // 2:pad_w + corner[0] + patch_size // 2 + 1])

        return array(patches, dtype=object)

    @staticmethod
    def pad(image: ndarray,
            padding: tuple) -> ndarray:
        """
        Create a padding for the image

        @param image: Image to apply padding
        @type image: Numpy Array
        @param padding: Padding dimensions (height, width)
        @type padding: Numpy Array
        @return: Image with padding added to image
        @rtype: Numpy Array
        """

        if len(image.shape) > 2:
            pad_h, pad_w = padding
            img_h, img_w, img_z = image.shape
            z = zeros((img_h + 2 * pad_h, img_w + 2 * pad_w, img_z))
            z[pad_h:img_h + pad_h, pad_w:img_w + pad_w] = image
        else:
            pad_h, pad_w = padding
            img_h, img_w = image.shape
            z = zeros((img_h + 2 * pad_h, img_w + 2 * pad_w))
            z[pad_h:img_h + pad_h, pad_w:img_w + pad_w] = image

        for i in range(pad_h):
            z[i, pad_w:img_w + pad_w] = image[pad_h - i - 1, :]
            z[img_h + pad_h + i, pad_w:img_w + pad_w] = image[img_h - i - 1, :]

        for j in range(pad_w):
            z[:, j] = z[:, 2 * pad_w - j - 1]
            z[:, img_w + pad_w + j] = z[:, img_w + pad_w - j - 1]

        return z

    def convolve(self,
                 image: ndarray,
                 kernel: ndarray,
                 padding: (tuple, None) = None,
                 conv_func: Callable = None) -> ndarray:
        """
        Convolve an image with a kernel patch

        @param image: Image to use convolve function
        @type image:  Numpy Array
        @param kernel: Patch/Filter to use convolve with
        @type kernel: Numpy Array
        @param padding: Padding size for padded image
        @type padding: Integer or None
        @param conv_func: Callable function to use when applying kernel
        @return: Image to convolve
        @rtype: Numpy Array
        """

        if conv_func is None:
            conv_func = self.convolve_sum_func

        if padding is None:
            pad_h, pad_w = (len(kernel) // 2, len(kernel) // 2)
        else:
            pad_h, pad_w = padding

        output = zeros_like(image)
        pad_image = self.pad(image, (pad_h, pad_w))

        for i in range(pad_h, len(image) + pad_h):
            for j in range(pad_w, len(image[0]) + pad_w):
                patch = pad_image[i - pad_h:i + pad_h + 1,
                        j - pad_w:j + pad_w + 1]
                output[i - pad_h, j - pad_w] = conv_func(patch,
                                                         kernel)

        return output

    @staticmethod
    def match_corners(patches: ndarray,
                      patches2: ndarray,
                      prev_corners: ndarray,
                      curr_corners: ndarray,
                      threshold: float = 0.7,
                      chunk_size: int = 500,
                      black_white: bool = False) -> tuple:
        """
        Calculates the normalized cross-correlation between patches and matches
        correlated corners together and returns two lists of correlated corners

        @param patches: Image Patches to find ncc
        @type patches:  Numpy Array
        @param patches2: 2nd Image Patches to find ncc
        @type patches2: Numpy Array
        @param prev_corners: Previous Corners to find best corners
        @type prev_corners: Numpy Array
        @param curr_corners: Current Corners to find best corners
        @type curr_corners: Numpy Array
        @param threshold: (default 0.7) Threshold value to select best ncc values
        @type threshold: Float
        @param chunk_size: (default 500) chunk_size to process patches x patches2
        @type chunk_size: Integer
        @param black_white: (Default False) Find corners using black white patches
        @type black_white: Bool
        @return: Image to convolve
        @rtype: Numpy Array
        """

        def subtract_mean(image):
            if black_white:
                return image - image \
                                   .mean(axis=1) \
                                   .mean(axis=1)[:, newaxis, newaxis]
            else:
                return image - image \
                                   .mean(axis=1) \
                                   .mean(axis=1)[:, newaxis, newaxis, :]

        def find_std(image):
            if black_white:
                return np_sqrt((image ** 2)\
                               .sum(axis=1)\
                               .sum(axis=1)[:,
                               newaxis, newaxis].astype(float32))
            else:
                return np_sqrt((image ** 2)\
                               .sum(axis=1)\
                               .sum(axis=1)[:,
                               newaxis, newaxis, :].astype(float32))

        # initialize parameters
        c_output = list()
        c_output2 = list()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # remove mean from f and g
        f = subtract_mean(patches)
        g = subtract_mean(patches2)

        # calculate standard deviation for f and g
        f_std = find_std(f)
        g_std = find_std(g)

        # calculate f_hat (f_hat = f / f_std) and g_hat (g_hat = g / g_std)
        if black_white:
            f_hat = torch.tensor((f / f_std).astype(float32)[:,
                                 newaxis, :]).to(device)
            g_hat = torch.tensor((g / g_std).astype(float32)[newaxis,
                                 :, :]).to(device)
        else:
            f_hat = torch.tensor((f / f_std).astype(float32)[:,
                                 newaxis, :, :]).to(device)
            g_hat = torch.tensor((g / g_std).astype(float32)[newaxis,
                                 :, :, :]).to(device)

        # calculate number of iterations to calculate ncc by chunk size
        num_of_iter = ceil(f_hat.shape[0] / chunk_size)

        # iterate from 1 to number of iterations
        for i in range(1, num_of_iter + 1):
            # Find max ncc by the sum of f_hat (according to chunk)
            # and g_hat for only dimensions 2 and 3
            ncc = (f_hat[i * chunk_size - chunk_size:i * chunk_size] * g_hat) \
                .sum(dim=2).sum(dim=2)

            if not black_white:
                ncc = ncc.mean(dim=2)

            # Calculate max ncc values for dimension 1 and threshold values
            ncc = ncc.max(dim=1)
            mask = ncc[0] > threshold
            ncc = ncc[1][mask]

            # if gpu available for pytorch then...
            if torch.cuda.is_available() and ncc.nelement() != 0:
                # append correlated corners for prev_corners and curr_corners
                c_output.append(prev_corners[i * chunk_size - chunk_size:i * chunk_size][mask.cpu().numpy()])
                c_output2.append(curr_corners[ncc.cpu().numpy()])
            elif ncc.nelement() != 0:
                # append correlated corners for prev_corners and curr_corners
                c_output.append(prev_corners[i * chunk_size - chunk_size:i * chunk_size][mask.numpy()])
                c_output2.append(curr_corners[ncc.numpy()])

        # combine both stacks of chunks together and return two numpy arrays
        return vstack(c_output), vstack(c_output2)

    def ransac(self,
               best_corners: ndarray,
               best_corners2: ndarray,
               threshold: float = 0.1,
               max_iter: int = 1000,
               sample_size: int = 4):
        def get_homograph_inliers(ind: ndarray) -> tuple:
            src_points, dest_points = best_corners[ind], best_corners2[ind]
            h = self.compute_homography(src_points, dest_points)
            pred = (h @ vstack((best_corners.T, ones(len(best_corners))))).T
            pred /= pred[:, 2].reshape(-1, 1)
            dist = np_sqrt(((pred[:, :2] - best_corners2) ** 2).sum(axis=1))
            mask = dist < threshold
            inliers = mask.sum()
            return h, inliers, dist[mask].mean()

        best_homography = None
        best_std = float('inf')
        best_num_of_inliers = 0

        for i in range(max_iter):
            indices = random.choice(best_corners.shape[0],
                                    size=sample_size,
                                    replace=False)
            homography, num_of_inliers, std = get_homograph_inliers(indices)

            if num_of_inliers > best_num_of_inliers or (num_of_inliers == best_num_of_inliers and std < best_std):
                print('Best Inlier:', num_of_inliers, ', STD:', best_std)
                best_std = std
                best_homography = homography
                best_num_of_inliers = num_of_inliers
        
        p = (best_homography @ vstack([best_corners.T, ones(best_corners.shape[0]).T])).T
        p /= p[:, 2].reshape(-1, 1)
        d = np_sqrt(((p[:, :2] - best_corners2) ** 2).sum(axis=1))
        indices = where(d < threshold)
        homography, num_of_inliers, std = get_homograph_inliers(indices)
        return indices, best_homography, num_of_inliers

    def compute_homography(self,
                           src_pts: ndarray,
                           dest_pts: ndarray):
        src_pts_norm, t1 = self.norm_points(src_pts)
        dest_pts_norm, t2 = self.norm_points(dest_pts)
        a = list()

        for i in range(src_pts_norm.shape[0]):
            x1, y1 = src_pts_norm[i]
            x2, y2 = dest_pts_norm[i]
            a.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2, -x2])
            a.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2, -y2])

        u, d, ut = linalg.svd(a)
        h = (ut[-1, :] / ut[-1, -1]).reshape(3, 3)
        return linalg.inv(t2) @ h  @ t1

    @staticmethod
    def norm_points(points: ndarray):
        points_mean = points.mean(axis=0)
        s = np_sqrt(2) / np_sqrt(((points - points_mean) ** 2).sum(axis=1)).mean()
        t = array([[s, 0, -s * points_mean[0]],
                   [0, s, -s * points_mean[1]],
                   [0, 0, 1]])
        norm_points = t @ column_stack((points, ones((len(points),)))).T
        return norm_points.T[:, :2], t

    def draw_corr_corners(self,
                          image1: Image,
                          image2: Image,
                          best_corners: ndarray,
                          best_corners2: ndarray,
                          filename_prefix: str = 'corr_corners'):
        def get_color():
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            return r, g, b

        w1, h1 = image1.size
        w2, h2 = image2.size
        image = Image.new('RGB', (max(w1, w2), h1 + h2))
        image.paste(image1, (0, 0))
        image.paste(image2, (0, h1))
        image = array(image)

        for i in range(best_corners.shape[0]):
            x1, y1 = best_corners[i].tolist()
            x2, y2 = best_corners2[i].tolist()
            y2 = y2 + h1
            cv2.line(image, (x1, y1), (x2, y2), get_color(), 1)

        fromarray(image).save('{}_{}{}_{}.png'.format(
            filename_prefix,
            self.filename_prefix,
            str(self.image_id_counter - 2).zfill(
                self.filename_zfill - 2),
            str(self.image_id_counter - 1).zfill(
                self.filename_zfill - 1)))
        
    def crop(self, image: ndarray):
        x_size = image[1,:].shape[0]
        y_size = image[:,1].shape[0]
        for i in range(y_size):
            #print(sum(sum(image[i,:])))
            if sum(sum(image[i,:])) > 255:
                top_of_img = i
                #print("Top of Image:",top_of_img)
                break
        
        for i in range(y_size-1,0,-1):
            #print(sum(sum(image[i,:])))
            if sum(sum(image[i,:])) > 255:
                bottom_of_img = i
                #print("Bottom of Image:",bottom_of_img)
                break
        
        for j in range(x_size):
            #print(sum(sum(image[:,j])))
            if sum(sum(image[:,j])) > 255:
                left_of_img = j
                #print("Left of Image:",left_of_img)
                break
        
        for j in range(x_size-1,0,-1):
            #print(sum(sum(image[:,j])))
            if sum(sum(image[:,j])) > 255:
                right_of_img = j
                #print("Right of Image:",right_of_img)
                break
        #print('Top:',top_of_img)
        #print('Bottom', bottom_of_img)
        #print('Left', left_of_img)
        #print('Right',right_of_img)    
        return image[top_of_img:bottom_of_img,left_of_img:right_of_img], left_of_img, bottom_of_img
           

    def warp_image(self,
                   image1: Image, 
                   image2: Image, 
                   h: ndarray, 
                   filename_prefix: str = 'warped_image'):

        image1 = array(image1)
        image2 = array(image2)
        height1, width1, _ = image1.shape
        height2, width2, _ = image1.shape
        
        pts1 = float32([[0, 0],
                      [0, height1],
                      [width1, height1],
                      [width1, 0]]).reshape(-1, 1, 2)
        pts2 = float32([[0, 0],
                      [0, height2],
                      [width2, height2],
                      [width2, 0]]).reshape(-1, 1, 2)
        image = np.concatenate((pts1, cv2.perspectiveTransform(pts2, h)), axis=0)
        [x_min, y_min] = np.int32(image.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(image.max(axis=0).ravel() + 0.5)
        h_t = array([[1, 0, -x_min],
                     [0, 1, -y_min],
                     [0, 0, 1]])
        print(x_min,x_max,y_min,y_max)
        warped_image = cv2.warpPerspective(image1, h_t @ h, (x_max - x_min, y_max - y_min),borderMode=0)
        shift = (-x_min,-y_min)
        pano_size_y, pano_size_x, _ = warped_image.shape
        
        #The start of blending - Average Blending
        RGBA_warped = Image.new('RGBA', size=(pano_size_x, pano_size_y), color=(0, 0, 0, 0))
        RGBA_warped.paste(fromarray(image2), shift)
        RGBA_warped.paste(fromarray(warped_image),(0, 0))
        
        newimage2 = Image.new('RGBA', size=(pano_size_x, pano_size_y), color=(0, 0, 0, 0))
        newimage2.paste(fromarray(warped_image), (0, 0))
        newimage2.paste(fromarray(image2), shift)
        
        result = cv2.addWeighted(np.array(RGBA_warped),0.1,np.array(newimage2),0.9,0)
        result = fromarray(result).convert('RGB')
        
        #Saves the warped and stitched image into the working folder
        #os.chdir(self.img_dir)
        result.save("{}_{}{}_{}.png".format(
            filename_prefix,
            self.filename_prefix,
            str(self.image_id_counter - 2).zfill(
                self.filename_zfill - 2),
            str(self.image_id_counter - 1).zfill(
                self.filename_zfill - 1)))
 

    def extra_credit(self,
                     image1: Image,
                     image2: Image):
        
        matlab_img1 = mpimg.imread(image1)
        img1plot = plt.imshow(matlab_img1)
               
        matlab_img2 = mpimg.imread(image2)    
        img2plot = plt.imshow(matlab_img2)
        print("Please select 4 coordinates")
        #Records mouse coordinates
        mouse_coords = plt.ginput(4)
        plt.close()
        
        #The following finds the corner orientation of the coordinates
        mouse_coords = sorted([(int(i), int(j)) for i, j in mouse_coords])

        left = mouse_coords[:2]
        left = sorted(left, key=lambda i: i[1])
        top_left = left[0]
        bot_left = left[1]
        
        right = mouse_coords[2:]    
        right = sorted(right, key=lambda i: i[1])
        top_right = right[0]
        bot_right = right[1]
        
        PIL_img1 = pil_open(image1)
        PIL_img2 = pil_open(image2)
        
        PIL_img1_arr = array(PIL_img1)
        PIL_img2_arr = array(PIL_img2)
        height1, width1, _ = PIL_img1_arr.shape
        height2, width2, _ = PIL_img2_arr.shape
        
        #The corner points of the image where we are warping
        img_pts1 = float32([[0, 0],
                            [0, height1],
                            [width1, height1],
                            [width1, 0]]).reshape(-1, 1, 2)
                
        #The corner points on the second image where we want the first image warped into    
        img_pts2 = float32([top_left,
                            bot_left,
                            bot_right,
                            top_right]).reshape(-1, 1, 2)
        
        #This returns the homography matrix since we know the src and dst points of the warped image.
        #The points are found with mouse clicks
        h_mat = cv2.getPerspectiveTransform(img_pts1,img_pts2)
        warped_image = cv2.warpPerspective(PIL_img1_arr, h_mat, (width2, height2))
      
        print(warped_image.shape)
                
        warped_mask = Image.new('RGBA', size=(width2,height2), color=(0, 0, 0, 0))
        warped_mask.paste(fromarray(warped_image),(0, 0))
        
        blended_image = Image.composite(fromarray(warped_image),fromarray(PIL_img2_arr),warped_mask)
        
        blended_image.save("extra_credit_image.png")     
           
if __name__ == '__main__':
    mosaic_mod = ImageMosaic(
        img_base_dir=image_setting['img_base_dir'],
        filename_prefix=image_setting['filename_prefix'],
        filename_ext=image_setting['filename_ext'],
        filename_zfill=image_setting['filename_zfill'],
        image_id_start=image_setting['image_id_start'],
        image_id_end=image_setting['image_id_end'])

    mosaic_mod.create_mosaic(
        thresh_perc=image_setting['thresh_perc'],
        block_size=image_setting['block_size'],
        k=image_setting['k'],
        patch_size=image_setting['patch_size'],
        match_corner_thresh=image_setting['match_corner_thresh'],
        match_corner_chunk_size=image_setting['match_corner_chunk_size'],
        black_white_patches=image_setting['black_white_patches'],
        ransac_threshold=image_setting['ransac_threshold'],
        ransac_max_iter=image_setting['ransac_max_iter'],
        ransac_sample_size=image_setting['ransac_sample_size'])



