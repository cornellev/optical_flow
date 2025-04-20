import math
import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
np.set_printoptions(suppress=True)

def get_rot_mx(angle):
    '''
    Input:
        angle -- Rotation angle in radians
    Output:
        A 3x3 numpy array representing 2D rotations.
    '''
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def get_trans_mx(trans_vec):
    '''
    Input:
        trans_vec -- Translation vector represented by an 1D numpy array with 2
        elements
    Output:
        A 3x3 numpy array representing 2D translation.
    '''
    assert trans_vec.ndim == 1
    assert trans_vec.shape[0] == 2

    return np.array([
        [1, 0, trans_vec[0]],
        [0, 1, trans_vec[1]],
        [0, 0, 1]
    ])

def get_scale_mx(s_x, s_y):
    '''
    Input:
        s_x -- Scaling along the x axis
        s_y -- Scaling along the y axis
    Output:
        A 3x3 numpy array representing 2D scaling.
    '''
    return np.array([
        [s_x, 0, 0],
        [0, s_y, 0],
        [0, 0, 1]
    ])


## Helper functions ############################################################

def inbounds(shape, indices):
    '''
        Input:
            shape -- int tuple containing the shape of the array
            indices -- int list containing the indices we are trying 
                       to access within the array
        Output:
            True/False, depending on whether the indices are within the bounds of 
            the array with the given shape
    '''
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    kernel = np.zeros((height, width))
    gauss = lambda x, y : (1 / (2 * np.pi * (sigma ** 2))) * (np.e ** (-((x - int(height/2)) ** 2 + (y - int(width/2)) ** 2) / (2 * (sigma ** 2))))
    for row in range(height):
        for col in range(width):
            kernel[row][col] = gauss(row, col)
    return (1/np.sum(kernel))*kernel


# WORKS JUST LIKE CV2.RESIZE
def downsample(srcImage, s, k):
    # pads image to work for nearest neighbor interpolation
    srcImage = np.pad(srcImage, ((0, srcImage.shape[0] % s), (0, srcImage.shape[1] % s)), mode='edge')
    h, w = srcImage.shape

    gauss = gaussian_blur_kernel_2d(1/(2*s), k, k)
    srcImage = scipy.ndimage.convolve(srcImage, gauss)

    blocks = srcImage.reshape(h // s, s, w // s, s).transpose(0, 2, 1, 3)
    return np.mean(blocks, axis = (2, 3))


def computeLocalMaximaHelper(harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, dtype=bool)
        destImage = ((harrisImage == scipy.ndimage.maximum_filter(harrisImage, size=(7, 7), mode="nearest")) & (harrisImage != 0))

        return destImage


def detectCorners(harrisImage, orientationImage, centers = np.array([None])):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        Output:
            features -- list of all detected features. Entries should 
            take the following form:
            (x-coord, y-coord, angle of gradient, the detector response)
            
            x-coord: x coordinate in the image
            y-coord: y coordinate in the image
            angle of the gradient: angle of the gradient in degrees
            the detector response: the Harris score of the Harris detector at this point
        '''
        h, w = harrisImage.shape[:2]
        features = []

        mask = np.zeros_like(harrisImage, dtype = object).astype(int)
        n = max(1, min(h, w) // 64)
        x = np.arange(-n, n + 1)
        y = np.arange(-n, n + 1)
        xv, yv = np.meshgrid(x, y)
        offsets = np.stack((xv.ravel(), yv.ravel()), axis=-1)

        if (np.any(centers == None)):
            mask = computeLocalMaximaHelper(harrisImage) == 1
        else:
            for center in centers:
                window_coords = center + offsets

                valid = (
                    (window_coords[:, 0] >= 0) & (window_coords[:, 0] < harrisImage.shape[0]) &
                    (window_coords[:, 1] >= 0) & (window_coords[:, 1] < harrisImage.shape[1])
                )
                coords = window_coords[valid]

                if coords.shape[0] == 0:
                    continue

                values = harrisImage[coords[:, 0], coords[:, 1]]
                max_idx = np.argmax(values)
                max_coord = coords[max_idx]
                mask[max_coord[0], max_coord[1]] = 1 if harrisImage[max_coord[0], max_coord[1]] > 0 else 0
            mask = mask.astype(bool)

        harrisMax = harrisImage[mask]
        orientationMax = orientationImage[mask]
        # row x is point on y-axis, col y is point on x-axis
        x, y = np.where(mask)
        features = np.array([x, y, orientationMax, harrisMax]).T
        return features

def getHarrisWindow(srcImage, s, k, thres):
    # windows are points
    h, w = srcImage.shape[:2]
    if (min(h, w) <= thres):
        # this gets all the centers of patches where we should recalculate Harris values for upsampled.
        y, x = np.indices(srcImage.shape)
        indices = np.stack((y.flatten(), x.flatten())).T
        harris, orientation, _ = computeHarrisValues(srcImage, indices)
        windows = detectCorners(harris, orientation)
        return windows
    
    windows = getHarrisWindow(downsample(srcImage, s, k), s,  k, thres)*2
    harris, orientation, centers = computeHarrisValues(srcImage, windows)
    detectors = detectCorners(harris, orientation, centers)
    return detectors

def computeHarrisValues(srcImage, windows):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        # compute harris values only on the windows returned by ... 7 by 7 window
        windows = windows[:, :2].astype(int)
        h, w = srcImage.shape[:2]
  
        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        gauss = gaussian_blur_kernel_2d(0.5, 5, 5)

        sobel = lambda window, axis : scipy.ndimage.sobel(window, axis = axis, mode = "nearest")
        # harris matrix H: sobel then gaussian
        img_deriv = lambda window : np.array([[sobel(window, 0) ** 2, (sobel(window, 0) * sobel(window, 1))],
                                      [(sobel(window, 1) * sobel(window, 0)), (sobel(window, 1) ** 2)]])

        det = lambda M : M[0,0] * M[1,1] - M[0,1] * M[1,0]
        trace = lambda M : M[0,0] + M[1,1]

        centers = windows

        if (srcImage.shape[0] * srcImage.shape[1] != windows.shape[0]):
            # make 7 x 7 window around windows points
            n = max(1, min(h, w) // 64)
            x = np.arange(-n, n + 1)
            y = np.arange(-n, n + 1)
            xv, yv = np.meshgrid(x, y)
            offsets = np.stack((xv.ravel(), yv.ravel()), axis=-1)

            windows = windows[:, None] + offsets[None, :]
            windows = windows.reshape(-1, 2)
            windows = windows[((windows[:, 0] >= 0) & (windows[:, 0] < srcImage.shape[0]) &
                            (windows[:, 1] >= 0) & (windows[:, 1] < srcImage.shape[1]))]
            
        mask = np.zeros_like(srcImage, dtype=bool)
        mask[windows[:, 0], windows[:, 1]] = True
        img = np.array([[np.where(mask, scipy.ndimage.convolve(img_deriv(srcImage)[0, 0], gauss), srcImage), np.where(mask, scipy.ndimage.convolve(img_deriv(srcImage)[0, 1], gauss), srcImage)],
                        [np.where(mask, scipy.ndimage.convolve(img_deriv(srcImage)[1, 0], gauss), srcImage), np.where(mask, scipy.ndimage.convolve(img_deriv(srcImage)[1, 1], gauss), srcImage)]])
        # get the maximum harrisImage value within each window
        harrisImage = det(img) - 0.1 * (trace(img) ** 2)

        orientationImage = np.degrees(np.arctan2(sobel(srcImage, 1), sobel(srcImage, 0)))
        return harrisImage, orientationImage, centers

def detect(srcImage, s, k, thres):
    h, w = srcImage.shape
    return getHarrisWindow(srcImage, s, k, max(50, thres))
    
def displayCorners(srcImage, s, k, thres, c):
    srcImageColored = cv2.cvtColor(srcImage, cv2.COLOR_GRAY2BGR)
    detector = getHarrisWindow(srcImage, s, k, max(50, thres))[:, :2].astype(int)
    for y, x in detector:
        # we flip x and y because pixels are represented by [row, col], where row = y-axis, col = x-axis
        cv2.rectangle(srcImageColored, (x-2,y-2), (x+2,y+2), color=c, thickness = -1)

    return srcImageColored


## Compute MOPS Descriptors ############################################################
def computeMOPSDescriptors(image, features):
    """"
    Input:
        image -- Grayscale input image in a numpy array with
                values in [0, 1]. The dimensions are (rows, cols).
        features -- the detected features, we have to compute the feature
                    descriptors at the specified coordinates
    Output:
        desc -- K x W^2 numpy array, where K is the number of features
                and W is the window size
    """
    image = image.astype(np.float32)
    image /= 255.
    # This image represents the window around the feature you need to
    # compute to store as the feature descriptor (row-major)
    windowSize = 8

    image = np.pad(image, ((windowSize, windowSize), (windowSize, windowSize), (0, 0)), mode='edge')
    desc = np.zeros((len(features), windowSize * windowSize * 3))
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = ndimage.gaussian_filter(grayImage, 0.5)

    image = ndimage.gaussian_filter(image, 1)
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    for i, f in enumerate(features):
        transMx = np.zeros((2, 3))

        # TODO-BLOCK-BEGIN
        x, y, angle, score = f
        M_t1 = get_trans_mx(np.array([-y + windowSize/2, -x + windowSize/2]))
        # M_r = np.vstack([cv2.getRotationMatrix2D((windowSize/2, windowSize/2), 90.-angle, 1/5), [0., 0., 1.]])
        M_r = np.vstack([cv2.getRotationMatrix2D((windowSize/2, windowSize/2), 0., 1/5), [0., 0., 1.]])

        transMx = (M_r @ M_t1)[:2]
        # TODO-BLOCK-END

        # Call the warp affine function to do the mapping
        # It expects a 2x3 matrix
        # destImage = cv2.warpAffine(grayImage, transMx, (windowSize, windowSize), flags=cv2.INTER_LINEAR)
        destImageR = cv2.warpAffine(red, transMx, (windowSize, windowSize), flags=cv2.INTER_LINEAR)
        destImageG = cv2.warpAffine(green, transMx, (windowSize, windowSize), flags=cv2.INTER_LINEAR)
        destImageB = cv2.warpAffine(blue, transMx, (windowSize, windowSize), flags=cv2.INTER_LINEAR)

        rgb_image = np.stack((destImageB, destImageG, destImageR), axis=-1)

        # TODO 5: Normalize the descriptor to have zero mean and unit
        # variance. If the variance is negligibly small (which we
        # define as less than 1e-10) then set the descriptor
        # vector to zero. Lastly, write the vector to desc.
        # TODO-BLOCK-BEGIN
        # destImage = destImage - np.mean(destImage)
        destImageR = destImageR - np.mean(destImageR)
        destImageG = destImageG - np.mean(destImageG)
        destImageB = destImageB - np.mean(destImageB)

        destImageR = destImageR/np.std(destImageR)
        destImageG = destImageG/np.std(destImageG)
        destImageB = destImageB/np.std(destImageB)

        desc[i] = np.vstack([destImageR, destImageG, destImageB]).flatten()
        # desc[i] = (destImage/np.std(destImage) if np.var(destImage) >= 1e-10 else np.zeros(destImage.shape)).flatten()
        # TODO-BLOCK-END
    return desc

## Compute Matches ############################################################
def produceMatches(desc_img1, desc_img2):
    """
    Input:
        desc_img1 -- corresponding set of MOPS descriptors for image 1
        desc_img2 -- corresponding set of MOPS descriptors for image 2

    Output:
        matches -- list of all matches. Entries should 
        take the following form:
        (index_img1, index_img2, score)

        index_img1: the index in corners_img1 and desc_img1 that is being matched
        index_img2: the index in corners_img2 and desc_img2 that is being matched
        score: the scalar difference between the points as defined
                    via the ratio test
    """
    matches = []
    assert desc_img1.ndim == 2
    assert desc_img2.ndim == 2
    assert desc_img1.shape[1] == desc_img2.shape[1]

    if desc_img1.shape[0] == 0 or desc_img2.shape[0] == 0:
        return []

    # TODO 6: Perform ratio feature matching.
    # This uses the ratio of the SSD distance of the two best matches
    # and matches a feature in the first image with the closest feature in the
    # second image. If the SSD distance is negligibly small, in this case less 
    # than 1e-5, then set the distance to 1. If there are less than two features,
    # set the distance to 0.
    # Note: multiple features from the first image may match the same
    # feature in the second image.
    # TODO-BLOCK-BEGIN
    # match first with second
    for i, desc_1 in enumerate(desc_img1):
        if desc_img2.shape[0] < 2:
            fst = 0
            ratio = 0
        else:
            ssd = np.sum((desc_1 - desc_img2) ** 2, axis = 1)
            fst, snd = np.argpartition(ssd, 2)[:2]
            ratio = ssd[fst]/ssd[snd] if ssd[snd] >= 1e-5 else 1
        matches.append((i, fst, ratio))
    # TODO-BLOCK-END

    return matches