import math
import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
np.set_printoptions(suppress=True)

# MAKE THIS SHIT FASTER:
# downsample -> find corners -> upsample -> use window around where harris corner score highest to find corner, instead of
# convolving entire image

## HELPER FUNCTIONS FOR TRANSFORMATIONS - OPTIONAL TO USE ######

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

## Compute Harris Values ############################################################
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
    # TODO-BLOCK-BEGIN
    kernel = np.zeros((height, width))
    gauss = lambda x, y : (1 / (2 * np.pi * (sigma ** 2))) * (np.e ** (-((x - int(height/2)) ** 2 + (y - int(width/2)) ** 2) / (2 * (sigma ** 2))))
    for row in range(height):
        for col in range(width):
            kernel[row][col] = gauss(row, col)
    return (1/np.sum(kernel))*kernel
    # TODO-BLOCK-END

def computeHarrisValues(srcImage):
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
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'. Also compute an 
        # orientation for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN

        gauss = gaussian_blur_kernel_2d(0.5, 5, 5)

        sobel = lambda window, axis : scipy.ndimage.sobel(window, axis = axis, mode = "nearest")
        w_p = lambda window : scipy.ndimage.gaussian_filter(window, sigma = 0.5)
        # harris matrix H: sobel then gaussian
        img_deriv = lambda window : np.array([[sobel(window, 0) ** 2, (sobel(window, 0) * sobel(window, 1))],
                                      [(sobel(window, 1) * sobel(window, 0)), (sobel(window, 1) ** 2)]])
        H = lambda window : np.array([[np.sum(window[0, 0]), np.sum(window[0, 1])],
                                      [np.sum(window[1, 0]), np.sum(window[1, 1])]])
        det = lambda M : M[0,0] * M[1,1] - M[0,1] * M[1,0]
        trace = lambda M : M[0,0] + M[1,1]

        # srcImage = scipy.ndimage.convolve(srcImage, gauss)
        img = np.array([[scipy.ndimage.convolve(img_deriv(srcImage)[0, 0], gauss), scipy.ndimage.convolve(img_deriv(srcImage)[0, 1], gauss)],
                        [scipy.ndimage.convolve(img_deriv(srcImage)[1, 0], gauss), scipy.ndimage.convolve(img_deriv(srcImage)[1, 1], gauss)]])
        harrisImage = det(img) - 0.1 * (trace(img) ** 2)
        orientationImage = np.degrees(np.arctan2(sobel(srcImage, 0), sobel(srcImage, 1)))
        return harrisImage, orientationImage

# WORKS JUST LIKE CV2.RESIZE
def downsample(srcImage, s, k):
    # pads image to work for nearest neighbor interpolation
    srcImage = np.pad(srcImage, ((0, srcImage.shape[0] % s), (0, srcImage.shape[1] % s)), mode='edge')
    h, w = srcImage.shape

    gauss = gaussian_blur_kernel_2d(1/(2*s), k, k)
    srcImage = scipy.ndimage.convolve(srcImage, gauss)

    blocks = srcImage.reshape(h // s, s, w // s, s).transpose(0, 2, 1, 3)
    return np.mean(blocks, axis = (2, 3)).astype(np.uint8)

## Compute Corners From Harris Values ############################################################

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

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        destImage = harrisImage == scipy.ndimage.maximum_filter(harrisImage, size=(7,7), mode="nearest")
        # TODO-BLOCK-END

        return destImage


def detectCorners(harrisImage, orientationImage):
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
        height, width = harrisImage.shape[:2]
        features = []

        # TODO 3: Select the strongest keypoints in a 7 x 7 area, according to
        # the corner strength function. Once local maxima are identified then 
        # construct the corresponding corner tuple of each local maxima.
        # Return features, a list of all such features.
        # TODO-BLOCK-BEGIN
        mask = computeLocalMaximaHelper(harrisImage) == 1
        harrisMax = harrisImage[mask]
        orientationMax = orientationImage[mask]
        y, x = np.where(mask)
        features = np.array([x, y, orientationMax, harrisMax]).T
        # TODO-BLOCK-END

        return features

def getHarrisWindow(srcImage, s, k):
    h, w = srcImage.shape[:2]
    if (h * w <= 5000):
        # this gets all the centers of patches where we should recalculate Harris values for upsampled.
        harris, orientation = computeHarrisValues(srcImage, np.ones(srcImage.shape))
        windows = detectCorners(harris, orientation)[:, :2].astype(int)
        return srcImage, windows

    srcImage, windows = getHarrisWindow(downsample(srcImage, s, k), s,  k) 
    harris, orientation = computeHarrisValues(srcImage, windows)
    return detectCorners(harris, orientation)[:, :2].astype(int)

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
        h, w = srcImage.shape[:2]
        print(windows)
  
        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'. Also compute an 
        # orientation for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN

        gauss = gaussian_blur_kernel_2d(0.5, 5, 5)

        sobel = lambda window, axis : scipy.ndimage.sobel(window, axis = axis, mode = "nearest")
        # harris matrix H: sobel then gaussian
        img_deriv = lambda window : np.array([[sobel(window, 0) ** 2, (sobel(window, 0) * sobel(window, 1))],
                                      [(sobel(window, 1) * sobel(window, 0)), (sobel(window, 1) ** 2)]])

        det = lambda M : M[0,0] * M[1,1] - M[0,1] * M[1,0]
        trace = lambda M : M[0,0] + M[1,1]

        # srcImage = scipy.ndimage.convolve(srcImage, gauss)
        img = np.array([[scipy.ndimage.convolve(img_deriv(srcImage)[0, 0], gauss), scipy.ndimage.convolve(img_deriv(srcImage)[0, 1], gauss)],
                        [scipy.ndimage.convolve(img_deriv(srcImage)[1, 0], gauss), scipy.ndimage.convolve(img_deriv(srcImage)[1, 1], gauss)]])
        harrisImage = det(img) - 0.1 * (trace(img) ** 2)
        orientationImage = np.degrees(np.arctan2(sobel(srcImage, 0), sobel(srcImage, 1)))
        return harrisImage, orientationImage

s, k = 2, 5
img = cv2.imread("img1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(getHarrisImage(img, np.ones(img)))
# down = downsample(img, s, k)
# down = downsample(down, s, k)
# down = downsample(down, s, k)

# harris, orientation = computeHarrisValues(down)
# getHarrisWindow(img, s, k)

# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# for x, y in detectCorners(harris, orientation)[:, :2].astype(int):
#     cv2.circle(img, (x*img.shape[0]//down.shape[0],y*img.shape[1]//down.shape[1]), radius = 1, color = (20, 255, 57))

# print(cv2.imwrite("harris.png", img))

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
    desc = np.zeros((len(features), windowSize * windowSize))
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = ndimage.gaussian_filter(grayImage, 0.5)

    for i, f in enumerate(features):
        transMx = np.zeros((2, 3))

        # TODO 4: Compute the transform as described by the feature
        # location/orientation and store in 'transMx.' You will need
        # to compute the transform from each pixel in the 40x40 rotated
        # window surrounding the feature to the appropriate pixels in
        # the 8x8 feature descriptor image. 'transformations.py' has
        # helper functions that might be useful
        # Note: use grayImage to compute features on, not the input image
        
        # TODO-BLOCK-BEGIN
        x, y, angle, score = f
        M_t1 = get_trans_mx(np.array([-x, -y]))
        M_r = get_rot_mx(-np.radians(angle))
        M_s = get_scale_mx(1/5, 1/5)
        M_t2 = get_trans_mx(np.array([4, 4]))
        transMx = (M_t2 @ M_s @ M_r @ M_t1)[:2]
        # TODO-BLOCK-END

        # Call the warp affine function to do the mapping
        # It expects a 2x3 matrix
        destImage = cv2.warpAffine(grayImage, transMx, (windowSize, windowSize), flags=cv2.INTER_LINEAR)

        # TODO 5: Normalize the descriptor to have zero mean and unit
        # variance. If the variance is negligibly small (which we
        # define as less than 1e-10) then set the descriptor
        # vector to zero. Lastly, write the vector to desc.
        # TODO-BLOCK-BEGIN
        destImage = destImage- np.mean(destImage)
        desc[i] = (destImage/np.std(destImage) if np.var(destImage) >= 1e-10 else np.zeros(destImage.shape)).flatten()
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
            ssd = np.sum((desc_1[:3] - desc_img2[:, :3]) ** 2, axis = 1)
            fst, snd = np.argpartition(ssd, 2)[:2]
            ratio = ssd[fst]/ssd[snd] if ssd[snd] >= 1e-5 else 1
        matches.append((i, fst, ratio))
    # TODO-BLOCK-END

    return matches
