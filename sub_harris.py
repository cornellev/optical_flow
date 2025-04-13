import math
import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
np.set_printoptions(suppress=True)

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
    return np.mean(blocks, axis = (2, 3)).astype(np.uint8)


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
        destImage = ((harrisImage == scipy.ndimage.maximum_filter(harrisImage, size=(5,5), mode="nearest")) & (harrisImage > 0))

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
        n = max(1, min(min(h, w) // 32, 20))
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
        y, x = np.where(mask)
        features = np.array([x, y, orientationMax, harrisMax]).T
        return features

def getHarrisWindow(srcImage, s, k, thres):
    # windows are points
    h, w = srcImage.shape[:2]
    if (min(h,w) <= thres):
        # this gets all the centers of patches where we should recalculate Harris values for upsampled.
        y, x = np.indices(srcImage.shape)
        indices = np.stack((y.flatten(), x.flatten())).T
        harris, orientation, _ = computeHarrisValues(srcImage, indices)
        windows = detectCorners(harris, orientation)[:, :2].astype(int)
        return windows

    windows = getHarrisWindow(downsample(srcImage, s, k), s,  k, thres)*2
    harris, orientation, centers = computeHarrisValues(srcImage, windows)

    return detectCorners(harris, orientation, centers)[:, :2].astype(int)

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
  
        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        gauss = gaussian_blur_kernel_2d(0.5, 5, 5)

        sobel = lambda window, axis : scipy.ndimage.sobel(window, axis = axis, mode = "nearest")
        # harris matrix H: sobel then gaussian
        img_deriv = lambda window : np.array([[sobel(window, 0) ** 2, (sobel(window, 0) * sobel(window, 1))],
                                      [(sobel(window, 1) * sobel(window, 0)), (sobel(window, 1) ** 2)]])

        det = lambda M : M[0,0] * M[1,1] - M[0,1] * M[1,0]
        trace = lambda M : M[0,0] + M[1,1]

        # make 7 x 7 window around windows points
        n = max(1, min(min(h, w) // 32, 20))
        x = np.arange(-n, n + 1)
        y = np.arange(-n, n + 1)
        xv, yv = np.meshgrid(x, y)
        offsets = np.stack((xv.ravel(), yv.ravel()), axis=-1)

        centers = windows
        windows = windows[:, None, :] + offsets[None, :, :]
        windows = windows.reshape(-1, 2)
        windows = windows[((windows[:, 0] >= 0) & (windows[:, 0] < srcImage.shape[0]) &
                           (windows[:, 1] >= 0) & (windows[:, 1] < srcImage.shape[1]))]

        mask = np.zeros_like(srcImage, dtype=bool)
        mask[windows[:, 0], windows[:, 1]] = True
        
        img = np.array([[np.where(mask, scipy.ndimage.convolve(img_deriv(srcImage)[0, 0], gauss), srcImage), np.where(mask, scipy.ndimage.convolve(img_deriv(srcImage)[0, 1], gauss), srcImage)],
                        [np.where(mask, scipy.ndimage.convolve(img_deriv(srcImage)[1, 0], gauss), srcImage), np.where(mask, scipy.ndimage.convolve(img_deriv(srcImage)[1, 1], gauss), srcImage)]])

        # get the maximum harrisImage value within each window
        harrisImage = det(img) - 0.1 * (trace(img) ** 2)

        orientationImage = np.degrees(np.arctan2(sobel(srcImage, 0), sobel(srcImage, 1)))
        return harrisImage, orientationImage, centers

s, k = 2, 5
img = cv2.imread("sobel.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = downsample(img, s, k)
# img = downsample(img, s, k)
# print(img.shape)

# cv2.imwrite("tri.png", img)

# print(np.min(img.shape[:2])//5)
imagg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

detector = getHarrisWindow(img, s, k, np.min(img.shape[:2])//5).astype(int)
print(len(detector))

for x, y in detector:
    cv2.circle(imagg, [x,y], radius = 5, color = (0, 0, 255), thickness = -1)

cv2.imwrite("harris.png", imagg)