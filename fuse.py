import cv2
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import timeit
from collections import defaultdict

np.set_printoptions(suppress=True)

class Node(object):
    def __init__(self, data=None, axis=None, left=None, right=None):
        self.data = data
        self.axis = axis
        self.left = left
        self.right = right
        self.prev = np.array([0,0,0,0])
        self.C_prev = np.array([], dtype=object)

    def make_kdtree (self, points, axis, dim):
        if (points.shape[0] <= 10):
            # left = None, right = None is a Leaf
            return Node(data = points, axis = axis) if points.shape[0] > 5 else Node()

        points = points[np.argsort(points[:, axis])]
        median = np.median(points[:, axis])
        left = points[points[:, axis] < median]
        right = points[points[:, axis] >= median]

        return Node(data = points, axis = axis, 
                    left = self.make_kdtree(points = left, axis = (axis + 1) % dim, dim = dim),
                    right = self.make_kdtree(points = right, axis = (axis + 1) % dim, dim = dim))
    
    def search (self, root, point):
        """
        like actually traverses through the kd tree and returns closest point
        """
        split_axis = np.median(self.data[:, self.axis])
        
        print(int(split_axis))
        if (self.left == None and self.right == None):
            print(self.data, point)
            return self.data[np.argmin(self.data[np.sqrt(np.sum((self.data - point)**2, axis = 1))])]
        
        if (point[self.axis] < int(split_axis)):
            point = self.left.search(root, point)
        elif (point[self.axis] >= int(split_axis)):
            point = self.right.search(root, point)
        
        return point

    def search_point(self, ax, point, radius, thres, C, C_prev):
        """
        modified version of [search] that checks whether points are within a radius, not guaranteed O(logn)
        """
        split_axis = np.median(self.data[:, self.axis])
        if (len(self.data[np.linalg.norm(self.data[:,:3] - point[0][:3], axis=1) <= radius]) > 0):
            in_radius = self.data[np.linalg.norm(self.data[:,:3] - point[0][:3], axis=1) <= radius]
                    
            return point + [in_radius[np.abs(in_radius[:,3] - point[0][3]) < thres]]
        
        if (point[0][:3][self.axis] - radius < split_axis):
            point = self.left.search_point(ax, point, radius, thres, C, C_prev)
        elif (point[0][:3][self.axis] + radius >= split_axis):
            point = self.right.search_point(ax, point, radius, thres, C, C_prev)

        return point

    def search_tree(self, ax, root, start_point, radius, thres, C, C_prev):
        """
        recursively builds the clusters
        """
        stack = [start_point]
        unexplored_set = {tuple(p) for p in Node.unexplored}

        while stack:
            # concave hull to initialize centroids
            point = stack.pop()
            neighbors = np.vstack(self.search_point(ax, [point], radius, thres, C, C_prev)[1:])
            neighbors = [tuple(p) for p in neighbors if tuple(p) in unexplored_set]
            
            for neighbor in neighbors:
                if neighbor not in C[-1]:
                    C[-1].append(neighbor)
                    stack.append(np.array(neighbor))  # Add to stack for further exploration
                    unexplored_set.remove(neighbor)

        Node.unexplored = np.array(list(unexplored_set)) if unexplored_set else np.empty((0, 3))
        return Node.unexplored

def merge(data, thres):
    merged_dict = defaultdict(list)
    for arr, x, y, z, I in data:
        if (abs(np.median(arr[:, 3]) - I) < 100):
            coords_tuple = (x,y,z,I) 
            # Append the first column (arr) to the corresponding coordinate key
            merged_dict[coords_tuple].extend(arr)
        else:
            coords_tuple = (x,y,z,np.median(arr[:, 3]))
            merged_dict[coords_tuple].extend(arr)

    # Convert the merged values back to NumPy arrays
    merged_result = np.array([[np.array(values), coords[0], coords[1], coords[2]] for coords, values in merged_dict.items()], dtype=object)
    return merged_result

def euclidean_cluster(ax, cloud, radius, intensity_threshold, MIN_CLUSTER_SIZE = 1, mode = "cartesian", cloud_prev = np.array([])):
    C = []
    prev = []

    if (mode == "spherical"):
        z, x, y = cloud[:, 0]*np.sin(cloud[:, 2])*np.cos(cloud[:, 1]), cloud[:, 0]*np.sin(cloud[:, 2])*np.sin(cloud[:, 1]), cloud[:, 0]*np.cos(cloud[:, 2])
        cloud = np.array([x, y, z, cloud[:, 3]]).T

    # uncomment to do voxel downsampling
    # o3d_pcd = o3d.geometry.PointCloud()
    # o3d_pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    # downsamp = o3d_pcd.voxel_down_sample(0.01)
    # o3d.visualization.draw_geometries([downsamp])
    # cloud = np.asarray(downsamp.points)

    Node.unexplored = np.array(cloud)

    kd_tree = Node().make_kdtree(cloud, 0, 3)

    while Node.unexplored.shape[0] != 0:
        next_point = Node.unexplored[0]
        C.append([tuple(next_point)])

        Node.unexplored = Node.unexplored[1:]
        Node.unexplored = kd_tree.search_tree(ax, kd_tree, next_point, radius, intensity_threshold, C, cloud_prev)
        prev.append(kd_tree.prev)

    clusters = np.array([np.array(cluster) for cluster in C], dtype = object)

    clusters = np.array([cluster for cluster in clusters if cluster.shape[0] > MIN_CLUSTER_SIZE], dtype = object)

    return clusters

def display_clusters(ax, clusters): 
    elev = ax.elev
    azim = ax.azim
    ax.clear()
    colors = plt.cm.hsv(np.linspace(0, 0.8, len(clusters)))
    
    # clusters, prev_centroids = clusters[:, 0], clusters[:, 1]
    # sort by closest cluster for visualization purposes
    sorted_indices = np.argsort([np.mean(arr) for arr in clusters])
    clusters = clusters[sorted_indices]

    for i, _ in enumerate(clusters):
        # DISPLAY PREVIOUS CENTROID ELLIPSOID
        data = clusters[i][:, :3]
        ax.scatter(-np.array(data)[:, 0], np.array(data)[:, 1], -np.array(data)[:, 2], color = colors[i], marker = 'o', alpha = 0.75, label=f'Cluster {i+1}')

    ax.legend()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=elev, azim=azim)
    plt.pause(0.1)

