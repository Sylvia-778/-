# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:05:46 2020

@author: LIQI JIANG - z5253065
"""

#......IMPORT .........
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from collections import defaultdict


def task1(image):
    threshold = 127   # select 127 as starting threshold
    height, width = image.shape[:2]
    thres_list = []
    while(True):
        thres_list.append(threshold)
        new_threshold = round((image[image < threshold].mean() + image[image >= threshold].mean()) / 2, 1)
        difference = threshold - new_threshold
        threshold = new_threshold
        if -0.1 < difference < 0.1:
            break
    for h in range(0, height):
        for w in range(0, width):
            if image[h][w] < threshold:
                image[h][w] = 255
            else:
                image[h][w] = 0
    #print(thres_list[-1])
    #print(thres_list)
    #print(image)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    title = "Threshold Value = "+str(int(threshold))
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    iteration = [i for i in range(1, len(thres_list)+1)]
    plt.plot(iteration, thres_list)
    plt.title("threshold value at every iteration")
    plt.xlabel("iteration")
    plt.ylabel("threshold value")
    for a, b in zip(iteration, thres_list):
        plt.text(a, b + 0.1, '%.1f' % b, ha='center', va='bottom', fontsize=7)
    fname = "./" + folder_name + "/" + input_file[:-4] + "_Task1.png"  # Input_filename_Task1.png
    plt.savefig(fname)
    plt.show()
    return image


def median_filter(src, ksize=5):
    height, width = src.shape[:2]
    edge = int((ksize-1)/2)
    if height-1-edge <= edge or width-1-edge <= edge:
        print("The ksize is too large")
        return None
    dst = np.zeros((height, width), dtype="uint8")
    for h in range(height):
        for w in range(width):
            up = max(0, h-edge)
            down = min(height-1, h+edge)
            left = max(0, w-edge)
            right = min(width - 1, w+edge)
            dst[h, w] = np.median(src[up:down+1, left:right+1])
    # plt.axis('off')
    # plt.title("Median Filter image")
    # plt.imshow(dst, cmap="gray")
    # plt.show()
    return dst


def two_pass(image):
    label = 0
    height, width = image.shape[:2]
    labels = np.full((height, width), 0)
    equivalence = []
    flag = 0
    # first pass
    for h in range(0, height, 1):
        for w in range(0, width, 1):
            if image[h, w] > 0:
                continue
            neighbour_h_up = max(0, h-1)
            neighbour_h_down = min(h+1, height-1)
            neighbour_w_left = max(0, w-1)
            neighbour_w_right = min(w+1, width-1)
            neighbour_label = list(np.array(labels[neighbour_h_up: neighbour_h_down+1, neighbour_w_left: neighbour_w_right+1]).flatten())
            while flag in neighbour_label:
                neighbour_label.remove(flag)
            # if no neighbours, uniquely label the current pixel
            if not neighbour_label:
                label += 1
                equivalence.append(label)
                labels[h][w] = label
            else:
                labels[h][w] = min(neighbour_label)
                for l in set(neighbour_label):
                    equivalence[l-1] = equivalence[labels[h][w]-1]
    # second pass
    lowest_label = []   # record the the smallest equivalent labels
    # area = defaultdict(lambda: 0)
    for h in range(0, height, 1):
        for w in range(0, width, 1):
            if image[h, w] > 0:
                continue
            while equivalence[labels[h][w] - 1] != labels[h][w]:
                labels[h][w] = equivalence[labels[h][w]-1]
            # area[labels[h][w]] += 1
            if labels[h][w] not in lowest_label:
                lowest_label.append(labels[h][w])
    num_of_label = len(lowest_label)
    # compute the average of connected components using the area above
    #sum = 0
    #for value in area.values():
    #    sum += value
    #avg = sum//45
    #print("avg = ",avg)
    #i = 0
    #for value in area.values():
    #    if value < avg:
    #        i += 1
    #print(i)
    #plt.imshow(labels)
    #plt.show()
    return labels, num_of_label


def task2(image):
    median_image = median_filter(image)
    label_image, rice_kernel_num = two_pass(median_image)
    plt.axis('off')
    plt.imshow(median_image, cmap="gray")
    title = "Number of rice kernels = "+str(rice_kernel_num)
    plt.title(title)
    fname = "./" + folder_name + "/" + input_file[:-4] + "_Task2.png"  # Input_filename_Task2.png
    plt.savefig(fname)
    plt.show()
    return label_image


def task3(image, minimun_area):
    label_area = defaultdict(lambda: 0)    # record the pixel areas of each label
    height, width = image.shape[:2]
    for h in range(height):
        for w in range(width):
            if image[h][w] != 0:
                label_area[image[h][w]] += 1
    total_num = len(label_area.keys())
    damaged_label = []
    for h in range(height):
        for w in range(width):
            if label_area[image[h][w]] <= minimun_area:
                if image[h][w] != 0 and image[h][w] not in damaged_label:
                    damaged_label.append(image[h][w])
                image[h][w] = 255
            else:
                image[h][w] = 0
    # print(len(damaged_label), total_num)
    percentage = round(len(damaged_label)/total_num*100, 2)
    title = "percentage of damaged rice kernels = "+str(percentage)+"%"
    plt.axis('off')
    plt.imshow(image, cmap="gray")
    plt.title(title)
    fname = "./" + folder_name + "/" + input_file[:-4] + "_Task3.png"  # Input_filename_Task3.png
    plt.savefig(fname)
    plt.show()


my_parser = argparse.ArgumentParser(description="COMP9517 Assignment1")
my_parser.add_argument('-o', '--OP_folder', type=str, help='Output folder name', default='OUTPUT')
my_parser.add_argument('-m', '--min_area', type=int, action='store', required=True, help='Minimum pixel area to be occupied, to be considered a whole rice kernel')
my_parser.add_argument('-f', '--input_filename', type=str, action='store', required=True, help='Filename of image ')
# Execute parse_args()
args = my_parser.parse_args()
input_file = args.input_filename
min_area = args.min_area
folder_name = args.OP_folder
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
#print(args)
img = cv2.imread(input_file)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
img_isodata = task1(img_gray)
label_img = task2(img_isodata)
task3(label_img, min_area)
#thresh = filters.threshold_isodata(img_gray)
#print(thresh)
#img_median = cv2.medianBlur(img_isodata, 5)
#labeled_img, num = label(img_median, connectivity=2, background=255, return_num=True)
#print(num)



