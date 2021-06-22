# -*- coding: utf-8 -*-
"""pim.py

__author__ = "Andre Temple"

Python Image Library
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt

# show image 1 and image 2
def show_img(img1, img2):
  fig = plt.figure()
  fig.set_figheight(10)
  fig.set_figwidth(10)

  fig.add_subplot(1,2,1)
  plt.imshow(img1, cmap='gray')

  # display the new image
  fig.add_subplot(1,2,2)
  plt.imshow(img2, cmap='gray')

  plt.show(block=True)

# transform a rgb to gray
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

# scale an image with given height and width
def scale(im, nH, nW):
  height = len(im)
  width = len(im[0]) 
  list = [[ im[int(height * r / nH)][int(width * c / nW)]  
             for c in range(nW)] for r in range(nH)]
  return np.asarray(list)

# get identity matrix
def identity():
  return np.array([
        [1, 0, 0], 
        [0, 1, 0], 
        [0, 0, 1]
  ])

# get translation matrix
def translation(tx, ty):
  return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
  ])

# get scaling matrix
def scaling(s):
  return np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, 1] 
  ])

# get rotation matrix
def rotation(theta):
  theta = math.radians(theta)
  return np.array([
        [np.cos(theta), -np.sin(theta), 0], 
        [np.sin(theta), np.cos(theta), 0], 
        [0,0,1]
  ])

# get shear matrix
def shear(sv, sh):
  return np.array([
        [1, sv, 0], 
        [sh, 1, 0], 
        [0 , 0, 1]
  ])

# transform an image with a given identity transformation matrix
def transform(identity, image):
  width, height = image.shape[0], image.shape[1]
  new_img = np.zeros(shape=image.shape, dtype = int)
  for i in range(width):
    for j in range(height):
      newpos = np.matmul(identity, [i, j, 1]).astype(int)
      if newpos[0] >= 0 and newpos[0] < new_img.shape[0] and newpos[1] >= 0 and newpos[1] < new_img.shape[1]:
        new_img[newpos[0], newpos[1]] = image[i, j]
  return new_img

# convert an image with binary 0 and 1
def binary(image_matrix, thresh_val):
  white = 255
  black = 0

  initial_conv = np.where((image_matrix <= thresh_val), image_matrix, white)
  final_conv = np.where((initial_conv > thresh_val), initial_conv, black)

  return final_conv

# exponential tranformation
def exponential(img):
  a = 255/(np.log(1 + np.max(img)))
  exp_img = a * (np.exp(img) - 1)
  return np.array(exp_img, dtype = np.uint8)

# logaritmic tranformation
def logaritmic(img):
  a = 255/(np.log(1 + np.max(img)))
  log_img = a * np.log(1 + img)
  return np.array(log_img, dtype = np.uint8)

# square tranformation
def square(img):
  a = 255/(np.log(1 + np.max(img)))
  sqrt_img = a * np.square(img)
  return np.array(sqrt_img, dtype = np.uint8)

# sqrt tranformation
def sqrt(img):
  a = 255/(np.log(1 + np.max(img)))
  sqrt_img = a * np.sqrt(img)
  return np.array(sqrt_img, dtype = np.uint8)

# exponential if intensity < 128, logaritmic otherwise
def exp_log(img):
  a = 255/(np.log(1 + np.max(img)))
  img_n = np.where(img < 128, a * (np.exp(img) - 1), a * np.log(1 + img))
  return np.array(img_n, dtype = np.uint8)

# get image histogram
def histogram(img):
  m, n = img.shape
  h = [0.0] * 256
  for i in range(m):
    for j in range(n):
      h[img[i, j]]+=1
  return np.array(h)/(m*n)

# histogram equalization
def histogram_eq(img):
	h = histogram(img)
	cdf = np.array(cumsum(h))
	sk = np.uint8(255 * cdf)
	s1, s2 = img.shape
	img_n = np.zeros_like(img)
	for i in range(0, s1):
		for j in range(0, s2):
			img_n[i, j] = sk[img[i, j]]
	return img_n

# show histogram 1 and histogram 2
def show_histogram(hist1, hist2):
  fig = plt.figure()
  fig.set_figheight(10)
  fig.set_figwidth(10)

  fig.add_subplot(221)
  plt.plot(hist1)

  # display the new image
  fig.add_subplot(222)
  plt.plot(hist2)

  plt.show(block=True)

# add salt and pepper noise to image
def add_sp_noise(img, prob):
  img_n = np.zeros(img.shape, np.uint8)
  thres = 1 - prob 
  for i in range(img.shape[0]):
      for j in range(img.shape[1]):
          rdn = random.random()
          if rdn < prob:
              img_n[i][j] = 0
          elif rdn > thres:
              img_n[i][j] = 255
          else:
              img_n[i][j] = img[i][j]
  return img_n.astype(np.uint8)

# mean filter
def mean_filter(img):
  m, n = img.shape
  img_n = np.zeros([m, n])
  
  for i in range(1, m-1):
      for j in range(1, n-1):
        img_n[i, j] = int((img[i-1, j-1] + img[i-1, j] + img[i-1, j+1] + 
                          img[i, j-1]   + img[i, j]   + img[i, j+1] +
                          img[i+1, j-1] + img[i+1, j] + img[i+1, j+1]) / 9)
  return img_n.astype(np.uint8)

# median filter
def median_filter(img):
  m, n = img.shape
  img_n = np.zeros([m, n])
  
  for i in range(1, m-1):
      for j in range(1, n-1):
          temp = [
                  img[i-1, j-1],
                  img[i-1, j],
                  img[i-1, j+1],
                  img[i, j-1],
                  img[i, j],
                  img[i, j+1],
                  img[i+1, j-1],
                  img[i+1, j],
                  img[i+1, j+1]]
            
          temp = sorted(temp)
          img_n[i, j]= temp[4]
    
  return img_n.astype(np.uint8)

# gaussian filter
def gaussian_filter(img):
  m, n = img.shape
  img_n = np.copy(img)

  kernel = [[1 / 256, 4  / 256,  6 / 256,  4 / 256, 1 / 256],
		        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
		        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
		        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
            [1 / 256, 4  / 256,  6 / 256,  4 / 256, 1 / 256]]

  offset = len(kernel) // 2
  
  for x in range(offset, m - offset):
    for y in range(offset, n - offset):
      acc = 0
      for a in range(len(kernel)):
        for b in range(len(kernel)):
          xn = x + a - offset
          yn = y + b - offset
          pixel = img[xn, yn]
          acc += pixel * kernel[a][b]
      img_n[x, y] = int (acc)

  return img_n.astype(np.uint8)

# highpass filter
def highpass_filter(img):
  m, n = img.shape
  img_n = np.copy(img)

  kernel = [[ -1 , -1 , -1 ],
            [ -1 ,  8 , -1 ],
            [ -1 , -1 , -1 ]]

  offset = len(kernel) // 2
  
  for x in range(offset, m - offset):
    for y in range(offset, n - offset):
      acc = 0
      for a in range(len(kernel)):
        for b in range(len(kernel)):
          xn = x + a - offset
          yn = y + b - offset
          pixel = img[xn, yn]
          acc += pixel * kernel[a][b]
      img_n[x, y] = int (acc)

  return img_n.astype(np.uint8)

# low pass gaussian filter
def low_pass_gaussian_filter(img):
  m, n = img.shape
  img_n = np.copy(img)

  kernel = [[1/16, 1/8, 1/16], 
            [1/8, 1/4, 1/8], 
            [1/16, 1/8, 1/16]]

  offset = len(kernel) // 2
  
  for x in range(offset, m - offset):
    for y in range(offset, n - offset):
      acc = 0
      for a in range(len(kernel)):
        for b in range(len(kernel)):
          xn = x + a - offset
          yn = y + b - offset
          pixel = img[xn, yn]
          acc += pixel * kernel[a][b]
      img_n[x, y] = int (acc)

  return img_n.astype(np.uint8)

# roberts border detect
def roberts(img):
  m, n = img.shape
  img_n = np.zeros([m, n])
  
  Gx = np.array([[1.0,  0.0], [ 0.0, -1.0]])
  Gy = np.array([[0.0, -1.0], [ 1.0,  0.0]])

  for i in range(1, m-2):
      for j in range(1, n-2):
        gx = np.sum(np.multiply(Gx, img[i:i + 2, j:j + 2]))  
        gy = np.sum(np.multiply(Gy, img[i:i + 2, j:j + 2]))  
        img_n[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)

  return img_n.astype(np.uint8)

# prewitt border detect
def prewitt(img):
  m, n = img.shape
  img_n = np.zeros([m, n])
  
  Gx = np.array([[-1.0,  0.0,  1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
  Gy = np.array([[-1.0, -1.0, -1.0], [ 0.0, 0.0, 0.0], [ 1.0, 1.0, 1.0]])

  for i in range(1, m-2):
      for j in range(1, n-2):
        gx = np.sum(np.multiply(Gx, img[i:i + 3, j:j + 3]))  
        gy = np.sum(np.multiply(Gy, img[i:i + 3, j:j + 3]))  
        img_n[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)

  return img_n.astype(np.uint8)

# sobel border detect
def sobel(img):
  m, n = img.shape
  img_n = np.zeros([m, n])
  
  Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [ 1.0,  0.0, -1.0]])
  Gy = np.array([[1.0, 2.0,  1.0], [0.0, 0.0,  0.0], [-1.0, -2.0, -1.0]])

  for i in range(1, m-2):
      for j in range(1, n-2):
        gx = np.sum(np.multiply(Gx, img[i:i + 3, j:j + 3]))  
        gy = np.sum(np.multiply(Gy, img[i:i + 3, j:j + 3]))  
        img_n[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)

  return img_n.astype(np.uint8)

# laplacian border detect
def laplacian(img):
  m, n = img.shape
  img_n = np.zeros([m, n])
  
  kernel = [[-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]]

  offset = len(kernel) // 2
  
  for x in range(offset, m - offset):
    for y in range(offset, n - offset):
      acc = 0
      for a in range(len(kernel)):
        for b in range(len(kernel)):
          xn = x + a - offset
          yn = y + b - offset
          pixel = img[xn, yn]
          acc += pixel * kernel[a][b]
      img_n[x, y] = int (acc)

  return img_n.astype(np.uint8)

# global threshold
def global_threshold(img, T):
  m, n = img.shape
  img_n = np.zeros([m, n])

  for i in range(1, m):
      for j in range(1, n):
        pix = img[i,j]
        if (pix <= T[0] and pix > T[1]):
          pix = 255
        elif (pix <= T[1] and pix > T[2]):
          pix = pix * 0.7
        elif (pix <= T[2] and pix > T[3]):
          pix = pix * 0.3
        else:
          pix = 0

        img_n[i,j] = pix

  return img_n.astype(np.uint8)

# image erosion
def erosion(img, element):
    img = np.asarray(img)
    element = np.asarray(element)
    m, n = img.shape
    m_el, n_el = element.shape
    m_eo, n_eo = (int(np.ceil((m_el - 1) / 2.0)), int(np.ceil((n_el - 1) / 2.0)))
    img_n = np.zeros([m, n])

    for i in range(m):
        for j in range(n):
            overlap = img[idx_check(i - m_eo):i + (m_el - m_eo),
                          idx_check(j - n_eo):j + (n_el - n_eo)]
            m_s, n_s = overlap.shape

            e_first_m_idx = int(np.fabs(i - m_eo)) if i - m_eo < 0 else 0
            e_first_n_idx = int(np.fabs(j - n_eo)) if j - n_eo < 0 else 0
            e_last_m_idx = m_el - 1 - (i + (m_el - m_eo) - m) if i + (m_el - m_eo) > m else m_el - 1
            e_last_n_idx = n_el - 1 - (j + (n_el - n_eo) - n) if j + (n_el - n_eo) > n else n_el - 1

            if m_s != 0 and n_s != 0 and np.array_equal(np.logical_and(overlap, 
                                                                       element[e_first_m_idx:e_last_m_idx+1, e_first_n_idx:e_last_n_idx+1]), 
                                                                       element[e_first_m_idx:e_last_m_idx+1, e_first_n_idx:e_last_n_idx+1]):
                img_n[i, j] = 255

    return img_n.astype(np.uint8)

# image dilation
def dilation(img, element):
    img = np.asarray(img)
    element = np.asarray(element)
    m, n = img.shape
    m_el, n_el = element.shape
    m_eo, n_eo = (int(np.ceil((m_el - 1) / 2.0)), int(np.ceil((n_el - 1) / 2.0)))
    img_n = np.zeros([m, n])

    for i in range(m):
        for j in range(n):
            overlap = img[idx_check(i - m_eo):i + (m_el - m_eo),
                          idx_check(j - n_eo):j + (n_el - n_eo)]
            m_s, n_s = overlap.shape

            e_first_m_idx = int(np.fabs(i - m_eo)) if i - m_eo < 0 else 0
            e_first_n_idx = int(np.fabs(j - n_eo)) if j - n_eo < 0 else 0
            e_last_m_idx = m_el - 1 - (i + (m_el - m_eo) - m) if i + (m_el - m_eo) > m else m_el - 1
            e_last_n_idx = n_el - 1 - (j + (n_el - n_eo) - n) if j + (n_el - n_eo) > n else n_el - 1

            if m_s != 0 and n_s != 0 and np.logical_and(element[e_first_m_idx:e_last_m_idx+1, e_first_n_idx:e_last_n_idx+1], overlap).any():
                img_n[i, j] = 255

    return img_n.astype(np.uint8)

# image opening operation
def opening(img, element):
  img_e = erosion(img, element)
  img_d = dilation(img_e, element)
  return img_d.astype(np.uint8)

# image closing operation
def closing(img, element):
  img_d = dilation(img, element)
  img_e = erosion(img_d, element)
  return img_e.astype(np.uint8)

# extract a border from an image
def border_extract(img, element):
  img = np.asarray(img)
  
  img_e = erosion(img, element)
  img_s = np.subtract(img, img_e)

  return img_s.astype(np.uint8)

def skeleton(img, element):
  img = np.asarray(img)
  skel = np.zeros(img.shape, np.uint8)

  while True:
      open = opening(img, element)
      temp = np.subtract(img, open)
      eroded = erosion(img, element)
      skel = np.add(skel, temp)
      img = eroded.copy()
      if np.count_nonzero(img) == 0:
          break

  return skel.astype(np.uint8)
