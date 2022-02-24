#!/usr/bin/env python3

import cv2
import dlib
import imutils
import math
import numpy as np

from imutils import face_utils

global feature_vector
global landmark
feature_vector = np.array([])
landmark = np.array([])


def rect_to_bb(rect):
    """
    Convert a bounding box to (x, y, w, h) coordinates
    args:       rect -> bounding box
    returns:    tuple of coordinates (x, y, w, h)
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    global feature_vector
    if len(feature_vector) == 0:
        feature_vector = np.append(feature_vector, [x, y, w, h])

    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize all coordinates to 0 (68 facial landmarks)
    coordinates = np.zeros((6, 2), dtype=dtype)

    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(j).y)

    return coordinates


def landmarks(path_to_image, path_to_shapes):
    """
    Detect facial landmarks
    args:       path_to_image -> path to drivers face image
                path_to_shapes -> path to shape predictor file (shape_predictor.dat)
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_shapes)
    image = cv2.imread(path_to_image)
    image = imutils.resize(image, width=500) # resize the image

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over all face detectoins
    for (i, rect) in enumerate(rects):
        # detect facial landmarks and conver to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        global landmark
        landmark = np.array([shape[36], shape[45], shape[30], shape[48], shape[54], shape[8]])
