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

        # extract left eye image
        facial_features_dict = face_utils.FACIAL_LANDMARKS_IDXS
        left_eye_coords = facial_features_dict['left_eye']
        (i, j) = left_eye_coords

        # extract ROI of the faces
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:J]]))
        left_eye = image[y:y+h, x:x+w]
        left_eye = imutils.resize(left_eye, width=60, height=36, inter=cv2.INTER_CUBIC)
        left_eye_resize = cv2.resize(left_eye, dsize=(60, 36), interpolation=cv2.INTER_CUBIC)

        # draw bounding box around the face
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # draw facial landmarks
        for (x, y) in landmark:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        if landmark.shape[1] != 3:
            landmark = np.append(landmark, [[0],
                                            [0],
                                            [0],
                                            [0],
                                            [0],
                                            [0]], axis=1)
            return left_eye_resize


def face_orientation(frame, landmarks):
    """
    args:       frame -> image frame of driver
                landmarks -> facial landmarks

    returns the roll, pitch and yaw angles of the facial image
    """
    size = frame.shape #(height, width, color_channel)

    image_points = np.array([
                            (landmarks[2][0], landmarks[2][1]),     # Nose tip
                            (landmarks[5][0], landmarks[5][1]),     # Chin
                            (landmarks[0][0], landmarks[0][1]),     # Left eye left corner
                            (landmarks[1][0], landmarks[1][1]),     # Right eye right corne
                            (landmarks[3][0], landmarks[3][1]),     # Left Mouth corner
                            (landmarks[4][0], landmarks[4][1])      # Right mouth corner
                        ], dtype = float)

    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

    # Camera internals
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = float
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    axis = np.float32([[500,0,0],
                          [0,500,0],
                          [0,0,500]])

    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    global feature_vector
    if len(feature_vector) == 4:
      feature_vector = np.append(feature_vector, [roll, pitch, yaw])

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[2][0], landmarks[2][1])


def head_pose(img_path):
    """
    args:       img_path -> path to image file
    returns an image with annotated pitch, roll and yaw angle axes
    """
    frame = cv2.imread(img_path)
    frame = imutils.resize(frame, width = 500)
    landmarks =  landmark

    print(img_path)
    imgpts, modelpts, rotate_degree, nose = face_orientation(frame, landmarks)

    cv2.line(frame, nose, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
    cv2.line(frame, nose, tuple(imgpts[0].ravel()), (255,0,), 3) #BLUE
    cv2.line(frame, nose, tuple(imgpts[2].ravel()), (0,0,255), 3) #RED

    remapping = [2,3,0,4,5,1]

    for j in range(len(rotate_degree)):
        cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

    cv2_imshow(frame)


def img_preprocess(img_path):
    """
    args:       img_path -> path to image file
    returns the facial feature vector and left eye image for
    a given image
    """

    #Initializing the feature vector
    global feature_vector
    feature_vector = np.array([])

    left_eye = landmarks(img_path)

    head_pose(img_path)

    global landmark

    if len(feature_vector) == 7:
        feature_vector = np.append(feature_vector, landmark[0][0])
        feature_vector = np.append(feature_vector, landmark[0][1])

    if len(feature_vector) == 9:
        feature_vector = np.append(feature_vector, landmark[1][0])
        feature_vector = np.append(feature_vector, landmark[1][1])

    if len(feature_vector) == 11:
        feature_vector = np.append(feature_vector, landmark[2][0])
        feature_vector = np.append(feature_vector, landmark[2][1])

    return feature_vector, left_eye
