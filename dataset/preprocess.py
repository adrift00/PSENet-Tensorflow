# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image

import util
import cv2
import random
# import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
from shapely.geometry import Polygon
import tensorflow as tf

from configuration import TRAIN_CONFIG
config = TRAIN_CONFIG
random.seed(123456)

_R_MEAN, _G_MEAN, _B_MEAN = 123., 117., 104.


def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def tf_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary.

    Returns:
      Centered image.
    """
    mean = tf.constant(means, dtype=image.dtype)
    image = image + mean
    if to_int:
        image = tf.cast(image, tf.int32)
    return image


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs


def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def random_scale(img, min_size, ran_scale=[0.5, 1.0, 2.0, 3.0]):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[0:2]
    random_scale = np.array(ran_scale)
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    # padding the image
    if h < th or w < tw:
        for idx in range(len(imgs)):
            image = imgs[idx]
            color = [123., 117., 104.] if len(image.shape) == 3 else [0]
            top = (th-h)//2 if th-h > 0 else 0
            bottom = th-top-h if th-h > 0 else 0
            left = (tw-w)//2 if tw-w > 0 else 0
            right = tw-left-w if tw-w > 0 else 0

            imgs[idx] = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    h, w = imgs[0].shape[0:2]
    if w == tw and h == th:
        return imgs

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis=1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)

        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs


def dist(a, b):
    return np.sqrt(np.sum((a-b)**2))


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area*(1-rate)/(peri+0.001)+0.5), max_shr)
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bboxes.append(shrinked_bbox)

    return np.array(shrinked_bboxes)


def process_data_np(image, label, bboxes):  # input one image, label for ignore or not and polys
    # FIXME the mine size ??
    img = random_scale(image, config['min_size'], config['ran_scale'])

    gt_text = np.zeros(img.shape[0:2], dtype='uint8')
    training_mask = np.ones(img.shape[0:2], dtype='uint8')

    if bboxes.shape[0] > 0:
        bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4), (bboxes.shape[0], int(bboxes.shape[1] / 2), 2)).astype(np.int32)
        # print(bboxes)
        # import ipdb;ipdb.set_trace()
        for i in range(bboxes.shape[0]):
            cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
            if not label[i]:
                cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

    gt_kernals = []
    for i in range(1, config['n']):
        rate = 1.0-(1.0-config['m'])/(config['n']-1)*i
        gt_kernal = np.zeros(img.shape[0:2], dtype='uint8')
        kernal_bboxes = shrink(bboxes, rate)
        for i in range(bboxes.shape[0]):
            cv2.drawContours(gt_kernal, [kernal_bboxes[i]], -1, 1, -1)
        gt_kernals.append(gt_kernal)

    # for threshord map
    gt_thresh = np.zeros(img.shape[0:2], dtype=np.float32)
    thresh_mask=np.zeros(img.shape[0:2],dtype=np.uint8)
    for i in range(bboxes.shape[0]):
        draw_border_map(bboxes[i],gt_thresh,thresh_mask) # use the smallest ratio as the expand ratio


    gt_thresh = gt_thresh * (0.7 - 0.3) + 0.3
    imgs = [img, gt_text, training_mask,gt_thresh,thresh_mask]
    imgs.extend(gt_kernals)

    imgs = random_horizontal_flip(imgs)
    imgs = random_rotate(imgs)
    imgs = random_crop(imgs, (640, 640))

    img, gt_text, training_mask, gt_thresh, thresh_mask, gt_kernals = imgs[0], imgs[1], imgs[2], imgs[3],imgs[4],imgs[5:]
    
    gt_text[gt_text > 0] = 1
    gt_kernals = np.array(gt_kernals)

    img = Image.fromarray(img)
    img=np.asarray(img)

    # cv2.imwrite('image.jpg',img[:,:,:])
    # cv2.imwrite('gt.jpg',gt_text*255)
    # cv2.imwrite('gt_thresh.jpg',gt_thresh[:,:]*255)
    # cv2.imwrite('thresh_mask.jpg',thresh_mask[:,:]*255)
    # cv2.imwrite('train_mask.jpg',training_mask[:,:]*255)
    # cv2.imwrite('kernel.jpg',gt_kernals[0,:,:]*255)
    # import ipdb;ipdb.set_trace()
    # gt_text: 完整的标签，全部标记为1，gt_kernels: 不同大小的kernel train_mask: 没有标签为0，不训练
    return img,gt_text,gt_kernals,training_mask,gt_thresh,thresh_mask 

def process_data_tf(image, label, polys, num_points, bboxes):
    # TODO: the images are normalized using the channel means and standard deviations
    image = tf.identity(image, 'input_image')

    img, gt_text, gt_kernals, training_mask, gt_thresh, thresh_mask= tf.py_func(process_data_np, [image, label, polys], [
        tf.uint8, tf.uint8, tf.uint8, tf.uint8,tf.float32,tf.uint8])

    # gt_kernals.set_shape([640,640,6])
    # training_mask.set_shape([640,640,1])
    img.set_shape([640,640,3])
    gt_text.set_shape([640,640])
    gt_kernals.set_shape([config['n']-1,640,640])
    training_mask.set_shape([640,640])
    gt_thresh.set_shape([640,640])
    thresh_mask.set_shape([640,640])

    img = tf.cast(img,tf.float32)
    gt_text = tf.cast(gt_text,tf.float32)
    gt_kernals = tf.cast(gt_kernals,tf.float32)
    training_mask = tf.cast(training_mask,tf.float32)
    gt_thresh=tf.cast(gt_thresh,tf.float32)
    thresh_mask=tf.cast(thresh_mask,tf.float32)

    img = tf.image.random_brightness(img, max_delta=32. / 255.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)

    img = tf_image_whitened(img, [123., 117., 104.])

    return img, gt_text, gt_kernals, training_mask, gt_thresh, thresh_mask


def process_td_np(image, label, bboxes):
    # generate mask
    height = image.shape[0]
    width = image.shape[1]
    patch_size = config['train_image_shape'][0]

    mask = np.zeros(image.shape[0:2], dtype='uint8')
    training_mask = np.ones(image.shape[0:2], dtype='uint8')
    if bboxes.shape[0] > 0:
        bboxes = np.reshape(bboxes * ([image.shape[1], image.shape[0]] * 4),
                            (bboxes.shape[0], int(bboxes.shape[1] / 2), 2)).astype(np.int32)
        for i in range(bboxes.shape[0]):
            cv2.drawContours(mask, [bboxes[i]], -1, i + 1, -1)
            if not label[i]:
                cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

    # random crop parameters
    loopTimes = 0
    MaxTimes = 100
    while True:
        # random parameters
        scale_h = np.random.uniform(0.05, 1)
        scale_w = np.random.uniform(0.05, 1)
        aspect_ratio = float(height)/width*scale_h/scale_w
        if aspect_ratio < 0.3 or aspect_ratio > 3:
            continue
        patch_h = int(height*scale_h)
        patch_w = int(width*scale_w)
        patch_h0 = np.random.randint(0, height-patch_h+1)
        patch_w0 = np.random.randint(0, width-patch_w+1)
        # compute overlap
        overlap_text = mask[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w] > 0
        overlap_text_count = np.sum(overlap_text)
        min_overlap_ratio = [0.01, 0.03, 0.05, 0.07]
        random_ratio = np.random.randint(0, 4)
        if overlap_text_count > patch_h*patch_w*min_overlap_ratio[random_ratio]:
            break
        loopTimes += 1
        if loopTimes >= MaxTimes:
            patch_h = height
            patch_w = width
            patch_h0 = 0
            patch_w0 = 0
            break

    # random crop & resize
    image = image.astype(np.float32)
    image = image[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
    image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

    mask = mask[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w]
    mask = cv2.resize(mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

    training_mask = training_mask[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w]
    training_mask = cv2.resize(training_mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

    # random rotate
    prob = np.random.uniform(0, 1)
    if prob <= 0.2:
        rtimes = 1
    elif prob >= 0.8:
        rtimes = 3
    else:
        rtimes = 0
    for rcount in range(rtimes):
        image = np.rot90(image)
        mask = np.rot90(mask)
        training_mask = np.rot90(training_mask)
    # cv2.imwrite('train_input/{}'.format(idx),image)

    # normalization
    # image = image.transpose((2,0,1))

    return image, mask, training_mask


def process_td_tf(image, label, polys, num_points, bboxes):
    # TODO: the images are normalized using the channel means and standard deviations
    image = tf.identity(image, 'input_image')

    img, gt_text, training_mask = tf.py_func(process_td_np, [image, label, polys], [
        tf.float32, tf.uint8, tf.uint8])

    # gt_kernals.set_shape([640,640,6])
    # training_mask.set_shape([640,640,1])
    img.set_shape([640, 640, 3])
    gt_text.set_shape([640, 640])
    training_mask.set_shape([640, 640])

    # img = tf.to_float(img)
    gt_text = tf.to_float(gt_text)
    training_mask = tf.to_float(training_mask)

    img = tf_image_whitened(img, [123., 117., 104.])

    return img, gt_text, training_mask


def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    """Resize an image and bounding boxes.
    """
    # Resize image.
    with tf.name_scope('resize_image'):
        height, width, channels = image.get_shape().as_list()
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image


def preprocess_for_eval(image, scale=1.0, out_shape=None, data_format='NHWC',
                        scope='preprocess_eval'):
    """Preprocess an image for evaluation.

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        out_shape: Output shape after pre-processing (if resize != None)
        resize: Resize strategy.

    Returns:
        A preprocessed image.
    """
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        image = tf.to_float(image)
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])

        if out_shape is None:
            i_shape = tf.to_float(tf.shape(image))
            shape = [tf.cast(i_shape[0]*scale, tf.int32), tf.cast(i_shape[1]*scale, tf.int32)]
            image = resize_image(image, shape,
                                 method=tf.image.ResizeMethod.BILINEAR,
                                 align_corners=False)
            image_shape = tf.shape(image)
            image_h, image_w = image_shape[0], image_shape[1]
            image_h = tf.cast(tf.rint(image_h/32)*32, tf.int32)
            image_w = tf.cast(tf.rint(image_w/32)*32, tf.int32)
            image = resize_image(
                image, [image_h, image_w], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        else:
            image = resize_image(image, out_shape,
                                 method=tf.image.ResizeMethod.BILINEAR,
                                 align_corners=False)

        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image

def draw_border_map(polygon, canvas, mask):
    shrink_ratio=config['m']
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon = np.array(padding.Execute(distance)[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(
        np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(
        np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros(
        (polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = calc_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = distance_map.min(axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid-ymin:ymax_valid-ymax+height,
            xmin_valid-xmin:xmax_valid-xmax+width],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

def calc_distance(xs, ys, point_1, point_2):
    '''
    compute the distance from point to a line
    ys: coordinates in the first axis
    xs: coordinates in the second axis
    point_1, point_2: (x, y), the end of the line
    '''
    height, width = xs.shape[:2]
    square_distance_1 = np.square(
        xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(
        xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(
        point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
        (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 *
                        square_sin / square_distance)

    result[cosin < 0] = np.sqrt(np.fmin(
        square_distance_1, square_distance_2))[cosin < 0]
    # self.extend_line(point_1, point_2, result)
    return result

def extend_line(point_1, point_2, result,shrink_ratio):
    ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + shrink_ratio))),
                    int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + shrink_ratio))))
    cv2.line(result, tuple(ex_point_1), tuple(point_1),
                4096.0, 1, lineType=cv2.LINE_AA, shift=0)
    ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + shrink_ratio))),
                    int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + shrink_ratio))))
    cv2.line(result, tuple(ex_point_2), tuple(point_2),
                4096.0, 1, lineType=cv2.LINE_AA, shift=0)
    return ex_point_1, ex_point_2
