#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 loop     Huazhong University of Science and Technology

import tensorflow as tf


from .net import resnet_v1
import configuration

config = configuration.TRAIN_CONFIG


def cal_dice_loss(pred, gt):
    union = tf.reduce_sum(tf.multiply(pred, gt), [1, 2])
    pred_square = tf.reduce_sum(tf.square(pred), [1, 2])
    gt_square = tf.reduce_sum(tf.square(gt), [1, 2])

    dice_loss = 1.-(2*union+1e-5)/(pred_square+gt_square+1e-5)

    # dice_loss=tf.Print(dice_loss,[gt_square],message='gt_square: ',summarize=5)
    return dice_loss


def online_hard_min(maps):
    pred_map, gt_map = maps

    # NOTE: OHM 3
    pos_mask = tf.cast(tf.equal(gt_map, 1.), dtype=tf.float32)     # [h,w,1]
    neg_mask = tf.cast(tf.equal(gt_map, 0.), dtype=tf.float32)
    n_pos = tf.reduce_sum((pos_mask), [0, 1])

    neg_val_all = tf.boolean_mask(pred_map, neg_mask)       # [N] # boolean mask, 取出mask对应的部分
    n_neg = tf.minimum(tf.shape(neg_val_all)[-1], tf.cast(n_pos*3, tf.int32))  # -1 代表对最完整的做OHM
    n_neg = tf.cond(tf.greater(n_pos, 0), lambda: n_neg, lambda: tf.shape(neg_val_all)[-1])
    neg_hard, neg_idxs = tf.nn.top_k(neg_val_all, k=n_neg)  # [batch_size,k][batch_size, k] 选择最大的那些，加上mask
    # TODO ERROR  slice index -1 of dimension 0 out of bounds.
    neg_min = tf.cond(tf.greater(tf.shape(neg_hard)[-1], 0), lambda: neg_hard[-1], lambda: 1.)      # [k]

    neg_hard_mask = tf.cast(tf.greater_equal(pred_map, neg_min), dtype=tf.float32)
    pred_ohm = pos_mask*pred_map+neg_hard_mask*neg_mask*pred_map
    return pred_ohm, gt_map


def calc_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


def calc_max_edge(mask):
    with tf.Session() as sess:
        mask = mask.eval()
    contour, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    max_dis = 0
    contour = contour.reshape(-1, 2)
    for i in range(contour.shape[0]):
        for j in range(contour.shape[0]):
            if i == j:
                continue
            dis = calc_distance(contour[i], contour[j])
            if dis > max_dis:
                max_dis = dis
    return max_dis


def calc_min_distance(mask1, mask2):
    with tf.Session() as sess:
        mask1 = mask1.eval()
        mask2 = mask2.eval()
    points1, points2 = [], []
    for y in range(mask1.shape[0]):
        for x in range(mask1.shape[1]):
            if mask1[y, x] == 1:
                points1.append([y, x])
            if mask2[y, x] == 1:
                points2.append([y, x])
    min_dis = np.inf
    for point1 in points1:
        for point2 in point2:
            dis = calc_distance(point1, point2)
            if dis < min_dis:
                min_dis = dis
    return min_dis


def calc_emb_loss_single(emb_pred_map, emb_gt_map):
    text_num = emb_gt_map.max()
    max_shape = max(emb_gt_map.get_shape().as_list())
    u_s = []
    l_var, l_dist = 0, 0
    eta, gamma = 0.5, 1.5
    for i in range(1, text_num+1):
        mask = (emb_gt_map == i)
        pred_map = emb_pred_map*mask
        u = (tf.reduce_sum(pred_map, [0, 1])/tf.reduce_sum(mask, [0, 1]))
        u_s.append(u)
        n = tf.reduce_sum(mask, [0, 1])
        w_scale = tf.exp(calc_max_edge(mask)/(2*max_shape))
        w_scale = tf.clip_by_value(w_scale, 1, 1.65)
        l_var += (tf.reduce_sum(tf.math.maximum(w_scale*tf.reduce_sum((pred_map-u).abs())-eta, 0))/n)

    for i in range(1, text_num+1):
        for j in range(1, text_num+1):
            if j == i:
                continue
            mask1 = (emb_gt_map == i)
            mask2 = (emb_gt_map == j)
            min_dis = calc_min_distance(mask1, mask2)
            w_dist = (1-20*tf.math.exp(-(4+min_dis/max_shape*10)))
            w_dist = tf.clip_by_value(w_dist, 0.63, 1)
            l_dist += tf.math.maximum(gamma-w_dist*tf.reduce_sum((u_s[i]-u_s[j]).abs()), 0)

    l_var = l_var/text_num
    l_dist = l_dist/(text_num*(text_num-1))
    return l_var, l_dist


def loss(pred_seg_maps, emb_pred_map, gt_map, kernels, training_mask):
    '''
    L = λLc + (1 − λ)Ls
    where Lc and Ls represent the losses for the complete text instances and the shrunk ones respec- tively, 
    and λ balances the importance between Lc and Ls
    It is common that the text instances usually occupy only an extremely small region in natural images,
    which makes the predictions of network bias to the non-text region, 
    when binary cross entropy [2] is used. Inspired by [20], 
    we adopt dice coefficient in our experiment. 
    The dice coefficient D(Si, Gi) is formulated as in Eqn
    '''
    with tf.name_scope('Loss'):
        n = config['n']
        # for complete loss
        pred_text_map = pred_seg_maps[:, 0, :, :]

        # NOTE: the mask is pred_map, may try gt_map?
        mask = tf.to_float(tf.greater(pred_text_map*training_mask, 0.5))
        pred_text_map = pred_text_map*training_mask

        emb_gt_map = tf.identity(gt_map)
        gt_map[gt_map > 0] = 1
        gt_map = gt_map*training_mask

        if config['OHM']:
            pred_maps, gt_maps = tf.map_fn(online_hard_min, (pred_text_map, gt_map))
        else:
            pred_maps, gt_maps = pred_text_map, gt_map
        ohm_dice_loss = cal_dice_loss(pred_maps, gt_maps)

        dice_loss = tf.reduce_mean(ohm_dice_loss)
        tf.add_to_collection('losses', 0.7*dice_loss)

        for i in range(config['n']-1):
            # for shrink loss
            pred_map = pred_seg_maps[:, i+1, :, :]
            gt_map = kernels[:, i, :, :]

            pred_map = pred_map*mask
            gt_map = gt_map*mask

            dice_loss = cal_dice_loss(pred_map, gt_map)
            dice_loss = tf.reduce_mean(dice_loss)
            # NOTE the paper is divide Ls by (n-1), I don't divide this for long time
            tf.add_to_collection('losses', (1-0.7)*dice_loss/(n-1))

        # loss for embedding maps
        loss_var, loss_dist = tf.map_fn(calc_emb_loss_single, [emb_pred_map, emb_gt_map])
        loss_var = tf.reduce_mean(loss_var)
        loss_dist = tf.reduce_mean(loss_dist)
        tf.add_to_collection('losses', loss_var+loss_dist)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# TODO: how to test this FPN net work prople
if __name__ == '__main__':
    # test this unit
    import numpy as np

    test_input = tf.Variable(initial_value=tf.ones(
        (5, 224, 224, 3), tf.float32))
    output, f = model(test_input)

    init_op = tf.global_variables_initializer()

    restore = slim.assign_from_checkpoint_fn(
        "resnet_v1_50.ckpt", slim.get_trainable_variables(), ignore_missing_vars=True)
    with tf.Session() as sess:
        sess.run(init_op)
        restore(sess)
        out, f_res = sess.run([output, f])
        print(np.sum(f_res[0]))
