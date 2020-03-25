#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 loop     Huazhong University of Science and Technology
import sys
import cv2
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
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        mask = sess.run(mask)
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
        mask1 = sess.run(mask1)
        mask2 = sess.run(mask2)
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


def calc_emb_loss_single(maps):
    emb_pred_map, emb_gt_map = maps
    text_num = tf.cast(tf.reduce_max(emb_gt_map), tf.int32)
    # text_num = tf.Print(text_num, ['text_num', text_num])
    l_var, l_dist = 0., 0.

    def func2():
        l_var, l_dist = 0., 0.
        # max_shape = max(emb_gt_map.get_shape().as_list())
        u_s = tf.zeros([text_num, 8])
        eta, gamma = 0.5, 1.5

        def cond(i, n, l_var, u_s):
            return tf.less(i, n)

        def body(i, n, l_var, u_s):
            idx = tf.cast(i, tf.float32)
            # idx = tf.Print(idx, ['idx', idx])
            mask = tf.cast(tf.equal(emb_gt_map, idx), tf.float32)
            pred_map = emb_pred_map*tf.expand_dims(mask, axis=-1)
            # pred_map = tf.Print(pred_map, ['pred_map_max', tf.reduce_max(pred_map)])
            num = tf.reduce_sum(mask)
            pred_sum = tf.reduce_sum(pred_map, [0, 1])
            # num = tf.Print(num, ['mask_sum', num])
            # pred_sum = tf.Print(pred_sum, ['pred_sum', pred_sum], summarize=8)
            # u=pred_sum/num
            u = tf.cond(tf.equal(num, 0.),
                        lambda: tf.zeros(8),
                        lambda: pred_sum/num)
            # u = tf.Print(u, ['u', u], summarize=8)
            u_s = tf.concat(values=[u_s[0:i], [u], u_s[i+1:]], axis=0)
            # calc_max_edge(mask)
            # w_scale = tf.exp(calc_max_edge(mask)/(2*max_shape))
            # w_scale = tf.clip_by_value(w_scale, 1, 1.65)
            w_scale = 1
            # num = tf.Print(num, ['mask_num', num])
            # import ipdb;ipdb.set_trace()
            tmp = tf.cond(tf.equal(num, 0.),
                          lambda: tf.constant(0.),
                          lambda: (tf.reduce_sum(
                                    tf.math.maximum(
                                    w_scale*tf.reduce_sum(tf.abs((pred_map-u)*tf.expand_dims(mask, axis=-1)), axis=-1)-eta, 0))/num))
            # tmp = tf.Print(tmp, ['every_loss_var: ', tmp])
            l_var += tmp
            i = i+1
            return i, n, l_var, u_s

        _, _, l_var, u_s = tf.while_loop(cond, body, [1, text_num+1, l_var, u_s])

        def cond1(i, n, l_dist):
            return tf.less(i, n)

        def body1(i, n, l_dist):
            def cond2(i, j, n, dist):
                return tf.less(j, n)

            def body2(i, j, n, l_dist):
                w_dist = 1
                l_dist += tf.math.maximum(gamma-w_dist*tf.reduce_sum(tf.abs(u_s[i]-u_s[j])), 0)
                j = tf.add(j, 1)
                return i, j, n, l_dist

            _, _, _, l_dist = tf.while_loop(cond2, body2, [i, 1, text_num+1, l_dist])
            i = tf.add(i, 1)
            return i, n, l_dist

        _, _, l_dist = tf.while_loop(cond1, body1, [1, text_num+1, l_dist])

        l_var = l_var/(tf.cast(text_num, tf.float32))
        l_dist = tf.cond(tf.equal(text_num-1, 0),
                         lambda: tf.constant(0.),
                         lambda: l_dist/(tf.cast((text_num*(text_num-1)), tf.float32)))
        # l_var = tf.Print(l_var, ['loss_var: ', l_var])
        # l_dist = tf.Print(l_dist, ['loss_dist: ', l_dist])
        return l_var, l_dist
    l_var, l_dist = tf.cond(tf.equal(text_num, 0), lambda: (l_var, l_dist), func2)
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
        emb_pred_map = emb_pred_map*tf.expand_dims(training_mask, axis=-1)

        gt_map = gt_map*training_mask
        emb_gt_map = tf.identity(gt_map)
        # change the gt_map tensor when it >0, assign 1 to it
        one_tensor = tf.ones_like(gt_map)
        zero_tensor = tf.zeros_like(gt_map)
        gt_map = tf.where(tf.greater(gt_map, 0.), one_tensor, zero_tensor)

        if config['OHM']:
            pred_maps, gt_maps = tf.map_fn(online_hard_min, (pred_text_map, gt_map))
        else:
            pred_maps, gt_maps = pred_text_map, gt_map
        ohm_dice_loss = cal_dice_loss(pred_maps, gt_maps)

        dice_loss = tf.reduce_mean(ohm_dice_loss)
        tf.add_to_collection('losses', 0.5*dice_loss)

        for i in range(config['n']-1):
            # for shrink loss
            pred_map = pred_seg_maps[:, i+1, :, :]
            gt_map = kernels[:, i, :, :]

            pred_map = pred_map*mask
            gt_map = gt_map*mask

            dice_loss = cal_dice_loss(pred_map, gt_map)
            dice_loss = tf.reduce_mean(dice_loss)
            # NOTE the paper is divide Ls by (n-1), I don't divide this for long time
            tf.add_to_collection('losses', (1-0.5)*dice_loss/(n-1))

        # loss for embedding maps
        loss_var, loss_dist = tf.map_fn(calc_emb_loss_single, (emb_pred_map, emb_gt_map))
        loss_var = tf.reduce_mean(loss_var)
        loss_dist = tf.reduce_mean(loss_dist)
        # loss_var = tf.Print(loss_var, ['loss_var: ', loss_var])
        # loss_dist = tf.Print(loss_dist, ['loss_dist: ', loss_dist])
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
