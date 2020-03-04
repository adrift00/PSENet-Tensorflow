#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 loop     Huazhong University of Science and Technology

import tensorflow as tf


from .net import resnet_v1
import configuration

config = configuration.TRAIN_CONFIG


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


def cal_dice_loss(pred, gt):
    union = tf.reduce_sum(tf.multiply(pred, gt), [1, 2])
    pred_square = tf.reduce_sum(tf.square(pred), [1, 2])
    gt_square = tf.reduce_sum(tf.square(gt), [1, 2])
    dice_loss = 1.-(2*union+1e-5)/(pred_square+gt_square+1e-5)

    # dice_loss=tf.Print(dice_loss,[gt_square],message='gt_square: ',summarize=5)
    return dice_loss


def calc_l1_loss(pred, gt, mask):
    mask_sum = tf.reduce_sum(mask)
    # mask_sum=tf.Print(mask_sum,['mask sum',mask_sum])

    def f():
        diff = tf.abs(pred-gt)
        loss = tf.reduce_sum(diff*mask)
        loss = loss/mask_sum
        return loss
    loss = tf.cond(tf.equal(mask_sum, 0), lambda: mask_sum, f)
    return loss
    # loss = tf.abs(pred - gt)
    # positive = loss * mask
    # negative = loss * (1 - mask)
    # positive_count = tf.reduce_sum(mask)
    # negative_count = tf.math.minimum(
    #         tf.reduce_sum((1 - mask)),
    #         (positive_count * 3))
    # negative_loss, _ = tf.nn.top_k(tf.reshape(negative,[-1]), tf.cast(negative_count,tf.int32))
    # negative_loss = tf.reduce_sum(negative_loss) / negative_count
    # positive_loss = tf.reduce_sum(positive) / positive_count
    # positive_loss=tf.Print(positive_loss,['positive: ',positive_loss])
    # negative_loss=tf.Print(negative_loss,['negative: ',negative_loss])
    # return positive_loss + negative_loss


def calc_BCE_loss(pred, gt):
    eps = 1e-7
    positive_mask = gt
    negative_mask = (1-gt)
    positive_num = tf.cast(tf.reduce_sum(positive_mask), tf.int32)
    negative_num = tf.math.minimum(tf.cast(tf.reduce_sum(negative_mask), tf.int32), tf.cast(positive_num*3, tf.int32))
    loss = -gt*tf.math.log(tf.clip_by_value(pred, eps, 1))-(1-gt)*tf.math.log(tf.clip_by_value(1-pred, eps, 1))
    positive_loss = loss*positive_mask
    negative_loss = loss*negative_mask
    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, [-1]), negative_num)
    # positive_loss=tf.Print(positive_loss,['positive_loss',positive_loss])
    # negative_loss=tf.Print(negative_loss,['negative_loss',negative_loss])
    balance_loss = (tf.reduce_sum(positive_loss)+tf.reduce_sum(negative_loss)) / \
        (tf.cast(positive_num+negative_num, tf.float32)+eps)
    return balance_loss


def loss(pred_seg_maps, gt_map, kernels, training_mask):
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

        n = len(config['rate'])+1
        # for complete loss
        pred_text_map = pred_seg_maps[:, 0, :, :]

        # NOTE: the mask is pred_map, may try gt_map?
        mask = tf.cast(tf.greater(pred_text_map*training_mask, 0.5), tf.float32)
        pred_text_map = pred_text_map*training_mask
        gt_map = gt_map*training_mask

        if config['OHM']:
            pred_maps, gt_maps = tf.map_fn(online_hard_min, (pred_text_map, gt_map))
        else:
            pred_maps, gt_maps = pred_text_map, gt_map
        ohm_dice_loss = cal_dice_loss(pred_maps, gt_maps)

        dice_loss = tf.reduce_mean(ohm_dice_loss)
        tf.compat.v1.add_to_collection('losses', 0.7*dice_loss)

        for i, _ in enumerate(config['rate']):
            # for shrink loss
            pred_map = pred_seg_maps[:, i+1, :, :]
            gt_map = kernels[:, i, :, :]

            pred_map = pred_map*mask
            gt_map = gt_map*mask

            dice_loss = cal_dice_loss(pred_map, gt_map)
            dice_loss = tf.reduce_mean(dice_loss)
            # NOTE the paper is divide Ls by (n-1), I don't divide this for long time
            tf.compat.v1.add_to_collection('losses', (1-0.7)*dice_loss/(n-1))

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.compat.v1.get_collection('losses'), name='total_loss')


def loss_with_thresh(pred_seg_maps, gt_map, kernels, training_mask, thresh_map, binary_map, gt_thresh, thresh_mask):
    '''
    L = λLc + (1 − λ)Ls
    where Lc and Ls represent the losses for the complete text instances and the shrunk ones respec- tively, 
    and λ balances the importance between Lc and Ls
    It is common that the text instances usually occupy only an extremely small region in natural images,
    which makes the predictions of network bias to the non-text region, 
    when binary cross entropy [2] is used. Inspired by [20], 
    we adopt dice coefficient in our experiment. 
    The dice coefficient D(Si, Gi) is formulated as in Eqn

    gt_map: the complete kernel map
    kernels: the gt shrink kernel maps 
    pred_seg_maps: the pred maps, index 0 is the complete, others is shrinked
    '''
    with tf.name_scope('Loss'):

        n = len(config['rate'])+1
        # for complete loss
        pred_text_map = pred_seg_maps[:, 0, :, :]

        # NOTE: the mask is pred_map, may try gt_map?
        mask = tf.cast(tf.greater(pred_text_map*training_mask, 0.5), tf.float32)
        pred_text_map = pred_text_map*training_mask
        gt_map = gt_map*training_mask
        # TODO: the kernels don't times the training_mask, maybe is a bug

        # if config['OHM']:
        #     pred_maps, gt_maps = tf.map_fn(online_hard_min,
        #                                    (pred_text_map, gt_map))  # pred_text_map(n,h,w) gt_map(n,h,w)
        # else:
        #     pred_maps, gt_maps = pred_text_map, gt_map
        # dice_loss = cal_dice_loss(pred_maps, gt_maps)
        # dice_loss = tf.reduce_mean(dice_loss)
        # # dice_loss=tf.Print(dice_loss,['comp_loss',dice_loss])
        # tf.compat.v1.add_to_collection('losses', config['complete_weight']*dice_loss)

        for i, _ in enumerate(config['rate']):
            # for shrink loss
            pred_map = pred_seg_maps[:, i+1, :, :]*training_mask
            gt_map = kernels[:, i, :, :]*training_mask

            # bce loss
            # bce_loss=calc_BCE_loss(pred_map,gt_map)
            # # bce_loss=tf.Print(bce_loss,['bce_loss',bce_loss])
            # tf.compat.v1.add_to_collection('losses',config['shrink_weight']*bce_loss)

            # dice_loss
            if config['OHM']:
                pred_map, gt_map = tf.map_fn(online_hard_min, (pred_map, gt_map))
            else:
                pred_map = pred_map*mask
                gt_map = gt_map*mask

            dice_loss = cal_dice_loss(pred_map, gt_map)
            dice_loss = tf.reduce_mean(dice_loss)
            # dice_loss=tf.Print(dice_loss,['shrink_loss',dice_loss])
            tf.compat.v1.add_to_collection('losses', config['shrink_weight']*dice_loss/(n-1))

        # add dice loss for binary map and l1 loss for thresh map
        # thresh loss
        thresh_map = thresh_map*training_mask
        gt_thresh = gt_thresh*training_mask
        thresh_loss = calc_l1_loss(thresh_map, gt_thresh, thresh_mask)
        # thresh_loss=tf.Print(thresh_loss,['thresh_loss',thresh_loss])
        tf.compat.v1.add_to_collection('losses', config['thresh_weight']*thresh_loss)

        # binary loss
        binary_map = binary_map*training_mask
        gt_map = kernels[:, -1, :, :]*training_mask  # the last kernels is the smallest
        if config['OHM']:
            binary_map, gt_map = tf.map_fn(online_hard_min, (binary_map, gt_map))
        else:
            binary_map, gt_map = binary_map*mask, gt_map*mask
        binary_loss = cal_dice_loss(binary_map, gt_map)
        binary_loss = tf.reduce_mean(binary_loss)
        # binary_loss=tf.Print(binary_loss,['binary_loss',binary_loss])
        tf.compat.v1.add_to_collection('losses', config['binary_weight']*binary_loss)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.compat.v1.get_collection('losses'), name='total_loss')


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
