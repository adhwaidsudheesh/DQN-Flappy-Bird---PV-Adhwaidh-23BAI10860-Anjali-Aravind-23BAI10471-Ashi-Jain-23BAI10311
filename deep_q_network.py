#!/usr/bin/env python
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
import os
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import pygame
from pygame.locals import * 

# --- QUICK RUN CONFIGURATION ---
GAME = 'bird' 
ACTIONS = 2 
GAMMA = 0.99 
OBSERVE = 1000.          # Starts training almost immediately (after 100 frames)
MAX_STEPS = 100000        # The program will stop automatically after 100000 steps
EXPLORE = 2000000. 
FINAL_EPSILON = 0.0001 
INITIAL_EPSILON = 0.1   
REPLAY_MEMORY = 50000 
BATCH = 32 
FRAME_PER_ACTION = 1
MANUAL_CONTROL = True # Set to True to play manually

# Automatic folder creation
if not os.path.exists("logs_bird"):
    os.makedirs("logs_bird")
if not os.path.exists("saved_networks"):
    os.makedirs("saved_networks")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    s = tf.placeholder("float", [None, 80, 80, 4])
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game_state = game.GameState()
    D = deque()

    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)

    epsilon = INITIAL_EPSILON
    t = 0
    # Stop when t hits MAX_STEPS
    while t < MAX_STEPS:
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0

        # Handle manual control
        if MANUAL_CONTROL:
            # Handle quit event
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Check for space or up key using get_pressed 
            # for more immediate response
            keys = pygame.key.get_pressed()
            if keys[K_SPACE] or keys[K_UP]:
                print("----------Manual Flap!----------")
                action_index = 1
            a_t[action_index] = 1
        # Handle AI control
        elif t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal_batch = minibatch[i][4]
                if terminal_batch:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            train_step.run(feed_dict = {y : y_batch, a : a_batch, s : s_j_batch})

        s_t = s_t1
        t += 1

        # Save on the very last step
        if t == MAX_STEPS:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        state = "observe" if t <= OBSERVE else "explore"
        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", format(epsilon, '.4f'), "/ ACTION", action_index, "/ Q_MAX %e" % np.max(readout_t))

    print("Target reached (", MAX_STEPS, " steps). Closing program.")
    a_file.close()
    h_file.close()
    sys.exit() # Properly close

def main():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

if __name__ == "__main__":
    main()