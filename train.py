#! /usr/bin/env python

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cross_validation import train_test_split

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
#tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#tf.flags.DEFINE_float("max_length", 500, "Maximum length of a sentence (Default: 500)")
tf.flags.DEFINE_string("training_file", "b_train", "Name of the training data file (default: b_train)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_training_data(datfile = FLAGS.training_file)

# Split train/test set
x_train, x_dev = train_test_split(x, test_size = 0.1, random_state=0)
y_train, y_dev = train_test_split(y, test_size = 0.1, random_state=0)
x_dev.to_csv('data/x_dev.csv')
y_dev.to_csv('data/y_dev.csv')

print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Fraction positive examples (train): {:d}/{:d}").format( sum(np.argmax(y_train,1)), len(y_train) )
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Keep track of best auc to save best model
best_auc = 0


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=2,
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)
        # TODO: figure out how to write precision summary
        #prec_summary = tf.scalar_summary("precision", cnn.precision) and search for acc_summary below

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)
        
        # Metrics summaries
        metric_summary_dir = os.path.abspath(os.path.join(out_dir, "metrics"))
        if not os.path.exists(metric_summary_dir):
            os.makedirs(metric_summary_dir)
        metric_summary_file = os.path.join(metric_summary_dir, "dev_metrics.txt")

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, scores = sess.run( 
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.softmax_scores], 
                feed_dict)
            predictions = np.argmax(scores,1)
            precision = precision_score(np.argmax(y_batch, 1), predictions)
            recall = recall_score(np.argmax(y_batch, 1), predictions)
            f1 = f1_score(np.argmax(y_batch, 1), predictions)
            pos_prob = np.array([score[1] for score in scores])
            y_batch_pos = np.array([y[1] for y in y_batch])
            roc_auc = roc_auc_score(y_batch_pos, pos_prob)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print("prec {:g}, recall {:g}, f1 {:g}, auc {:g}".format(precision, recall, f1, roc_auc))
            with open(metric_summary_file, 'a') as metric_file:
                metric_file.write("{}: step {}, loss {:g}, acc {:g} \n".format(time_str, step, loss, accuracy))
                metric_file.write("prec {:g}, recall {:g}, f1 {:g}, auc {:g} \n".format(precision, recall, f1, roc_auc))
            if writer:
                writer.add_summary(summaries, step)

            return roc_auc
            
            

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                roc_auc = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
                # Note: Only saves models that improve best AUC!
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            #if current_step % FLAGS.checkpoint_every == 0:
            #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #    print("Saved model checkpoint to {}\n".format(path))
