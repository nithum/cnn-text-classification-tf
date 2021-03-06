#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1464282930/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filename", "b_test", "Name of file to run eval on: usually one of [b_test, r_test, br_test]")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data. Load your own data here
print("Loading data...")
x_test, y_test, vocabulary, vocabulary_inv = data_helpers.load_data(datfile = FLAGS.eval_filename)
y_test = np.argmax(y_test, axis=1)
print("Vocabulary size: {:d}".format(len(vocabulary)))
print("Test set size {:d}".format(len(y_test)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        softmax_scores = graph.get_operation_by_name("output/softmax_scores").outputs[0]

        # Generate batches for one epoch
        #batches = data_helpers.batch_iter(x_test, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        #all_scores = []
        #all_predictions = []

        #for x_test_batch in batches:
        predictions, scores = sess.run([predictions, softmax_scores], {input_x: x_test, dropout_keep_prob: 1.0})
        #print batch_scores[0]
        #batch_scores_pos = [score[1] for score in batch_scores]
        #all_scores = np.concatenate([all_scores, batch_scores_pos])
        #all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy
my_predictions = np.argmax(scores, axis = 1)
my_correct_predictions = float(sum(my_predictions == np.array(y_test)))
my_accuracy = my_correct_predictions/float(len(y_test))
correct_predictions = float(sum(predictions == y_test))
print("Total number of test examples: {}".format(len(y_test)))
print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
print("My Accuracy: {:g}".format(my_accuracy))

# Set/create directories
# Metrics summaries
#metric_summary_dir = os.path.abspath(os.path.join(FLAGS.checkpoint_dir, "eval"))
#if not os.path.exists(metric_summary_dir):
#    os.makedirs(metric_summary_dir)
#metric_summary_file = os.path.join(metric_summary_dir, FLAGS.eval_filename + "_metrics.txt")


# Print other metrics
#y_test = np.array(y_test)
precision = precision_score(y_test, my_predictions)
recall = recall_score(y_test, my_predictions)
f1 = f1_score(y_test, my_predictions)
roc_auc = roc_auc_score(y_test, np.array([score[1] for score in scores])) 
time_str = datetime.datetime.now().isoformat()
print("prec {:g}, recall {:g}, f1 {:g}, auc {:g}".format(precision, recall, f1, roc_auc))
"""
with open(metric_summary_file, 'w') as metric_file:
    metric_file.write("Accuracy: {:g} \n".format(accuracy))
    metric_file.write("prec {:g}, recall {:g}, f1 {:g}, auc {:g} \n".format(precision, recall, f1, roc_auc))
"""