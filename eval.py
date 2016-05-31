#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from ngram import get_binary_classifier_metrics

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filename", "b_test", "Name of file to run eval on: usually one of [b_test, r_test, br_test]")
tf.flags.DEFINE_string("checkpoint_num", None, "Number of the checkpoint to use")
tf.flags.DEFINE_boolean("char_cnn", False, "Train on the character level instead of word level (default: False)")

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

if FLAGS.char_cnn:
    x_test, y_test, vocabulary, vocabulary_inv = data_helpers.load_eval_data_char(datfile = FLAGS.eval_filename)
else:
    x_test, y_test, vocabulary, vocabulary_inv = data_helpers.load_eval_data(datfile = FLAGS.eval_filename)
y_test = np.argmax(y_test, axis=1)
print("Vocabulary size: {:d}".format(len(vocabulary)))
print("Test set size {:d}".format(len(y_test)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
if FLAGS.checkpoint_num is None:
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
else:
    checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'model-' + FLAGS.checkpoint_num)
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
        softmax_scores = graph.get_operation_by_name("output/softmax_scores").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(x_test, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_scores = np.array([])

        for x_test_batch in batches:
            batch_scores = sess.run(softmax_scores, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            batch_scores_pos = [score[1] for score in batch_scores]
            all_scores = np.concatenate([all_scores, batch_scores_pos])

# Print accuracy
predictions = [score >= 0.5 for score in all_scores]
correct_predictions = float(sum(predictions == y_test))
accuracy = correct_predictions/float(len(y_test))
print("@ threshold = 0.5")
print("Total number of test examples: {}".format(len(y_test)))
print("Accuracy: {:g}".format(accuracy))
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, all_scores) 
time_str = datetime.datetime.now().isoformat()
print("prec {:g}, recall {:g}, f1 {:g}, auc {:g}".format(precision, recall, f1, roc_auc))
print (" ")
print('@ optimal F1 threshold')
threshold, scores = get_binary_classifier_metrics(all_scores, y_test)

# Set/create directories
# Metrics summaries
metric_summary_dir = os.path.abspath(os.path.join(FLAGS.checkpoint_dir, "eval"))
if not os.path.exists(metric_summary_dir):
    os.makedirs(metric_summary_dir)
metric_summary_file = os.path.join(metric_summary_dir, FLAGS.eval_filename + "_metrics.txt")


# Print other metrics
# TODO: Optimal F1 score
with open(metric_summary_file, 'w') as metric_file:
    metric_file.write("Accuracy: {:g} \n".format(accuracy))
    metric_file.write("@ threshold = 0.5 \n")
    metric_file.write("prec {:g}, recall {:g}, f1 {:g}, auc {:g} \n".format(precision, recall, f1, roc_auc))
    metric_file.write("@ optimal f1 threshold: {:g} \n".format(threshold))
    metric_file.write("prec {:g}, recall {:g}, f1 {:g}, auc {:g} \n".format(scores['precision @ optimal F1'], 
                                                                    scores['recall @ optimal F1'], scores['optimal F1'],
                                                                    scores['roc']))

