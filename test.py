#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import text_handling
from cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_file",  "D:/Projects/file_exmple.txt", "file for test .")
tf.flags.DEFINE_string("new_file", "D:/Projects/test.txt", "new file for test.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "D:/Projects/CNNForFakePapersDetection/runs/1547237271/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("vocab_file", "D:/Projects/CNNForFakePapersDetection/runs/1547237271/vocab", "vocabulary file from training run")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nTraining run parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#  Load data from user
def load_user_file(user_file) :
    text_file_output = text_handling.ConvertPdfFilesToText(user_file)
    return user_file

# Map data into vocabulary
#vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_file)
#x_test = np.array(list(vocab_processor.transform(x_raw)))

file_for_test = load_user_file(FLAGS.data_file)
text_handling.prepare_for_test(file_for_test,FLAGS.new_file)
x_raw = text_handling.load_data(FLAGS.new_file)
x_raw = x_raw[0]
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_file)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nTesting ...\n")

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
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = text_handling.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            #all_predictions = np.concatenate([all_predictions, batch_predictions])



# Save the evaluation to a csv
"""
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
"""
