#!/usr/bin/python3

import os
import argparse
import numpy as np
import pandas as pd
from tkinter import _flatten
from Bio import SeqIO
import func as f
import preprocessing as pp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf


# Allocate gpu memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)


def predict_all(input_seqs, output_path, thr, thread, skip_gene_scan, skip_blast):
    """
    Predict all input seqs

    :param input_seqs: the fasta file of the input contigs
    :param output_path: the output file name ending in ".csv"
    :param thr: the threshold for prediction
    :param thread: the number of CPU cores to be used for prediction
    :param skip_gene_scan: skip the step of gene scan
    :param skip_blast: skip the step of BLAST
    """
    ids = [[], [], [], []]
    lengths = [[], [], [], []]
    records_list = [[], [], [], []]
    ratios_list = [[], [], [], []]

    # scan genes
    if not skip_gene_scan:
        print("Running FragGeneScan...")
        pp.scan_gene(input_seqs, thread)
    gene_file = f.remove_ext(input_seqs) + "_fragscan" + "-w.out"

    # blast
    blast_out = f.remove_ext(input_seqs) + ".blastp"
    if not skip_blast:
        print("Running BLAST...")
        query = f.remove_ext(gene_file)+".faa"
        pp.run_blastp(query, blast_out, thread)

    print("Start predicting...")

    # get gene hit ratios
    gene_list = pp.read_gene(gene_file)
    gene_ratios = pp.get_gene_feature(gene_list, blast_out)

    # distribute seqs
    for index, record in enumerate(SeqIO.parse(input_seqs, "fasta")):
        seq_len = len(record)
        if 100 <= seq_len < 500:
            ids[0] += [record.id]
            lengths[0] += [seq_len]
            ratios_list[0] += [gene_ratios[index]]
            records_list[0] += [record]
        elif 500 <= seq_len < 1000:
            ids[1] += [record.id]
            lengths[1] += [seq_len]
            ratios_list[1] += [gene_ratios[index]]
            records_list[1] += [record]
        elif 1000 <= seq_len < 1500:
            ids[2] += [record.id]
            lengths[2] += [seq_len]
            ratios_list[2] += [gene_ratios[index]]
            records_list[2] += [record]
        elif seq_len >= 1500:
            ids[3] += [record.id]
            lengths[3] += [seq_len]
            ratios_list[3] += [gene_ratios[index]]
            records_list[3] += [record]

    # do prediction
    prob_list = []
    label_list = []
    for index, records in enumerate(records_list):
        if records != []:
            if index < 3:
                max_len = 500*(index+1)
                code_list = pp.encode_fasta_short(records, max_len, gene_file)
                cnn_feature = code_list
            else:
                code_list = pp.encode_fasta_long(records, gene_file)
                cnn_feature = pp.get_average_output(code_list, cnn_models)
            gene_feature = np.array(ratios_list[index]).reshape(-1, 1)
            probs = models[index].predict([cnn_feature, gene_feature])[:, -1]
            labels = (probs >= thr)
            prob_list += probs.tolist()
            label_list += labels.tolist()
    res = get_result(list(_flatten(ids)), list(_flatten(lengths)), prob_list, label_list)
    out_dir, out_file = os.path.split(output_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    res.to_csv(f.remove_ext(output_path)+'.csv', index=False)


def get_result(ids, lengths, probs, labels):
    """
    Return the final result

    :param ids: the id list for the id column
    :param lengths: the length list for the length column
    :param probs: the probability list for the score column
    :param labels: the label list for the label column
    """
    res = pd.DataFrame(ids, columns=["ID"])
    res["Length"] = lengths
    res["Phage"] = labels
    res["Score"] = probs
    return res


PARSER = argparse.ArgumentParser(description='Identify phage contigs from a given metagenome.')
PARSER.add_argument("-i", type=str, action="store", dest="input_seqs",
                    help="The file path of contigs to be predicted.", required=True)
PARSER.add_argument("-t", type=float, action="store", dest="thr", default=0.5,
                    help="The threshold for prediction.", required=False)
PARSER.add_argument("-s", type=str, action="store", dest="out_file",
                    help="The file path for saving prediction result.", required=False)
PARSER.add_argument("-c", type=int, action="store", dest="thread", default=4,
                    help="The number of CPU cores to be used for prediction.", required=False)
PARSER.add_argument("--G", action='store_true', dest="no_gene_scan", default=False,
                    help="Skip gene scan.", required=False)
PARSER.add_argument("--B", action='store_true', dest="no_blast", default=False,
                    help="Skip BLAST.", required=False)

args = PARSER.parse_args()

print("Loading models...")
model_paths = ["models/0.5k_concat.h5",
               "models/1k_concat.h5",
               "models/1.5k_concat.h5",
               "models/10k_concat.h5"]
models = []
for path in model_paths:
    models += [load_model(path)]

cnn_model_paths = ["models/0.5k_cnn.h5",
                   "models/1k_cnn.h5",
                   "models/1.5k_cnn.h5"]
cnn_models = []
for path in cnn_model_paths:
    cnn_models += [load_model(path)]


out_file_None = False
if args.out_file is None:
    args.out_file = f.remove_ext(args.input_seqs) + "_virSearcher.csv"
    out_file_None = True
predict_all(os.path.realpath(args.input_seqs), os.path.realpath(args.out_file), args.thr, args.thread, args.no_gene_scan, args.no_blast)

if out_file_None:
    print("Done, and the result has been saved in the same folder as the input file.")
    print("Thank you for using virSearcher.")
else:
    print("Done. Thank you for using virSearcher.")


