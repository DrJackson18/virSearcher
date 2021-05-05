#!/usr/bin/python3

import os
import re
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.preprocessing.sequence import pad_sequences
import func as f


GENE_K = 3
NON_GENE_K = 1

ACCUM_NUM = 5

ARRAY_TYPE = 'int8'

# Allocate gpu memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

letters = ['A', 'C', 'G', 'T']


def scan_gene(fa_file, thread):
    """
    Scan genes with FragGeneScan

    :param fa_file: the fasta file to be scanned
    :param thread: the number of CPU cores to be used
    """
    program_dir = "./FragGeneScan1.31/"
    out_file = f.remove_ext(fa_file)+"_fragscan"
    program_path = os.path.join(program_dir, "FragGeneScan")
    cmd = program_path+" -s "+fa_file+" -o "+out_file+"-w 1 -t complete -p "+str(thread)
    f.run_proc(cmd)
    gene_file = out_file+"-w.out"
    return gene_file


def run_blastp(query, out, thread):
    """
    Run BLASTp for aligning the query genes to the virus gene database

    :param query: the file with the query genes
    :param thread: the number of CPU cores to be used
    """
    blast_dir = "./database"
    os.chdir(blast_dir)
    cmd = "blastp -query "+query+" -out "+out+" -db virusPDB -outfmt 6 -evalue 1e-5 -num_threads "+str(thread)+" -max_target_seqs 1"
    f.run_proc(cmd)


def read_gene(scan_file):
    """
    Read gene information from the result of FragGeneScan

    :param scan_file: the output file of FragGeneScan
    """
    gene_list = []
    with open(scan_file) as rf:
        line = rf.readline()
        gene = None
        while line:
            if line.startswith('>'):
                if gene is not None:
                    gene_list += [gene]
                gene = {}
                gene['id'] = line[1:len(line)-1]
                gene['infor'] = []
            else:
                re_str = re.findall(r"^(\d+)\s+(\d+)\s+([+|-])", line)[0]
                infor = {}
                infor['strand'] = re_str[2]
                infor['start'] = re_str[0]
                infor['end'] = re_str[1]
                gene['infor'] += [infor]
            line = rf.readline()
        gene_list += [gene]
    return gene_list


def get_word2num_dict(k):
    """
    Construct a word2num dictionary according to the word length k

    :param k: the given word length
    """
    word2num_dict = {}
    letter_code = {'A':0, 'C':1, 'G':2, 'T':3}
    kmer_list = all_kmer(k)
    for kmer in kmer_list:
        num = 0
        for i in range(k):
            num += letter_code[kmer[-i-1]]*int(math.pow(4, i))
        word2num_dict[kmer] = num+1
    word2num_dict['N'*k] = int(math.pow(4, k))+1  # NNN
    return word2num_dict


def all_kmer(k):
    """
    Get all the k mers

    :param k: the given length of a word
    """
    if k == 1:
        return letters
    res = []
    sub_list = all_kmer(k-1)
    for l in letters:
        for sub in sub_list:
            res += [l+sub]
    return res


def encode_fasta_short(record_list, max_len, gene_file, saved_path=None):
    """
    Encode a list of records storing the information of contigs with length <=1.5kbp

    :param record_list: the list of the records
    :param max_len: the max length that equal to 0.5kbp, 1kbp or 1.5kbp
    :param gene_file: the output file of FragGeneScan
    :param saved_path: the file for saving the final code
    """
    code_list = []
    word2num_dict = [get_word2num_dict(NON_GENE_K)]  # 1 mer
    word2num_dict += [get_word2num_dict(GENE_K)]  # 3 mer
    gene_list = read_gene(gene_file)

    for i, record in enumerate(record_list):
        fw_code = encode_seq(record.seq, gene_list[i], word2num_dict, strand='+')
        bw_code = encode_seq(record.seq.reverse_complement(), gene_list[i], word2num_dict, strand='-')
        code_list += [fw_code+bw_code]
    max_len = max_len * 2
    code_array = pad_sequences(code_list, maxlen=max_len, dtype=ARRAY_TYPE, padding='post')
    if saved_path is not None:
        np.save(saved_path, code_array)
    return code_array


def encode_fasta_long(record_list, gene_file, saved_path=None):
    """
    Encode a list of records storing the information of contigs with length >1.5kbp

    :param record_list: the list of the records
    :param gene_file: the output file of FragGeneScan
    :param saved_path: the file for saving the final code
    """
    code_list = []
    word2num_dict = [get_word2num_dict(NON_GENE_K)]  # 1
    word2num_dict += [get_word2num_dict(GENE_K)]  # 3
    gene_list = read_gene(gene_file)
    for i, record in enumerate(record_list):
        curr_code = []
        left_seq = record.seq
        while len(left_seq) > 1500:
            frag = left_seq[0:1500]
            fw_code = encode_seq(frag, gene_list[i], word2num_dict, strand='+')
            bw_code = encode_seq(frag.reverse_complement(), gene_list[i], word2num_dict, strand='-')
            curr_code += [fw_code+bw_code]
            left_seq = left_seq[1500:len(left_seq)]
        curr_code = pad_sequences(curr_code, maxlen=1500*2, dtype=ARRAY_TYPE, padding='post').tolist()

        # deal with the last fragment
        omit = False
        if 100 <= len(left_seq) <= 500:
            max_len = 500
        elif 500 <= len(left_seq) <= 1000:
            max_len = 1000
        elif 1000 <= len(left_seq) <= 1500:
            max_len = 1500
        else:
            omit = True
        if not omit:
            fw_code = encode_seq(left_seq, gene_list[i], word2num_dict, strand='+')
            bw_code = encode_seq(left_seq.reverse_complement(), gene_list[i], word2num_dict, strand='-')
            last_code = [fw_code + bw_code]
            last_code = pad_sequences(last_code, maxlen=max_len*2, dtype=ARRAY_TYPE, padding='post').tolist()
            curr_code += last_code
        code_list += [curr_code]
    if saved_path is not None:
        np.save(saved_path, code_list)
    return code_list


def encode_seq(seq, gene_infor, word2num_dict, strand):
    """
    Encode a contig

    :param seq: the seq object of the contig to be encoded
    :param gene_infor: the gene information of the contig
    :param word2num_dict: the dictionary storing the mapping between words(k mer) and codes
    :param strand: the strand being encoded
    """
    seq_len = len(seq)
    gene_region = []
    if strand == '+':
        for infor in gene_infor['infor']:
            if infor['strand'] == strand:
                gene_region += [(int(infor['start'])-1, int(infor['end']))]  # [(start, end), (start, end),...]
    else:  # strand == '-'
        seq_len = len(seq)
        for infor in gene_infor['infor']:
            if infor['strand'] == strand:
                gene_region += [(seq_len-int(infor['end']), seq_len-int(infor['start'])+1)]
        gene_region.reverse()

    # encode seq
    last_end = 0
    seq_code = []
    for index, region in enumerate(gene_region):
        (start, end) = region
        if index == 0 and start >= 3:
            # encode non-coding region
            seq_code += encode_by_kmer(seq[0: start], NON_GENE_K, word2num_dict[0])  # 1 mer
        if index > 0 and start > last_end:
            # encode non-coding region
            seq_code += encode_by_kmer(seq[last_end: start], NON_GENE_K, word2num_dict[0])  # 1 mer
        # encode gene_region
        seq_code += encode_by_kmer(seq[start: end], GENE_K, word2num_dict[1])   # 3 mer
        last_end = end
    if seq_len-last_end >= 3:
        # encode non-coding region
        seq_code += encode_by_kmer(seq[last_end:seq_len], NON_GENE_K, word2num_dict[0])  # 1 mer

    return seq_code


def encode_by_kmer(seq, k, word2num_dict):
    """
    Encode a sequence regarding k mer as the smallest unit

    :param seq: the seq object to be encoded
    :param k:  the given length of a word(k mer)
    :param word2num_dict: the dictionary storing the mapping between words(k mer) and codes
    """
    seq_code = []
    seq_len = len(seq)
    for step in range(k):
        for start in range(step, seq_len, k):
            end = start + k
            if end <= seq_len:
                k_mer = seq[start: end]
                try:
                    if k == GENE_K:
                        num = word2num_dict[k_mer] + ACCUM_NUM
                    else:         # k == NON_GENE_K
                        num = word2num_dict[k_mer]
                except KeyError:
                    num = 0
                seq_code += [num]
            else:
                break
    return seq_code


def get_gene_feature(gene_res, blast_res, saved_path=None):
    """
    Calculate gene features

    :param gene_res: the list of gene information
    :param blast_res: the result file of BLASTp
    :param saved_path: the file for saving the gene features
    """
    # get hit ids
    blast_df = pd.read_csv(blast_res, sep='\t', header=None)
    gene_ids = blast_df.iloc[:, 0]
    hit_ids = gene_ids.str.extract(r'^(.+)_\d+_\d+_[+|-]$', expand=False)
    hit_nums = hit_ids.value_counts()

    # calculate the ratios of hit genes
    hit_ratios = []
    for seq_genes in gene_res:
        total_num = len(seq_genes['infor'])
        seq_id = seq_genes['id']
        try:
            hit_ratios += [hit_nums[seq_id]/total_num]
        except KeyError:
            hit_ratios += [0]
        except ZeroDivisionError:
            hit_ratios += [0]
    if saved_path is not None:
        np.save(saved_path, hit_ratios)
    return hit_ratios


def get_average_output(code_list, models):
    """
    Calculate the average output of the CNN models for the contigs longer than 1.5kbp

    :param code_list: the code list of the contigs longer than 1.5kbp
    :param models: the CNN models used for getting the prediction score of the fragmented sequences
    """
    output_list = np.zeros(shape=(len(code_list), models[0].output.shape[1].value))
    for i, seq_code in enumerate(code_list):
        feature_list = []
        weights = []
        for frag_code in seq_code:
            seq_len = len(frag_code)
            frag_code = np.array([frag_code])
            if seq_len == 500*2:
                feature_list += models[0].predict(frag_code).tolist()
            elif seq_len == 1000*2:
                feature_list += models[1].predict(frag_code).tolist()
            elif seq_len == 1500*2:
                feature_list += models[2].predict(frag_code).tolist()
            weights += [seq_len]
        weights = np.array(weights)/sum(weights)
        weights = weights.reshape((len(weights), 1))
        avg_output = sum(weights * np.array(feature_list))
        output_list[i] = avg_output
    return output_list
