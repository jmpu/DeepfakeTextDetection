# -*- coding: utf-8 -*-
"""
Adapted from TextFooler attack: https://github.com/jind11/TextFooler

"""

import os
import numpy as np
import torch
import time
import jsonlines
from sklearn.linear_model import LogisticRegression
import random
import argparse
import transformers
import pickle
import criteria
import re
import nltk.tokenize as nt
import nltk

import tensorflow_hub as hub
import torch.nn as nn

nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
import csv
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time

random.seed(0)

class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


def read_data(path, dataset_tag):
    data = []
    labels = []
    full_info_list = []
    if 'gltr' in dataset_tag or dataset_tag == 'gbrain':
        with jsonlines.open(path, 'r') as input_articles:
            for i, article in enumerate(input_articles):
                labels.append(1 if article['label'] == 'machine' else 0)
                word_list = article['text'].split(' ')
                clean_word_list = [word for word in word_list if word]
                data.append(clean_word_list)
                full_info_list.append(article)
    elif dataset_tag == 'grover' or dataset_tag == 'roberta' or dataset_tag == 'fast' or dataset_tag == 'df':
        with jsonlines.open(path, 'r') as input_articles:
            for i, article in enumerate(input_articles):
                if article['split'] == 'test' and article['orig_split'] == 'gen':
                    labels.append(1 if article['label'] == 'machine' else 0)
                    word_list = article['article'].split(' ')
                    clean_word_list = [word for word in word_list if word]
                    data.append(clean_word_list)
                    full_info_list.append(article)
    return data, labels, full_info_list


def get_rank_stat_from_an_article(article, lm, backend):
    raw_text = ' '.join(article)
    payload = lm.check_probabilities(raw_text, topk=5)

    payload_bpe_strings = payload['bpe_strings']
    payload = np.asarray(payload['real_topk'])

    rank_list = []
    prob_list = []
    for i in range(payload.shape[0]):
        rank = payload[i][0].astype(np.int64)
        prob = payload[i][1].astype(np.float32)

        rank_list.append(rank)
        prob_list.append(prob)

    if backend == 'bert':
        word_map = lm.process_token_and_ranks(payload_bpe_strings, rank_list[:-1], prob_list[:-1], raw_text)
    else:
        word_map = lm.process_token_and_ranks(payload_bpe_strings, rank_list, prob_list, raw_text)

    return word_map


def random_perturb_adversarial_attack(full_info_list, test_dataset, args, stop_words_set, word2idx, idx2word, cos_sim,
                            sim_predictor):
    num_attack = 0
    with open(args.attack_stat_csv1, mode='w') as csvf1:
        fieldnames = ['n_iter', 'original', 'perturb']
        writer = csv.DictWriter(csvf1, fieldnames=fieldnames)
        writer.writeheader()

    with open(args.attack_stat_csv2, mode='w') as csvf2:
        fieldnames = ['perturb idx', 'word pairs']
        writer = csv.DictWriter(csvf2, fieldnames=fieldnames)
        writer.writeheader()

    with jsonlines.open(args.output_new_file, 'w') as out_file:
        for i, article_data in enumerate(test_dataset):
            print('+++processing+++', i)
            if article_data[1] == 1:
                current_article = article_data[0]
                article_cache = current_article
                if num_attack == args.num_samples_to_attack: break
                perturb_stat = []
                iter_run, current_article = call_attack(args, article_cache,
                                                             current_article,
                                                             stop_words_set, word2idx, idx2word, cos_sim,
                                                             sim_predictor, perturb_stat)
                article_full_info = full_info_list[i]
                text_tag = 'text' if 'text' in article_full_info else 'article'
                article_full_info[text_tag] = ' '.join(current_article)
                out_file.write(article_full_info)
                num_attack += 1

    print("++++++++ATTACK Finished++++++++")


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count, threshold):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]


        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


def find_synonym_to_alter(article_cache, current_article, idx, filtered_synonyms, len_text):
    synonym_to_alter = ''
    pos_ls = criteria.get_pos(article_cache)


    for filtered_synonym in filtered_synonyms:

        synonym = filtered_synonym[0]
        new_text = current_article[:idx] + [synonym] + current_article[min(idx + 1, len_text):]
        pass_pos = check_pos_tag(new_text, idx, pos_ls)
        if pass_pos:
            synonym_to_alter = synonym # choose the most similar synonym
            break
    return synonym_to_alter


def check_pos_tag(current_article, idx, pos_ls):
    synonyms_pos_ls = [criteria.get_pos(current_article[max(idx - 4, 0):idx + 5])[min(4, idx)] if len(current_article) > 10 else criteria.get_pos(current_article)[idx]]

    pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls)) # synonyms_pos_ls in batch

    return pos_mask[0]


def sentence_similarity_filter(current_article, synonyms, values, len_text, idx, sim_predictor,
                               sim_score_window, sim_score_threshold):
    valid_synonyms = []
    for i, (synonym, value) in enumerate(zip(synonyms[0], values[0])):

        new_text = current_article[:idx] + [synonym] + current_article[min(idx + 1, len_text):]
        half_sim_score_window = (sim_score_window - 1) // 2
        # compute semantic similarity
        if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
            text_range_min = idx - half_sim_score_window
            text_range_max = idx + half_sim_score_window + 1
        elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
            text_range_min = 0
            text_range_max = sim_score_window
        elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
            text_range_min = len_text - sim_score_window
            text_range_max = len_text
        else:
            text_range_min = 0
            text_range_max = len_text
        semantic_sims = \
            sim_predictor.semantic_sim([' '.join(current_article[text_range_min:text_range_max])],
                                       [' '.join(new_text[text_range_min:text_range_max])])[0]

        if semantic_sims >= sim_score_threshold:
            valid_synonyms.append((synonym, value))

    return valid_synonyms


def save_stat(article_cache, current_article, perturb_stat, csv_file1, csv_file2, iter):
    with open(csv_file1, mode='a') as csvf1:
        csvwriter = csv.writer(csvf1)
        csvwriter.writerow([iter, ' '.join(article_cache), ' '.join(current_article)])
    perturb_idxs = [t[2] for t in perturb_stat]
    perturb_word_pairs = [(t[0], t[1]) for t in perturb_stat]
    with open(csv_file2, mode='a') as csvf2:
        csvwriter = csv.writer(csvf2)
        csvwriter.writerow([perturb_idxs, perturb_word_pairs])


def call_attack(args, article_cache, current_article, stop_words_set,
                     word2idx, idx2word,
                     cos_sim,
                     sim_predictor, perturb_stat):
    len_text = len(current_article)
    candidate_set_raw = random.sample(list(enumerate(current_article)), len_text)
    candidate_set_final = []
    for idxx, wordx in candidate_set_raw:
        if wordx not in stop_words_set and wordx in word2idx:
            candidate_set_final.append((idxx, wordx))

    num_replacement = 0
    while len(candidate_set_final) > 0:
        if num_replacement == args.max_iter: break
        idx, word = candidate_set_final.pop(0)
        words_perturb_idx = [word2idx[word]]

        synonyms, sim_values = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, 50,
                                                             0.7)

        filtered_synonyms = sentence_similarity_filter(current_article, synonyms, sim_values,
                                                       len_text, idx, sim_predictor,
                                                       15, args.sim_thre)

        synonym_to_alter = find_synonym_to_alter(article_cache, current_article, idx, filtered_synonyms,
                                                    len_text)

        if len(synonym_to_alter) < 1 or synonym_to_alter == '': continue

        perturb_stat.append((current_article[idx], synonym_to_alter, idx))

        current_article = current_article[:idx] + [synonym_to_alter] + current_article[min(idx + 1, len_text):]
        num_replacement += 1


    save_stat(article_cache, current_article, perturb_stat, args.attack_stat_csv1, args.attack_stat_csv2,
              num_replacement)
    return num_replacement, current_article


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--counter_fitting_cos_sim_path',
        default='./cos_sim_counter_fitting_validate.npy',
        type=str,
        required=False,
    )

    parser.add_argument(
        '--counter_fitting_embeddings_path',
        default='./counter-fitted-vectors.txt',
        type=str,
        required=False,
    )

    parser.add_argument(
        '--USE_cache_path',
        default='./models/tf',
        type=str,
        required=False,
    )

    parser.add_argument(
        '--attack_dataset_path', # the dataset under attack
        default='',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--dataset_tag',   # indicates the format of dataset to be processed. used in read_data.
        default='roberta',
        type=str,
        required=False,
    )

    parser.add_argument(
        '--max_iter', # number of word perturbations per document
        default=10,
        type=int,
        required=False,
    )

    parser.add_argument(
        '--sim_thre',  # sentence semantic similarity threshold.
        default=0.7,
        type=float,
        required=False,
    )

    parser.add_argument(
        '--num_samples_to_attack',
        default=1000,
        type=int,
        required=False,
    )

    parser.add_argument(
        '--attack_stat_csv1', # a csv file to save statistics/records during the perturbation attack
        default='',
        type=str,
        required=False,
    )
    parser.add_argument(
        '--attack_stat_csv2', # second csv file to save statistics/records during the perturbation attack
        default='',
        type=str,
        required=False,
    )

    parser.add_argument(
        '--output_new_file', # specify a jsonl file to save perturbed articles
        default='',
        type=str,
        required=False,
    )

    args = parser.parse_args()

    data, labels, full_info_list = read_data(args.attack_dataset_path, args.dataset_tag)
    data = list(zip(data, labels))

    data = data[:args.num_samples_to_attack]  # choose how many samples for adversarial attack (add 500 buffer)
    full_info_list = full_info_list[:args.num_samples_to_attack]
    print("Data import finished!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        cos_sim = np.load(args.counter_fitting_cos_sim_path)
    else:
        # calculate the cosine similarity matrix
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        product = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos_sim = product / np.dot(norm, norm.T)
    print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(args.USE_cache_path)
    stop_words_set = criteria.get_stopwords()
    random_perturb_adversarial_attack(full_info_list, data, args, stop_words_set, word2idx, idx2word, cos_sim,
                            sim_predictor=use)
