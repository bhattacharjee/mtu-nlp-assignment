#/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from tqdm import tqdm
from backtranslate import *
import traceback
import random


def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    return dataset_dict

def print_question(question, context, ind, answer):
    print(f"ID:            {ind}")
    print(f"CONTEXT:       {context}")
    print(f"QUESTOIN:      {context}")
    print(f"ANSWER:        {answer}")
    anslen = len(ans)

def get_new_context(context:str, answer:str)->tuple:
    """
    Replace the answer in the context with a number
    so that it doesn't backtranslate

    Return a tuple:
        1. string that was replaced, handy while restoring
           the answer in the translated string
        2. translated question
    """
    replaced_str = ""
    start_ind = context.index(answer)
    orig_ans = answer
    len_orig_ans = len(orig_ans)
    context = list(context)

    #rc = list("123456789abcdefgh0ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST")
    #random.shuffle(rc)

    for i in range(len_orig_ans):
        replaced_str +=  '1'
        context[i + start_ind] = '1'

    print('-' * 80)
    print(replaced_str)
    print('-' * 80)
    return "".join(context), replaced_str

def get_start_end_index(context, replaced):
    with open("save.txt", "a") as f:
        print(context, file=f)
        print(replaced, file=f)

    try:
        start_ind = context.index(replaced)
    except:
        # Sometimes the replaced string is truncated by the
        # translation, try again with half the string to see
        # if there is a match
        if len(replaced) < 4:
            return -1, -1
        try:
            start_ind = context.index(replaced[len(replaced) // 2])
        except:
            if len(replaced) < 8:
                return -1, -1
            try:
                start_ind = context.index(replaced[len(replaced) // 4])
            except:
                return -1, -1

    start_char = replaced[0]
    end_ind = start_ind
    while end_ind < len(context) and context[end_ind] == start_char:
        end_ind += 1
    return start_ind, end_ind


    
def reconstruct_context(context:str, replaced:str, origans:str)->str:
    """
    Take the backtranslated context, and replace the placeholder
    for the answer with the actual answer
    """
    start_index, end_index = get_start_end_index(context, replaced)
    str1 = context[:start_index]
    str2 = context[end_index:]
    return str1 + origans + str2

def back_translate_context(context, answer):
    temp_ctx, replaced = get_new_context(context, answer)
    trns_ctx = back_translate(temp_ctx)[0]
    new_ctx = reconstruct_context(trns_ctx, replaced, answer)
    return new_ctx


def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
    log = util.get_logger(args.save_dir, 'log_train')
    train_dataset = get_dataset(args, args.train_datasets, args.train_dir, None, 'train')

    print(train_dataset.keys())

    forig = open("orig.txt", "w")
    fnew = open("new.txt", "w")

    questions = train_dataset['question']
    contexts = train_dataset['context']
    ids = train_dataset['id']
    answers = train_dataset['answer']
    j = 0
    for q, c, i, a in zip(questions, contexts, ids, answers):
        answer = a['text'][0]
        nc = back_translate_context(c, answer)
        nq = back_translate(q)

        print(c, file=forig)
        print(q, file=forig)
        print(answer, file=forig)
        print("", file=forig)

        print(nc, file=fnew)
        print(nq, file=fnew)
        print(answer, file=fnew)
        print("", file=fnew)

        j += 1
        if j > 10: break

    forig.close()
    fnew.close()


if __name__ == '__main__':
    main()
