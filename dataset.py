from typing import Optional

import itertools
import random
import math
import numpy as np


def read_file(filename : str):
    with open(filename, 'r') as f:
        words = f.readlines()
    for i in range(len(words)):
        words[i] = words[i].strip().split()
    return words


def augment_word(word: str, num_holes: Optional[int] = None):
    if num_holes is not None:
        assert num_holes > 0 and num_holes <= len(word)
        
        word_char = list(set(list(word)))
        num_comb = math.comb(len(word_char), num_holes)
        
        indices = itertools.combinations(range(len(word_char)), num_holes)
        if num_comb > 5:
            prob = 5. / num_comb
        else:
            prob = 1
        
        words = []
        for index in indices:
            if prob < np.random.rand():
                continue
            
            word_copy = word
            for i in index:
                word_copy = word_copy.replace(word_char[i], '_')
            words.append((word_copy, word))
        return words
    else:
        words = []
        for i in range(1, len(word) + 1):
            words.extend(augment_word(word, i))
        return words


def preprocess_word(word: str, mode="train"):
    word = word.lower()
    
    if mode == "test":
        return ["_" * len(word), word]
    elif mode == "train":
        return augment_word(word)
    else:
        raise NotImplementedError


def prepare_dataset(filename: str, train_test_split: float = 0.8):
    words = []
    with open(filename, "r") as f:
        for line in f:
            words.append(line.strip())
    
    word_len = [len(word) for word in words]
    max_word_length = max(word_len)
    
    split_len = int(len(words) * train_test_split)
    train_words = words[:split_len]
    test_words = words[split_len:]
    
    return list(itertools.chain(*[preprocess_word(word, mode="train") for word in train_words])), [preprocess_word(word, mode="test") for word in test_words], max_word_length
    # return [preprocess_word(word, mode="test") for word in train_words], [preprocess_word(word, mode="test") for word in test_words], max_word_length
    