from typing import Optional

import itertools


def read_file(filename : str):
    words = []
    with open(filename, "r") as f:
        for line in f:
            words.append(line.strip())
    return words


def augment_word(word: str, num_holes: Optional[int] = None):
    if num_holes is not None:
        assert num_holes > 0 and num_holes <= len(word)
        
        word_char = list(set(list(word)))
        indices = itertools.combinations(range(len(word_char)), num_holes)
        
        words = []
        for index in indices:
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
    words = read_file(filename)
    word_len = [len(word) for word in words]
    max_word_length = max(word_len)
    
    split_len = int(len(words) * train_test_split)
    train_words = words[:split_len]
    test_words = words[split_len:]
    
    return [preprocess_word(word, mode="train") for word in train_words], [preprocess_word(word, mode="test") for word in test_words], max_word_length
    