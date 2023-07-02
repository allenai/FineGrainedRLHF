# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the helper functions that split long text into sentences and subsentences
#  All Rights Reserved.
#  *********************************************************

import re

# split long text into sentences
def split_text_to_sentences(long_text, spacy_nlp):
    doc = spacy_nlp(long_text)
    return [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0]
    
    
# split long text into subsentences
def split_text_to_subsentences(long_text, spacy_nlp):
    def get_sub_sentence_starts(tokens, min_subsent_words=5):

        def _is_tok_end_of_subsent(tok):
            if re.match('[,;!?]', tok[-1]) is not None:
                return True
            return False

        # assert len(tokens) > 0
        is_subsent_starts = [True]
        prev_tok = tokens[0]
        prev_subsent_start_idx = 0
        for i, tok in enumerate(tokens[1:]):
            tok_id = i + 1
            if _is_tok_end_of_subsent(prev_tok) and tok_id + min_subsent_words < len(tokens):
                if tok_id - prev_subsent_start_idx < min_subsent_words:
                    if prev_subsent_start_idx > 0:
                        is_subsent_starts += [True]
                        is_subsent_starts[prev_subsent_start_idx] = False
                        prev_subsent_start_idx = tok_id
                    else:
                        is_subsent_starts += [False]
                else:
                    is_subsent_starts += [True]
                    prev_subsent_start_idx = tok_id
            else:
                is_subsent_starts += [False]
            prev_tok = tok

        return is_subsent_starts


    def tokenize_with_indices(text):
        tokens = text.split()
        token_indices = []

        current_index = 0
        for token in tokens:
            start_index = text.find(token, current_index)
            token_indices.append((token, start_index))
            current_index = start_index + len(token)

        return token_indices
    
    doc = spacy_nlp(long_text)
    sentence_start_char_idxs= [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0]
    
    char_starts = []
    
    for sentence_idx, sentence_start_char_idx in enumerate(sentence_start_char_idxs[:-1]):
        
        sentence = long_text[sentence_start_char_idx: sentence_start_char_idxs[sentence_idx+1]]
        
        tokens_with_indices = tokenize_with_indices(sentence)
        
        tokens = [i[0] for i in tokens_with_indices]
        is_sub_starts = get_sub_sentence_starts(tokens, min_subsent_words=5)
        
        for token_with_idx, is_sub_start in zip(tokens_with_indices, is_sub_starts):
            if is_sub_start:
                char_starts.append(sentence_start_char_idx + token_with_idx[1])
    
    return char_starts + [len(long_text)]