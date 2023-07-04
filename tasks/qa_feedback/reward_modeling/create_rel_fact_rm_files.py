import json
import spacy
from collections import defaultdict, Counter
import argparse
import re
import os

nlp = spacy.load('en_core_web_sm') # Load the English Model

IGNORE_TAG = "Ignore"
SEP_TOKEN = "</s>"
NO_ERROR_TAG = "O"

NON_FACTUAL_ERROR_TAG = "NF-ERR"
FACTUAL_ERROR_TAG = "F-ERR"
FACTUAL_ERRORS = ["Wrong-Grounding", "Unverifiable"] # "Wrong-Grounding" refers to "inconsistent fact" in the paper.
ERROR_CATEGORIES = [NON_FACTUAL_ERROR_TAG, FACTUAL_ERROR_TAG]

MIN_SUBSENT_WORDS = 5


def get_token_labels(
    args,
    tokens, 
    token_is_sent_starts, 
    text, 
    spans
):

    token_idx2char_idx = defaultdict(list)
    cur_char_idx = 0
    for i, t in enumerate(tokens):
        l = len(t.strip())
        for j in range(cur_char_idx, cur_char_idx + l):
            token_idx2char_idx[i] += [j]
        cur_char_idx = cur_char_idx + l
    
    # sanity check: tokens and text has the same number of non-empty characters
    assert cur_char_idx == len(''.join(text.strip().split()))
    
    """
        orig_char_idx2char_idx: map the original character index to the character index in the tokenized string
        example:
            original_string:   " xxx yyy zzz "
            char_idx:          "0012334566789"
    """
    orig_char_idx2char_idx = {}
    cur_char_idx = 0
    for i, char in enumerate(text):
        orig_char_idx2char_idx[i] = cur_char_idx
        if char.strip() != '':
            cur_char_idx += 1
    
    char_idx2label = defaultdict(str)
    for span in spans:

        start = span["start"]
        end = span["end"]
        etype = span['error type']

        error_words = text[start:end].strip().split()
        # filter out error spans that are too short
        # Note: that this only applies to irrelevance / redundant / incoherence errors
        # See Appendix D
        if args.min_span_len is not None and len(error_words) < args.min_span_len and etype not in FACTUAL_ERRORS:
            continue

        e_category = FACTUAL_ERROR_TAG if etype in FACTUAL_ERRORS else NON_FACTUAL_ERROR_TAG
        
        assert args.error_category in ERROR_CATEGORIES
        assert start >= 0
        assert end >= 0

        # make sure the left of start index or the end index points to an empty char
        assert text[start].strip() != '' and (start == 0 or text[start-1].strip() == '')
        assert (end == len(text) or text[end].strip() == '')

        for orig_char_idx in range(start, end):
            char_idx2label[orig_char_idx2char_idx[orig_char_idx]] = e_category   # assuming no overlapped error span
        
    # make sure the first token in a string is always the start of a sentence
    assert token_is_sent_starts[0]

    def _get_sent_label(token_error_labels):
        if args.error_category == NON_FACTUAL_ERROR_TAG:
            return NON_FACTUAL_ERROR_TAG if NON_FACTUAL_ERROR_TAG in token_error_labels else NO_ERROR_TAG
        else:
            if FACTUAL_ERROR_TAG in token_error_labels:
                return FACTUAL_ERROR_TAG
            # due to our feedback annotation restriction #2 (see paper),
            # if a sentence contains non-factual errors, we ignore this sentence for training of factual error RM
            elif NON_FACTUAL_ERROR_TAG in token_error_labels:     
                return IGNORE_TAG
            else:
                return NO_ERROR_TAG 

    # get sentence labels
    sent_labels = []
    error_labels = set()
    for token_idx in range(len(tokens)):
        char_indices = token_idx2char_idx[token_idx]
        labels = [char_idx2label[idx] for idx in char_indices]

        # make sure each token finds at most one tag label (see in paper annotation restriction #1)
        assert len(set(labels)) <= 1
        if token_is_sent_starts[token_idx]:
            if token_idx > 0:
                sent_labels += [_get_sent_label(error_labels)]
            error_labels = set()
        error_labels.update(set(labels))
    sent_labels += [_get_sent_label(error_labels)]

    # assign the sentence label to the first sentence token and ignore the rest
    token_labels = []
    new_tokens = []
    sent_idx = 0
    for i, (t, is_start) in enumerate(list(zip(tokens, token_is_sent_starts))):
        if is_start:
            new_tokens += [SEP_TOKEN, t]
            token_labels += [sent_labels[sent_idx], IGNORE_TAG]
            sent_idx += 1
        else:
            new_tokens += [t]
            token_labels += [IGNORE_TAG]
        
    return new_tokens, token_labels


def create_example_input(
    id, 
    question_tokens, 
    question_token_labels, 
    answer_tokens, 
    answer_token_labels, 
    passage_tokens, 
    passage_token_labels
):

    # question tokens
    tokens = ["question:"] + question_tokens
    token_labels = [IGNORE_TAG] + question_token_labels

    # add passage tokens
    for i, p_tokens in enumerate(passage_tokens):
        if i == 0:
            tokens += ["context:"]
            token_labels += [IGNORE_TAG]
        tokens += p_tokens
        token_labels += passage_token_labels[i]
        
    # answer tokens
    tokens = tokens + ['answer:'] + answer_tokens
    token_labels = token_labels + [IGNORE_TAG] + answer_token_labels
    
    return {"id": str(id), "text": tokens, "feedback_tags": token_labels}


def get_subsentence_starts(tokens):

    def _is_tok_end_of_subsent(tok):
        if re.match('[,;!?]', tok[-1]) is not None:
            return True
        return False

    assert len(tokens) > 0
    is_subsent_starts = [True]
    prev_tok = tokens[0]
    prev_subsent_start_idx = 0
    for i, tok in enumerate(tokens[1:]):
        tok_id = i + 1
        if _is_tok_end_of_subsent(prev_tok) and tok_id + MIN_SUBSENT_WORDS < len(tokens):
            if tok_id - prev_subsent_start_idx < MIN_SUBSENT_WORDS:
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


def construct_model_inputs(args, ids, labels, inputs, split):

    all_results = []
    all_labels = []
    for id, label, input in zip(ids, labels, inputs):
    
        # get (LM-predicted) answer tokens
        doc = nlp(input['prediction 1'].strip())   # we only labeled feedback for the first prediction
        answer_tokens = []
        answer_token_is_sent_starts = []
        for s in doc.sents:
            s_text = s.text
            answer_tokens += s_text.split()
            if args.feedback_level == 'subsentence':
                answer_token_is_sent_starts += get_subsentence_starts(s_text.split())
            else:
                answer_token_is_sent_starts += [True] + (len(s_text.split()) - 1) * [False]
        text = input['prediction 1']
        answer_token_is_sent_starts[0] = True

        spans = []
        if len(label['errors']) > 0:
            spans = label['errors']
        
        ans_tokens, token_labels = get_token_labels(
            args,
            answer_tokens, 
            answer_token_is_sent_starts, 
            text, 
            spans
        )
                       
        if all([l == IGNORE_TAG for l in token_labels]):
            continue

        all_labels += [l for l in token_labels if l != IGNORE_TAG]

        # get question tokens
        question_tokens = input['question'].strip().split()
        # get default question token error labels
        question_token_labels = [IGNORE_TAG] * len(question_tokens)

        passage_tokens = []
        passage_token_labels = []
        if not args.ignore_context:
            for i, p in enumerate(input['passages']):
                title_string = f"wikipage: {p[0]}"
                p_tokens = title_string.split()
                p_tokens += ['text:'] + ' '.join(p[1:]).split()
                p_token_labels = [IGNORE_TAG] * len(p_tokens)
        
                passage_tokens += [p_tokens]
                passage_token_labels += [p_token_labels]
        
        input_json = create_example_input(
            id, 
            question_tokens, 
            question_token_labels, 
            ans_tokens, 
            token_labels, 
            passage_tokens, 
            passage_token_labels
        )
        all_results += [input_json]

    counter = Counter(all_labels)
    print("label distribution: ")
    for k in counter:
        print(k, counter[k]*1.0 / sum(list(counter.values())))
    
    print("total examples: ", len(all_results))

    output_dir = os.path.join(args.data_dir, f"{args.error_category}_{args.feedback_level}")
    os.makedirs(output_dir, exist_ok = True)

    with open(os.path.join(output_dir, f"{split}.json"), "w") as fout:
        for line in all_results:
            fout.write(json.dumps(line)+'\n')
    
    
def _read_id2example(filename):
    id2example = {}
    with open(filename) as fin:
        examples = json.loads(fin.read())
        for id, e in enumerate(examples):
            id2example[id] = e
    return id2example


def main(args):

    for split in ["train", "dev"]:
        id2example = _read_id2example(os.path.join(args.data_dir, f"{split}_feedback.json"))
        labels = []
        inputs = []
        ids = id2example.keys()
        for id in sorted(list(ids)):
            e = id2example[id]
            label = e['feedback']
            labels += [label]
            inputs += [e]
        construct_model_inputs(args, sorted(list(ids)), labels, inputs, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--feedback_level", type=str, default="sentence", choices=['sentence', 'subsentence'], 
                        help="fine-grained feedback density level")
    
    parser.add_argument("--data_dir", type=str, default="./", help="path to the data directory")
    
    parser.add_argument("--error_category", type=str, choices=ERROR_CATEGORIES, required=True, 
                        help="fine-grained feedback category")

    parser.add_argument("--min_span_len", type=int, default=5, 
                        help="used to filter out error spans that are too short")
    
    parser.add_argument("--ignore_context", action='store_true', help="whether to ignore passage context")

    args = parser.parse_args()
    main(args)