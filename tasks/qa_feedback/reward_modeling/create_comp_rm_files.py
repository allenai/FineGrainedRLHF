import argparse
import json
import spacy
import sys
import re
import os

from difflib import SequenceMatcher as SM

from nltk.util import ngrams


nlp = spacy.load('en_core_web_sm') # Load the English Model
MIN_SUBSENT_WORDS = 5
MIN_SIM_SCORE = 0.85

def _find_approximate_matching_sequence(context, target):
    """ Find some substring in the context which closely matches the target, returning this substring with a score.
        Source: https://github.com/apple/ml-qrecc/blob/main/utils/span_heuristic.py
    """
    if target in context:
        return target, 1.0

    target_length = len(target.split())
    max_sim_val = 0
    max_sim_string = ''
    seq_matcher = SM()
    seq_matcher.set_seq2(target)
    for ngram in ngrams(context.split(), target_length + int(0.05 * target_length)):
        candidate_ngram = ' '.join(ngram)
        seq_matcher.set_seq1(candidate_ngram)
        similarity = seq_matcher.quick_ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = candidate_ngram
        if similarity == 1.0:
            # early exiting
            break

    return max_sim_string, max_sim_val


def get_subsentence_starts(tokens):
    """Get the indices of the tokens that start a subsentence."""
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
    return [i for i, is_start in enumerate(is_subsent_starts) if is_start]


def find_best_p_sents(text, all_p_sents):
    """ Find the best passage sentences for each subsentence in the LM predicted output. """
    doc = nlp(' '.join(text.strip().split()))
    subsents = []

    for s in doc.sents:
        s_text = s.text
        tokens = s_text.split()
        sent_start_idx = get_subsentence_starts(tokens)
        for i, idx in enumerate(sent_start_idx):
            if i < len(sent_start_idx) - 1:
                subsents += [' '.join(tokens[idx:sent_start_idx[i+1]])]
            else:
                subsents += [' '.join(tokens[idx:])]

    res = []
    for subsent in subsents:
        max_sim = -1
        max_sim_sent = ""
        max_sim_sent_id = (-1, -1)
        for sent_id, p_sent in all_p_sents.items():
            subsent_length = len(subsent.split())
            p_sent_length = len(p_sent.split())
            diff = subsent_length + int(0.05 * subsent_length) - p_sent_length + 1
            if diff > 0:  # add some padding, otherwise the returned sim score is always 0
                _, cur_max_sim = _find_approximate_matching_sequence(p_sent+' #' * diff, subsent)
            else:
                _, cur_max_sim = _find_approximate_matching_sequence(p_sent, subsent)
            if cur_max_sim > max_sim:
                max_sim = cur_max_sim
                max_sim_sent = p_sent
                max_sim_sent_id = sent_id
        res += [(subsent, max_sim_sent, max_sim_sent_id, max_sim)]
    
    return res


def get_coverage_score(pred, all_p_sents, all_info_ids):
    """Compute the percentage of information ids that are covered by the LM predicted output."""
    
    # map LM output to passage sentences
    pred_p_sents = find_best_p_sents(pred, all_p_sents)
    pred_info_ids = set()

    for sent in pred_p_sents:
        subsent, sim_sent, sim_sent_id, sim = sent

        if sim < MIN_SIM_SCORE:
            continue

        info_id = (sim_sent_id[0], sim_sent_id[1])
        pred_info_ids.add(info_id)
    
    score = len(pred_info_ids.intersection(all_info_ids)) * 1.0 / len(all_info_ids)

    return score


def get_preference(score1, score2):
    if score1 > score2:
        return 1
    elif score1 < score2:
        return 2
    else:
        return 0


def main(args):

    # load examples
    for data_split in ['dev', 'train']:
        with open(os.path.join(args.input_dir, f'{data_split}_feedback.json')) as fin:
            examples = json.loads(fin.read())
            output_examples = []
            for e_id, e in enumerate(examples):
                output_example = dict(e)

                # read all passage sentences
                all_p_sents = {}
                for i, p in enumerate(e["passages"]):
                    for j, sent in enumerate(p[1:]):
                        all_p_sents[(i+1, j+1)] = ' '.join(sent.split())

                # some sentences only contain minor relevant info
                minor_info_ids = set()
                for m in e["feedback"]["missing-info"]:
                    if m['error type'] == "Missing-Minor-Auxiliary":
                        for sid in m["sentence_id"]:
                            minor_info_ids.add((m['passage_id'], sid))

                # get all passage sentences with relevant info by mapping each subsent 
                # in human-written corrections to the grounding passage sentences
                # read such sentences as all_info_ids
                correct = e["feedback"]["corrected-prediction"]
                correct = ' '.join(correct.strip().split())
                correct_p_sents = find_best_p_sents(correct, all_p_sents)
                all_info_ids = set()
                minus_info_ids = set()
                for sent in correct_p_sents:
                    subsent, sim_sent, sim_sent_id, sim = sent
                    info_id = (sim_sent_id[0], sim_sent_id[1])
                    if sim < MIN_SIM_SCORE:
                        continue
                    all_info_ids.add(info_id)

                    # do not count sentences with minor info only
                    if info_id in minor_info_ids:
                        minus_info_ids.add(info_id)
                        if subsent.lower() in e["prediction 1"].lower():
                            minus_info_ids.discard(info_id)

                all_info_ids = all_info_ids.difference(minus_info_ids)

                # skip examples without relevant info (could be because of 
                # no useful grounding passage)
                if len(all_info_ids) == 0:
                    continue

                # match each LM output to all relevant info and calcalate the completion ratio
                pred = e["prediction 1"]
                score1 = get_coverage_score(pred, all_p_sents, all_info_ids)

                pred = e["prediction 2"]
                score2 = get_coverage_score(pred, all_p_sents, all_info_ids)

                pred = e["prediction 3"]
                score3 = get_coverage_score(pred, all_p_sents, all_info_ids)

                pred = e["prediction 4"]
                score4 = get_coverage_score(pred, all_p_sents, all_info_ids)

                prefs = [
                    get_preference(score1, score2),
                    get_preference(score1, score3),
                    get_preference(score1, score4),
                    get_preference(score2, score3),
                    get_preference(score2, score4),
                    get_preference(score3, score4)
                ]

                output_example["preference"] = prefs
                output_examples += [output_example]

        os.makedirs(args.output_dir, exist_ok=True)

        with open(os.path.join(args.output_dir, f'{data_split}.json'), 'w') as fout:
            fout.write(json.dumps(output_examples, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./data/COMP_sequence/")

    args = parser.parse_args()

    main(args)