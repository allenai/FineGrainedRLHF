from rouge_score import rouge_scorer
import nltk

scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)

def postprocess_text(preds, list_of_labels):
    
    # rougeLSum expects newline after each sentence
    preds = ['\n'.join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    list_of_labels = [['\n'.join(nltk.sent_tokenize(label.strip())) for label in labels] 
                      for labels in list_of_labels]
    return preds, list_of_labels


def get_rouge_scores(preds, list_of_labels):

    # Post-process text
    preds, list_of_labels = postprocess_text(preds, list_of_labels)

    # Score all predictions
    all_scores = []
    for pred, labels in zip(preds, list_of_labels):
        # We calculate scores for each label, and take the max
        label_scores = [scorer.score(pred, label) for label in labels]
        max_score = max(label_scores, key=lambda x: x['rougeLsum'].fmeasure)
        all_scores.append(max_score)
    
    all_scores = [round(v['rougeLsum'].fmeasure * 100, 4) for v in all_scores]
    return all_scores