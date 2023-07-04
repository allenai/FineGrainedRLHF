import evaluate
import nltk

metric = evaluate.load("rouge")

def postprocess_text(preds, list_of_labels):
    
    # rougeLSum expects newline after each sentence
    preds = ['\n'.join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    list_of_labels = [['\n'.join(nltk.sent_tokenize(label.strip())) for label in labels] 
                      for labels in list_of_labels]
    return preds, list_of_labels

def get_rouge_scores(preds, labels):

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(preds, labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=False)
    result = [round(v * 100, 4) for v in result['rougeLsum']]
    return result
