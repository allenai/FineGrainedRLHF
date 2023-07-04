import evaluate
import datasets

NO_ERROR_TAG = "O"

class Seqeval(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="",
            citation="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                }
            )
        )

    def _calc_scores(self, predictions, references, is_baseline=False):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for prediction, reference in zip(predictions, references):
            for p, g in zip(prediction, reference):
                if p != NO_ERROR_TAG and g != NO_ERROR_TAG:
                    tp += 1
                if p != g and g == NO_ERROR_TAG:
                    fp += 1
                if p != g and p == NO_ERROR_TAG:
                    fn += 1
                if p == g == NO_ERROR_TAG:
                    tn += 1
        
        # f1, precision and recall for "has error" class
        f1 = 0.0 if (tp+0.5*(fp+fn)) == 0.0 else (tp)/(tp+0.5*(fp+fn))
        p = 0.0 if (tp+fp) == 0.0 else tp/(tp+fp)
        r = 0.0 if (tp+fn) == 0.0 else tp/(tp+fn)
        
        acc = (tp + tn) / (tp + fp + tn + fn)
        
        # f1, precision and recall for "no error" class
        n_f1 = 0.0 if (tn+0.5*(fn+fp)) == 0.0 else (tn)/(tn+0.5*(fn+fp))
        n_p = 0.0 if (tn+fn) == 0.0 else tn/(tn+fn)
        n_r = 0.0 if (tn+fp) == 0.0 else tn/(tn+fp)
            
        err_result = {"precision": p, "recall": r, "f1": f1, "# class examples": tp+fn}
        no_err_result = {"precision": n_p, "recall": n_r, "f1": n_f1, "# class examples": tn+fp}

        macro_result = {"accuracy": acc, "# total examples": tp+fp+tn+fn}

        if not is_baseline:
            return {"hasError": err_result, "noError": no_err_result, "overall": macro_result}
        else:
            return {"accuracy": acc} # only report accuracy for trivial baselines (predicting everything as has error or no error)


    def _compute(
        self,
        predictions,
        references,
        is_baseline=False
    ):

        labels = set([item for sublist in references for item in sublist])
        scores = self._calc_scores(predictions, references, is_baseline=is_baseline)

        return scores
