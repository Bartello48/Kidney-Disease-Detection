from datetime import datetime


class EvaluationMetric:
    def __init__(
        self,
        tp, fn, fp, tn,
        threshold_cancer,
        threshold_cyst,
        loss=None
    ):
        now = datetime.now()
        now = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.timestamp = now

        self.tp = tp
        self.fn = fn
        self.fp = fp
        self.tn = tn
        self.threshold_cancer = threshold_cancer
        self.threshold_cyst = threshold_cyst

        self.precission = tp.float() / (tp + fp).clamp_min(1)
        self.recall = tp.float() / (tp + fn).clamp_min(1)
        self.f1 = (
            2 * self.precission * self.recall
            / (self.precission + self.recall).clamp_min(1e-8)
        )
        self.accuracy = (tp + tn).float() / (tp + tn + fp + fn).clamp_min(1)
        self.fpr = fp.float() / (fp + tn).clamp_min(1)

        self.loss = loss
        self.loss_aplicable = True
        if loss is None:
            self.loss_aplicable = False

    @property
    def false_positive_rate(self):
        return self.fpr

    @property
    def payload(self):
        payload = {
            'timestamp': self.timestamp,
            'threshold_cancer': self.threshold_cancer,
            'threshold_cyst': self.threshold_cyst,
            'cancer': {
                'tp': self.tp[0].item(),
                'fn': self.fn[0].item(),
                'fp': self.fp[0].item(),
                'tn': self.tn[0].item(),
                'accuracy': self.accuracy[0].item(),
                'precission': self.precission[0].item(),
                'recall': self.recall[0].item(),
                'false_positive_rate': self.fpr[0].item(),
                'f1': self.f1[0].item()
            },
            'cyst': {
                'tp': self.tp[1].item(),
                'fn': self.fn[1].item(),
                'fp': self.fp[1].item(),
                'tn': self.tn[1].item(),
                'accuracy': self.accuracy[1].item(),
                'precission': self.precission[1].item(),
                'recall': self.recall[1].item(),
                'false_positive_rate': self.fpr[1].item(),
                'f1': self.f1[1].item()
            }
        }
        if self.loss_aplicable:
            payload['loss'] = self.loss
        return payload

    def print_results(self):
        print("--- RESULTS ---")
        print(f"cancer threshold: {self.threshold_cancer}")
        print(f"cyst threshold: {self.threshold_cyst}")
        if self.loss_aplicable:
            print(f"loss: {self.loss}")
        # cancer
        print("--- CANCER ---")
        print(f"tp: {self.tp[0].item()}\tfn: {self.fn[0].item()}")
        print(f"fp: {self.fp[0].item()}\ttn: {self.tn[0].item()}")
        print(f"Accuracy: {self.accuracy[0].item()}")
        print(f"Precission: {self.precission[0].item()}")
        print(f"Recall: {self.recall[0].item()}")
        print(f"False Positive Rate: {self.fpr[0].item()}")
        print(f"F1 score: {self.f1[0].item()}")
        # cyst
        print("--- CYST ---")
        print(f"tp: {self.tp[1].item()}\tfn: {self.fn[1].item()}")
        print(f"fp: {self.fp[1].item()}\ttn: {self.tn[1].item()}")
        print(f"Accuracy: {self.accuracy[1].item()}")
        print(f"Precission: {self.precission[1].item()}")
        print(f"Recall: {self.recall[1].item()}")
        print(f"False Positive Rate: {self.fpr[1].item()}")
        print(f"F1 score: {self.f1[1].item()}")
