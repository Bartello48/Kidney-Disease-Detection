from datetime import datetime


class EvaluationMetric:
    def __init__(
        self,
        tp, fn, fp, tn,
        accuracy,
        precission,
        recall,
        f1
    ):
        now = datetime.now()
        now = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.payload = {
            'timestamp': now,
            'cancer': {
                'tp': tp[0].item(),
                'fn': fn[0].item(),
                'fp': fp[0].item(),
                'tn': tn[0].item(),
                'accuracy': accuracy[0].item(),
                'precission': precission[0].item(),
                'recall': recall[0].item(),
                'f1': f1[0].item()
            },
            'cyst': {
                'tp': tp[1].item(),
                'fn': fn[1].item(),
                'fp': fp[1].item(),
                'tn': tn[1].item(),
                'accuracy': accuracy[1].item(),
                'precission': precission[1].item(),
                'recall': recall[1].item(),
                'f1': f1[1].item()
            }
        }
