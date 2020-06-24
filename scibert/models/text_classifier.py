from typing import Dict, Optional, List, Any

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Auc
from overrides import overrides


@Model.register("text_classifier")
class TextClassifier(Model):
    """
    Implements a basic text classifier:
    1) Embed tokens using `text_field_embedder`
    2) Seq2SeqEncoder, e.g. BiLSTM
    3) Append the first and last encoder states
    4) Final feedforward layer

    Optimized with CrossEntropyLoss.  Evaluated with CategoricalAccuracy & F1.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 verbose_metrics: False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 ) -> None:
        super(TextClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward
        self.prediction_layer = torch.nn.Linear(self.classifier_feedforward.get_output_dim(), self.num_classes)

        self.label_auc_global = Auc(positive_label=1)
        self.verbose_metrics = verbose_metrics
        self.label_f1_global = F1Measure(positive_label=1)

        self.label_average_accuracy = 0
        self.label_auc_local = {}
        for i in range(self.num_classes):
            self.label_auc_local[vocab.get_token_from_index(index=i, namespace="labels")] = Auc(positive_label=1)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.pool = lambda text, mask: util.get_final_encoder_states(text, mask, bidirectional=True)

        initializer(self)

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.
        Returns
        -------
        An output dictionary consisting of:
        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self.text_field_embedder(text)

        mask = util.get_text_field_mask(text)
        encoded_text = self.text_encoder(embedded_text, mask)
        pooled = self.pool(encoded_text, mask)
        ff_hidden = self.classifier_feedforward(pooled)
        logits = self.prediction_layer(ff_hidden)
        class_probs = torch.sigmoid(logits)

        output_dict = {"logits": logits}
        if label is not None:
            loss = self.loss(logits, label.float())
            output_dict["loss"] = loss
            # compute F1
            combine = torch.cat((1 - class_probs.unsqueeze(-1), class_probs.unsqueeze(-1)), -1)
            self.label_f1_global(combine, label)

            class_probs[class_probs > 0.5] = 1
            class_probs[class_probs <= 0.5] = 0
            class_probs = class_probs.int()
            # compute Auc per label
            for i in range(self.num_classes):
                metric = self.label_auc_local[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(class_probs[:, i].view(-1), label[:, i].view(-1))
            self.label_auc_global(class_probs, label)
            self.label_average_accuracy = float(sum(class_probs.view(-1) == label.view(-1))) / class_probs.numel()

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        class_probabilities = torch.sigmoid(output_dict['logits'])
        output_dict['class_probs'] = class_probabilities
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        sum_auc = 0.0
        for name, metric in self.label_auc_local.items():
            metric_val = metric.get_metric(reset)
            if self.verbose_metrics:
                metric_dict[name + '_Auc'] = metric_val
            sum_auc += metric_val

        names = list(self.label_auc_local.keys())
        total_len = len(names)
        average_auc = sum_auc / total_len
        metric_dict['average_Auc'] = average_auc
        metric_dict['global_Auc'] = self.label_auc_global.get_metric(reset)

        precision, recall, f1 = self.label_f1_global.get_metric()
        metric_dict['average_Precision'] = precision
        metric_dict['average_Recall'] = recall
        metric_dict['average_F1'] = f1

        metric_dict['average_Accuracy'] = self.label_average_accuracy

        return metric_dict
