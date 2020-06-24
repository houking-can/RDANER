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
from scibert.models.text_classifier import TextClassifier


@Model.register("bert_text_classifier")
class BertTextClassifier(TextClassifier):
    """
    Implements a basic text classifier:
    1) Embed tokens using `text_field_embedder`
    2) Get the CLS token
    3) Final feedforward layer

    Optimized with CrossEntropyLoss.  Evaluated with CategoricalAccuracy & F1.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 verbose_metrics: False,
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 ) -> None:
        super(TextClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.dropout = torch.nn.Dropout(dropout)
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.classifier_feedforward = torch.nn.Linear(self.text_field_embedder.get_output_dim(), self.num_classes)

        self.label_auc_global = Auc(positive_label=1)
        self.verbose_metrics = verbose_metrics

        self.label_f1_global = F1Measure(positive_label=1)
        self.label_average_accuracy = 0

        self.label_auc_local = {}
        for i in range(self.num_classes):
            self.label_auc_local[vocab.get_token_from_index(index=i, namespace="labels")] = Auc(positive_label=1)

        # self.label_f1_local = {}
        # for i in range(self.num_classes):
        #     self.label_f1_local[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=1)

        self.loss = torch.nn.BCEWithLogitsLoss()

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
        pooled = self.dropout(embedded_text[:, 0, :])
        logits = self.classifier_feedforward(pooled)
        class_probs = torch.sigmoid(logits)
        output_dict = {"logits": logits}
        if label is not None:
            loss = self.loss(logits, label.float())
            output_dict["loss"] = loss

            # compute F1 per label
            combine = torch.cat((1 - class_probs.unsqueeze(-1), class_probs.unsqueeze(-1)), -1)
            self.label_f1_global(combine, label)

            class_probs[class_probs > 0.5] = 1
            class_probs[class_probs <= 0.5] = 0
            class_probs = class_probs.int()
            # compute Auc per label
            for i in range(self.num_classes):
                metric = self.label_auc_local[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(class_probs[:, i].view(-1), label[:, i].view(-1))
            self.label_auc_global(class_probs.view(-1), label.view(-1))
            self.label_average_accuracy = float(sum(class_probs.view(-1) == label.view(-1))) / class_probs.numel()

        return output_dict
