"""Fast Gradient Sign Method (FGSM) adversarial attack.

Fast Gradient Sign Method from 'Explaining and Harnessing Adversarial Examples'
[https://arxiv.org/abs/1412.6572]

    Typical usage example:

    attack = FastGradient(classifier, epsilon=0.2)
    adversarial_inputs, predicted_labels = attack.generate(inputs, labels)
"""
from typing import Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F

class FastGradient:
    """Fast Gradient Sign Method adversarial attack with L_inf distance measure."""

    def __init__(
        self,
        classifier: torch.nn.Module,
        epsilon: Union[int, float],
        clamp_values: Optional[tuple[Union[int, float], Union[int, float]]] = (0, 1)
    ) -> None:
        """
        Args:
            classifier: A trained classifier.
            epsilon: Attack step size.
            clamp_values: Valid input range.
        """
        self.classifier = classifier
        self.epsilon = epsilon
        self.clamp_min, self.clamp_max = clamp_values

    def generate(self, inputs: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            inputs: A tensor with shape [N, C, H, W].
            labels: A tensor with shape [N].
        """
        inputs.requires_grad = True

        # Zero gradient buffers
        self.classifier.zero_grad()

        # Forward pass
        outputs = self.classifier(inputs)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass
        loss.backward()

        # FGSM
        sign_inputs_grad = inputs.grad.data.sign()
        adversarial_inputs = inputs + self.epsilon * sign_inputs_grad
        adversarial_inputs = adversarial_inputs.clamp(self.clamp_min, self.clamp_max)

        adversarial_predicted_labels = self.classifier(adversarial_inputs).argmax(dim=1)

        return adversarial_inputs, adversarial_predicted_labels
