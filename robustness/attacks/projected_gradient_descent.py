"""Projected Gradient Descent (PGD) adversarial attack.

Projected Gradient Descent from 'Towards Deep Learning Models Resistant to Adversarial Attacks'
[https://arxiv.org/abs/1706.06083]

    Typical usage example:

    attack = ProjectedGradientDescent(classifier, epsilon=0.03, alpha=0.2, steps=7)
    adversarial_inputs, predicted_labels = attack.generate(inputs, labels)
"""
from typing import Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F

class ProjectedGradientDescent:
    """Projected Gradient Descent adversarial attack with L_inf distance measure."""

    def __init__(
        self,
        classifier: torch.nn.Module,
        epsilon: Union[int, float],
        alpha: Union[int, float],
        steps: int,
        clamp_values: Optional[tuple[int, int]] = (0, 1)
        ) -> None:
        """
        Args:
            classifier: A trained classifier.
            epsilon: Maximum pertubation distance.
            alpha: Attack step size.
            steps: Number of steps.
            clamp_values: Valid input range.
        """
        self.classifier = classifier
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.clamp_min, self.clamp_max = clamp_values

    def generate(self, inputs: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            inputs: A tensor with shape [N, C, H, W].
            labels: A tensor with shape [N].
        """

        adversarial_inputs = inputs.detach().clone()
        adversarial_inputs = inputs + torch.empty_like(inputs).uniform_(-self.epsilon, self.epsilon)
        adversarial_inputs = adversarial_inputs.clamp(self.clamp_min, self.clamp_max)

        for _ in range(self.steps):

            adversarial_inputs.requires_grad = True

            # Zero gradient buffers
            self.classifier.zero_grad()

            # Forward pass
            outputs = self.classifier(adversarial_inputs)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass
            loss.backward()

            # PGD
            sign_grad = adversarial_inputs.grad.data.sign()
            adversarial_inputs = adversarial_inputs.detach().clone() + self.alpha * sign_grad
            adversarial_inputs = adversarial_inputs.clamp(inputs - self.epsilon, inputs + self.epsilon)
            adversarial_inputs = adversarial_inputs.clamp(self.clamp_min, self.clamp_max)

            adversarial_predicted_labels = self.classifier(adversarial_inputs).argmax(dim=1)

        return adversarial_inputs, adversarial_predicted_labels
