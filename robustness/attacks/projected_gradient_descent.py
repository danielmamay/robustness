import torch
import torch.nn.functional as F

class ProjectedGradientDescent:
    """
    Projected Gradient Descent (PGD) from 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    """
    def __init__(self, model, epsilon=0.03, alpha=0.2, steps=7, clamp_values=(0, 1)):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.clamp_min, self.clamp_max = clamp_values

    def generate(self, inputs, labels):

        adversarial_inputs = inputs.detach().clone()
        adversarial_inputs = inputs + torch.empty_like(inputs).uniform_(-self.epsilon, self.epsilon)
        adversarial_inputs = adversarial_inputs.clamp(self.clamp_min, self.clamp_max)

        for _ in range(self.steps):

            adversarial_inputs.requires_grad = True

            # Zero gradient buffers
            self.model.zero_grad()

            # Forward pass
            outputs = self.model(adversarial_inputs)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass
            loss.backward()

            # PGD
            sign_grad = adversarial_inputs.grad.data.sign()
            adversarial_inputs = adversarial_inputs.detach().clone() + self.alpha * sign_grad
            adversarial_inputs = adversarial_inputs.clamp(inputs - self.epsilon, inputs + self.epsilon)
            adversarial_inputs = adversarial_inputs.clamp(self.clamp_min, self.clamp_max)

            # Diff
            predicted_labels = self.model(adversarial_inputs).argmax(dim=1)
        
        return adversarial_inputs, predicted_labels
