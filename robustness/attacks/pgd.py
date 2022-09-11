import torch

class PGD:
    """
    Projected Gradient Descent (PGD) from 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    """
    def __init__(self, epsilon=0.03, alpha=0.2, steps=7, clamp_values=(0, 1)):
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.clamp_min, self.clamp_max = clamp_values

    def generate(self, model, criterion, inputs, labels):

        adversarial_inputs = inputs.detach().clone()
        adversarial_inputs = inputs + torch.empty_like(inputs).uniform_(-self.epsilon, self.epsilon)
        adversarial_inputs = adversarial_inputs.clamp(self.clamp_min, self.clamp_max)

        for _ in range(self.steps):

            adversarial_inputs.requires_grad = True

            # Zero gradient buffers
            model.zero_grad()

            # Forward pass
            outputs = model(adversarial_inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # PGD
            sign_grad = adversarial_inputs.grad.data.sign()
            adversarial_inputs = adversarial_inputs.detach().clone() + self.alpha * sign_grad
            adversarial_inputs = adversarial_inputs.clamp(inputs - self.epsilon, inputs + self.epsilon)
            adversarial_inputs = adversarial_inputs.clamp(self.clamp_min, self.clamp_max)

            # Diff
            predicted_labels = model(adversarial_inputs).argmax(dim=1)
        
        return adversarial_inputs, predicted_labels
