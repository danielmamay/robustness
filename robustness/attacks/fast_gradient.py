import torch.nn.functional as F

class FastGradient:
    """
    Fast Gradient Sign Method (FGSM) from 'Explaining and Harnessing Adversarial Examples'
    [https://arxiv.org/abs/1412.6572]
    """
    def __init__(self, model, epsilon=0.007, clamp_values=(0, 1)):
        self.model = model
        self.epsilon = epsilon
        self.clamp_min, self.clamp_max = clamp_values

    def generate(self, inputs, labels):
        inputs.requires_grad = True

        # Zero gradient buffers
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass
        loss.backward()

        # FGSM
        sign_inputs_grad = inputs.grad.data.sign()
        adversarial_inputs = inputs + self.epsilon * sign_inputs_grad
        adversarial_inputs = adversarial_inputs.clamp(self.clamp_min, self.clamp_max)

        # Diff
        predicted_labels = self.model(adversarial_inputs).argmax(dim=1)
        
        return adversarial_inputs, predicted_labels
