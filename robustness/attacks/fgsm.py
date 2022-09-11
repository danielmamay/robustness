class FGSM:
    """
    Fast Gradient Sign Method (FGSM) from 'Explaining and Harnessing Adversarial Examples'
    [https://arxiv.org/abs/1412.6572]
    """
    def __init__(self, epsilon=0.007, clamp_values=(0, 1)):
        self.epsilon = epsilon
        self.clamp_min, self.clamp_max = clamp_values

    def generate(self, model, criterion, inputs, labels):
        inputs.requires_grad = True

        # Zero gradient buffers
        model.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # FGSM
        sign_inputs_grad = inputs.grad.data.sign()
        adversarial_inputs = inputs + self.epsilon * sign_inputs_grad
        adversarial_inputs = adversarial_inputs.clamp(self.clamp_min, self.clamp_max)

        # Diff
        predicted_labels = model(adversarial_inputs).argmax(dim=1)
        
        return adversarial_inputs, predicted_labels
