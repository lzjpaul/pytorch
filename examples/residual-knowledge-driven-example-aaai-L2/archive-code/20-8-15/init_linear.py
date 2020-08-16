import torch.nn as nn
import math

class InitLinear(nn.Linear):
    """Applies a linear transformation to the incoming data: :math:`y = Ax + b`
       Inherit Linear layer, but different initialziation!
    """
    def reset_parameters(self):
        avg = 2
        stdv = math.sqrt(2.0 * avg / (self.weight.size(1) + self.weight.size(0)))
        self.weight.data.normal_(mean=0, std=stdv)
        if self.bias is not None:
            self.bias.data.zero_() # Fills self tensor with zeros
