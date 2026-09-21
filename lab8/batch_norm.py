import torch

class BatchNormScratch:

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x, training=True):

        if training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            self.running_mean = ((1 - self.momentum) * self.running_mean
                + self.momentum * mean)

            self.running_var = ((1 - self.momentum) * self.running_var
                + self.momentum * var)

        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        y = self.gamma * x_hat + self.beta

        return y


x = torch.tensor([
    [2., 4., 6.],
    [8., 10., 12.],
    [14., 16., 18.]
])

bn = BatchNormScratch(num_features=3)

output = bn.forward(x)

print("Original:")
print(x)

print(f"After Batch Normalization:{output}")
