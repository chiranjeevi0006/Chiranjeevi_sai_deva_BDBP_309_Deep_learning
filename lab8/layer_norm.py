import torch


class LayerNormScratch:

    def __init__(self, num_features, eps=1e-5):
        self.eps = eps

        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(
            dim=1,
            keepdim=True,
            unbiased=False
        )
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.gamma * x_hat + self.beta

        return y

x = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])

ln = LayerNormScratch(num_features=3)

output = ln.forward(x)

print("Original:")
print(x)

print(f"After Layer Normalization:{output}")
