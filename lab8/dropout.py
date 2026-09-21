import torch


def dropout(x, p=0.5, training=True):

    if not training:
        return x

    if p < 0 or p >= 1:
        raise ValueError("p must be between 0 and 1")

    random_values = torch.rand_like(x)

    mask = (random_values > p).float()

    output = x * mask

    output = output / (1 - p)

    return output


x = torch.tensor([
    [2., 4., 6., 8.],
    [10., 12., 14., 16.]
])

print("Original:")
print(x)

print(f"After Dropout:{dropout(x, p=0.5)}")
