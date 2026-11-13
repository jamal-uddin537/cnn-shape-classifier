"""Convolutional Neural Network (CNN) image classifier.

The model classifies 28x28 grayscale images into three shapes:
circle, square, and triangle. Images are drawn in code, so no download is needed.
"""
import torch
import torch.nn as nn

CLASSES = ["circle", "square", "triangle"]
SIZE = 28


def draw_shape(label, g):
    """Draw one random shape as a 28x28 image tensor."""
    ys, xs = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    r = torch.randint(5, 10, (1,), generator=g).item()
    cx = torch.randint(r, SIZE - r, (1,), generator=g).item()
    cy = torch.randint(r, SIZE - r, (1,), generator=g).item()

    if label == 0:    # circle
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
    elif label == 1:  # square
        mask = ((xs - cx).abs() <= r) & ((ys - cy).abs() <= r)
    else:             # triangle (pointing up)
        top, bottom = cy - r, cy + r
        rows = (ys - top).clamp(min=0).float()
        half_width = rows * r / (2 * r)
        mask = (ys >= top) & (ys <= bottom) & ((xs - cx).abs() <= half_width)

    img = mask.float() + 0.1 * torch.randn(SIZE, SIZE, generator=g)
    return img.clamp(0, 1)


def make_dataset(n_samples=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, len(CLASSES), (n_samples,), generator=g)
    images = torch.stack([draw_shape(int(l), g) for l in labels]).unsqueeze(1)
    return images, labels


class ShapeCNN(nn.Module):
    """Two conv blocks followed by a small classifier."""

    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train(epochs=8, batch_size=64, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    x, y = make_dataset(seed=seed)
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = ShapeCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            acc = (model(x_test).argmax(1) == y_test).float().mean().item()
        print(f"epoch {epoch:2d} | test accuracy {acc:.2%}")

    return model


if __name__ == "__main__":
    trained = train()
    torch.save(trained.state_dict(), "cnn_model.pt")
    print("Saved model to cnn_model.pt")
