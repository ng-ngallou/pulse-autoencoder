import logging

import matplotlib.pyplot as plt
import torch

from pulse_autoencoder.autoencoder.model.autoencoder_model import PulseAutoEncoder
from pulse_autoencoder.manipulate_data.load_data import prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

train_loader, test_loader = prepare_data("/data/mixed/mixed_df.pkl")
criterion = torch.nn.MSELoss()


def train_autoencoder() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"  # "mps" for mac

    model = PulseAutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 35
    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)  # noqa
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"epoch {epoch} loss {total_loss/len(train_loader)}")

    torch.save(model.state_dict(), "../model/mixed_data_autoencoder.pth")


def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PulseAutoEncoder()
    model.load_state_dict(
        torch.load(
            "/pulse_autoencoder/autoencoder/model/mixed_data_autoencoder.pth",
            map_location=device,
        )
    )

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)  # noqa
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            test_loss += loss.item()

    logger.info(f"Test Loss: {test_loss / len(test_loader):.6f}")

    with torch.no_grad():
        # Get first batch from test_loader
        batch = next(iter(test_loader))
        sample = batch[0].unsqueeze(0).to(device)

        reconstructed = model(sample).cpu()

    plt.figure()
    plt.plot(sample.squeeze().cpu().numpy(), label="original")
    plt.plot(reconstructed.squeeze().numpy(), label="reconstructed")
    plt.title("PulseAutoEncoder - evaluation")
    plt.legend()
    plt.show()
    plt.savefig("mixed_data_autoencoder_eval.png")


def main() -> None:
    train_autoencoder()
    evaluate()


if __name__ == "__main__":
    main()
