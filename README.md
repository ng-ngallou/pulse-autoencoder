# pulse-autoencoder

Unsupervised RF pulse anomaly detection using a 1D convolutional autoencoder. Trains on unlabeled, mostly-normal RF voltage pulses and flags anomalies (voltage breakdowns) by thresholding reconstruction error.

> Trained model weights are excluded from the repo.

## Install

```bash
git clone https://github.com/ng-ngallou/pulse-autoencoder.git
cd pulse-autoencoder
pip install -e .
```

## Usage
#### Prepare data
python -m pulse_autoencoder.manipulate_data.load_data --data-dir ./data --output ./data/mixed/mixed_df.pkl

#### Train
train_autoencoder --data ./data/mixed/mixed_df.pkl

#### Classify pulses
python -m pulse_autoencoder.autoencoder.pipeline.predict --data ./data/mixed/mixed_df.pkl --model <path_to.pth>


## Example Output
| Error Distribution | Reconstructed Bad Pulses |
|---|---|
| ![Error distribution](plots/error_distribution.png) | ![Bad pulses](plots/reconstructed_bad_pulses.png) |

## Dependencies

- **torch** — model definition, training, inference
- **numpy**, **pandas** — data handling
- **scikit-learn** — train/test splitting
- **scipy** — signal resampling (interpolation)
- **matplotlib** — visualization
