import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from model2 import Correncoder_model  # import your model class

def run_ensemble_test(testX, testy, fold_test_losses, model_folder='.', num_samples=5, device='cpu', num_folds=5):
    testX_flat = testX.reshape(-1, testX.shape[-1])[:, np.newaxis, :]
    testy_flat = testy.reshape(-1, testy.shape[-1])

    print(f"Total test samples: {testX_flat.shape[0]}")

    # random_indices = random.sample(range(testX_flat.shape[0]), num_samples)
    random_indices = [1480, 3215, 2457, 12113, 4959, 13782, 1305, 12241, 3010, 4946]
    print(f"Randomly selected test sample indices: {random_indices}")

    test_samples = torch.from_numpy(testX_flat[random_indices].astype(np.float32)).to(device)

    all_predictions = []
    for fold in range(1, num_folds + 1):
        print(f"Loading model fold {fold}...")
        model = Correncoder_model().to(device)
        model.load_state_dict(torch.load(f"{model_folder}/best_model_fold_{fold}.pth"))
        model.eval()
        with torch.no_grad():
            preds = model(test_samples).cpu().numpy()
            all_predictions.append(preds)
        print(f"Fold {fold} prediction shape: {preds.shape}")

    ensemble_preds = np.mean(all_predictions, axis=0)
    print(f"Ensemble prediction shape: {ensemble_preds.shape}")

    # Find indices of 3 best folds (lowest test losses)
    best_3_indices = np.argsort(fold_test_losses)[:3]
    print(f"Best 3 folds based on test loss: {best_3_indices}")

    colors = ['green', 'orange', 'purple']
    labels = [f'Best Model {i + 1}' for i in range(3)]

    # Plot ensemble + best 3 models + ground truth
    for i, idx in enumerate(random_indices):
        plt.figure(figsize=(12, 3))
        plt.plot(ensemble_preds[i][0], label='Ensemble Prediction', color='blue', linewidth=2)
        for rank, model_idx in enumerate(best_3_indices):
            plt.plot(all_predictions[model_idx][i][0], label=labels[rank], color=colors[rank], linestyle='--')
        plt.plot(testy_flat[idx], label='Ground Truth', color='red', linestyle=':')
        plt.title(f"Test Sample Index {idx}")
        plt.xlabel("Time Steps")
        plt.ylabel("Normalized Amplitude")
        plt.legend()
        plt.show()

    return ensemble_preds, random_indices


if __name__ == "__main__":
    device = torch.device("mps")  # or "cuda" or "cpu"
    # Load preprocessed test data
    data = np.load('test_data.npz')
    testX = data['testX']
    testy = data['testy']

    # Load fold test losses saved previously
    fold_test_losses = np.load('fold_test_losses.npy')

    # Set these according to your setup
    model_folder = '.'  # folder with saved best_model_fold_*.pth files
    num_folds = 19       # number of saved folds
    num_samples = 20     # how many random samples to plot

    run_ensemble_test(testX, testy, fold_test_losses, model_folder, num_samples, device, num_folds)