import os
import io

import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset, Audio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def load_audio_dataset(dataset_name, label_column):
    dataset = load_dataset(dataset_name)

    train_data = dataset["train"].cast_column("audio", Audio(decode=False))
    test_data = dataset["test"].cast_column("audio", Audio(decode=False))

    class_names = sorted(set(train_data[label_column]))
    class_to_id = {name: i for i, name in enumerate(class_names)}

    return train_data, test_data, class_names, class_to_id


def load_audio_from_hf(example):
    audio = example["audio"]

    if audio.get("bytes") is not None:
        array, sampling_rate = sf.read(io.BytesIO(audio["bytes"]))
    else:
        raise ValueError("В audio нет bytes")

    return array, sampling_rate


def prepare_audio(array, sampling_rate, target_sampling_rate=16000):
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 2:
        array = array.mean(axis=1)

    if sampling_rate != target_sampling_rate:
        array = librosa.resample(
            array, orig_sr=sampling_rate, target_sr=target_sampling_rate
        )

    return array, target_sampling_rate


def save_features(prefix, X_train, y_train, X_test, y_test, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)

    np.save(os.path.join(cache_dir, f"{prefix}_X_train.npy"), X_train)
    np.save(os.path.join(cache_dir, f"{prefix}_y_train.npy"), y_train)
    np.save(os.path.join(cache_dir, f"{prefix}_X_test.npy"), X_test)
    np.save(os.path.join(cache_dir, f"{prefix}_y_test.npy"), y_test)


def load_features(prefix, cache_dir="cache"):
    X_train_path = os.path.join(cache_dir, f"{prefix}_X_train.npy")
    y_train_path = os.path.join(cache_dir, f"{prefix}_y_train.npy")
    X_test_path = os.path.join(cache_dir, f"{prefix}_X_test.npy")
    y_test_path = os.path.join(cache_dir, f"{prefix}_y_test.npy")

    paths = [X_train_path, y_train_path, X_test_path, y_test_path]

    if all(os.path.exists(path) for path in paths):
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

        return X_train, y_train, X_test, y_test

    return None


def extract_mfcc_from_example(example, n_mfcc=40, target_sampling_rate=16000):
    array, sampling_rate = load_audio_from_hf(example)

    array, sampling_rate = prepare_audio(
        array, sampling_rate, target_sampling_rate=target_sampling_rate
    )

    mfcc = librosa.feature.mfcc(y=array, sr=sampling_rate, n_mfcc=n_mfcc)

    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std])

    return features


def extract_mfcc_features(
    dataset,
    dataset_name,
    label_column,
    class_to_id,
    n_mfcc=40,
    target_sampling_rate=16000,
):
    X = []
    y = []

    print(f"\nExtracting MFCC features from {dataset_name}...")

    for i, example in enumerate(dataset):
        features = extract_mfcc_from_example(
            example, n_mfcc=n_mfcc, target_sampling_rate=target_sampling_rate
        )

        label_name = example[label_column]
        label_id = class_to_id[label_name]

        X.append(features)
        y.append(label_id)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(dataset)}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"{dataset_name} MFCC X shape:", X.shape)
    print(f"{dataset_name} MFCC y shape:", y.shape)

    return X, y


def extract_ast_features_from_dataset_with_model(
    backbone_model,
    feature_extractor,
    dataset,
    dataset_name,
    label_column,
    class_to_id,
    device,
    batch_size=4,
):
    X = []
    y = []

    backbone_model.eval()

    print(f"\nExtracting AST features from {dataset_name}...")

    batch_audio = []
    batch_labels = []

    target_sampling_rate = feature_extractor.sampling_rate

    for i, example in enumerate(dataset):
        array, sampling_rate = load_audio_from_hf(example)

        array, sampling_rate = prepare_audio(
            array, sampling_rate, target_sampling_rate=target_sampling_rate
        )

        label_name = example[label_column]
        label_id = class_to_id[label_name]

        batch_audio.append(array)
        batch_labels.append(label_id)

        is_batch_ready = len(batch_audio) == batch_size
        is_last_item = i == len(dataset) - 1

        if is_batch_ready or is_last_item:
            inputs = feature_extractor(
                batch_audio,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
                padding=True,
            )

            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = backbone_model(**inputs)

            embeddings = outputs.last_hidden_state.mean(dim=1)

            X.append(embeddings.cpu().numpy())
            y.extend(batch_labels)

            batch_audio = []
            batch_labels = []

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(dataset)}")

    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"{dataset_name} AST X shape:", X.shape)
    print(f"{dataset_name} AST y shape:", y.shape)

    return X, y


def visualize_pca(X_train, y_train, X_test, y_test, class_names, title):
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_all_scaled)

    print("\nPCA explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("Total explained variance:", pca.explained_variance_ratio_.sum())

    print("X_all shape:", X_all.shape)
    print("X_all_scaled shape:", X_all_scaled.shape)
    print("X_2d shape:", X_2d.shape)
    print("Unique labels:", np.unique(y_all))

    plt.figure(figsize=(8, 6))

    for class_id, class_name in enumerate(class_names):
        mask = y_all == class_id

        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=class_name, alpha=0.7)

    plt.title(title)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return X_2d


def plot_mel_spectrogram(example, title, target_sampling_rate=16000):
    array, sampling_rate = load_audio_from_hf(example)

    array, sampling_rate = prepare_audio(
        array, sampling_rate, target_sampling_rate=target_sampling_rate
    )

    mel_spectrogram = librosa.feature.melspectrogram(
        y=array, sr=sampling_rate, n_mels=128
    )

    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mel_spectrogram_db, sr=sampling_rate, x_axis="time", y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def find_closest_and_farthest_pairs(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    distances = pairwise_distances(X_scaled, metric="cosine")

    row_indices, col_indices = np.triu_indices_from(distances, k=1)
    pair_distances = distances[row_indices, col_indices]

    closest_position = np.argmin(pair_distances)
    farthest_position = np.argmax(pair_distances)

    closest_i = row_indices[closest_position]
    closest_j = col_indices[closest_position]
    closest_distance = pair_distances[closest_position]

    farthest_i = row_indices[farthest_position]
    farthest_j = col_indices[farthest_position]
    farthest_distance = pair_distances[farthest_position]

    closest_pair = (closest_i, closest_j, closest_distance)
    farthest_pair = (farthest_i, farthest_j, farthest_distance)

    return closest_pair, farthest_pair
