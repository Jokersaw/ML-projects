import torch
from torch import nn
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

from utils import load_audio_from_hf, prepare_audio


def evaluate_classical_models(
    X_train, y_train, X_test, y_test, class_names, feature_space_name
):
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
            ]
        ),
        "SVM RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", random_state=42)),
            ]
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }

    results = []

    print(f"\nEvaluating classical models on {feature_space_name} features...")

    for model_name, model in models.items():
        print(f"\nModel: {model_name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        print("Accuracy:", accuracy)
        print("Macro-F1:", f1_macro)

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        results.append(
            {
                "feature_space": feature_space_name,
                "model": model_name,
                "accuracy": accuracy,
                "f1_macro": f1_macro,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_macro", ascending=False)

    print(f"\nSummary for {feature_space_name}:")
    print(results_df)

    return results_df


class AudioDatasetForAST(Dataset):

    def __init__(self, hf_dataset, feature_extractor, label_column, class_to_id):
        self.hf_dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.label_column = label_column
        self.class_to_id = class_to_id

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]

        array, sampling_rate = load_audio_from_hf(example)

        array, sampling_rate = prepare_audio(
            array,
            sampling_rate,
            target_sampling_rate=self.feature_extractor.sampling_rate,
        )

        label_name = example[self.label_column]
        label_id = self.class_to_id[label_name]

        return array, label_id


def make_ast_collate_fn(feature_extractor):

    def ast_collate_fn(batch):
        audios = []
        labels = []

        for audio, label in batch:
            audios.append(audio)
            labels.append(label)

        inputs = feature_extractor(
            audios,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        inputs["labels"] = torch.tensor(labels, dtype=torch.long)

        return inputs

    return ast_collate_fn


class ASTWithClassificationHead(nn.Module):

    def __init__(self, backbone, num_classes):
        super().__init__()

        self.backbone = backbone

        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden_size, num_classes)
        )

    def forward(self, **inputs):
        outputs = self.backbone(**inputs)

        # outputs.last_hidden_state:
        # batch_size × sequence_length × hidden_size
        pooled = outputs.last_hidden_state.mean(dim=1)

        logits = self.classifier(pooled)

        return logits


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_idx, batch in enumerate(dataloader):
        labels = batch.pop("labels").to(device)

        inputs = {key: value.to(device) for key, value in batch.items()}

        logits = model(**inputs)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(
                f"  batch {batch_idx + 1}/{len(dataloader)} | "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average="macro")

    return avg_loss, accuracy, f1_macro


def evaluate_dl_model(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)

            inputs = {key: value.to(device) for key, value in batch.items()}

            logits = model(**inputs)

            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average="macro")

    return avg_loss, accuracy, f1_macro
