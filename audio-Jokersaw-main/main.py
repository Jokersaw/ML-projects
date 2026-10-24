import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, ASTModel

from utils import (
    load_audio_dataset,
    load_audio_from_hf,
    load_features,
    save_features,
    extract_mfcc_features,
    extract_ast_features_from_dataset_with_model,
    visualize_pca,
    find_closest_and_farthest_pairs,
    plot_mel_spectrogram,
)

from models import (
    evaluate_classical_models,
    AudioDatasetForAST,
    make_ast_collate_fn,
    ASTWithClassificationHead,
    train_one_epoch,
    evaluate_dl_model,
)

SEED = 42

DATASET_NAME = "capa2000/binary-classifier-birdnet"
LABEL_COLUMN = "nombreCientifico"

CACHE_DIR = "cache"

TARGET_SAMPLING_RATE = 16000
N_MFCC = 40

AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

BATCH_SIZE = 4
FROZEN_EPOCHS = 5
FINETUNE_EPOCHS = 1


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    os.makedirs(CACHE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nDevice:", device)

    # 1 Load dataset

    train_data, test_data, class_names, class_to_id = load_audio_dataset(
        dataset_name=DATASET_NAME, label_column=LABEL_COLUMN
    )

    print("\nClass mapping:")
    print(class_to_id)

    array, sampling_rate = load_audio_from_hf(train_data[0])

    print("\nSampling rate:", sampling_rate)
    print("Array shape:", array.shape)
    print("Array dtype:", array.dtype)

    # 2 MFCC features

    mfcc_cached = load_features("mfcc", cache_dir=CACHE_DIR)

    if mfcc_cached is not None:
        print("\nLoading MFCC features from cache...")
        X_train_mfcc, y_train, X_test_mfcc, y_test = mfcc_cached
    else:
        X_train_mfcc, y_train = extract_mfcc_features(
            dataset=train_data,
            dataset_name="train",
            label_column=LABEL_COLUMN,
            class_to_id=class_to_id,
            n_mfcc=N_MFCC,
            target_sampling_rate=TARGET_SAMPLING_RATE,
        )

        X_test_mfcc, y_test = extract_mfcc_features(
            dataset=test_data,
            dataset_name="test",
            label_column=LABEL_COLUMN,
            class_to_id=class_to_id,
            n_mfcc=N_MFCC,
            target_sampling_rate=TARGET_SAMPLING_RATE,
        )

        save_features(
            "mfcc", X_train_mfcc, y_train, X_test_mfcc, y_test, cache_dir=CACHE_DIR
        )

        print("\nMFCC features saved to cache.")

    print("\nFinal MFCC shapes:")
    print("X_train_mfcc:", X_train_mfcc.shape)
    print("y_train:", y_train.shape)
    print("X_test_mfcc:", X_test_mfcc.shape)
    print("y_test:", y_test.shape)

    # 3 MFCC PCA visualization

    visualize_pca(
        X_train_mfcc,
        y_train,
        X_test_mfcc,
        y_test,
        class_names,
        "MFCC feature space visualization using PCA",
    )

    # 4 Classical models on MFCC

    mfcc_results = evaluate_classical_models(
        X_train_mfcc, y_train, X_test_mfcc, y_test, class_names, "MFCC"
    )

    # 5 Pretrained AST features

    print("\nLoading AST model...")

    ast_feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
    ast_model = ASTModel.from_pretrained(AST_MODEL_NAME)

    ast_model = ast_model.to(device)
    ast_model.eval()

    print("AST model loaded")
    print("AST required sampling rate:", ast_feature_extractor.sampling_rate)

    ast_cached = load_features("ast", cache_dir=CACHE_DIR)

    if ast_cached is not None:
        print("\nLoading AST features from cache...")
        X_train_ast, y_train_ast, X_test_ast, y_test_ast = ast_cached
    else:
        X_train_ast, y_train_ast = extract_ast_features_from_dataset_with_model(
            backbone_model=ast_model,
            feature_extractor=ast_feature_extractor,
            dataset=train_data,
            dataset_name="train",
            label_column=LABEL_COLUMN,
            class_to_id=class_to_id,
            device=device,
            batch_size=BATCH_SIZE,
        )

        X_test_ast, y_test_ast = extract_ast_features_from_dataset_with_model(
            backbone_model=ast_model,
            feature_extractor=ast_feature_extractor,
            dataset=test_data,
            dataset_name="test",
            label_column=LABEL_COLUMN,
            class_to_id=class_to_id,
            device=device,
            batch_size=BATCH_SIZE,
        )

        save_features(
            "ast", X_train_ast, y_train_ast, X_test_ast, y_test_ast, cache_dir=CACHE_DIR
        )

        print("\nAST features saved to cache.")

    print("\nFinal AST shapes:")
    print("X_train_ast:", X_train_ast.shape)
    print("y_train_ast:", y_train_ast.shape)
    print("X_test_ast:", X_test_ast.shape)
    print("y_test_ast:", y_test_ast.shape)

    visualize_pca(
        X_train_ast,
        y_train_ast,
        X_test_ast,
        y_test_ast,
        class_names,
        "AST feature space visualization using PCA",
    )

    ast_results = evaluate_classical_models(
        X_train_ast, y_train_ast, X_test_ast, y_test_ast, class_names, "AST"
    )

    # 6 Frozen AST + classification head

    train_dataset = AudioDatasetForAST(
        hf_dataset=train_data,
        feature_extractor=ast_feature_extractor,
        label_column=LABEL_COLUMN,
        class_to_id=class_to_id,
    )

    test_dataset = AudioDatasetForAST(
        hf_dataset=test_data,
        feature_extractor=ast_feature_extractor,
        label_column=LABEL_COLUMN,
        class_to_id=class_to_id,
    )

    ast_collate_fn = make_ast_collate_fn(ast_feature_extractor)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ast_collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ast_collate_fn
    )

    num_classes = len(class_names)

    dl_model = ASTWithClassificationHead(
        backbone=ast_model, num_classes=num_classes
    ).to(device)

    # Freeze AST backbone
    for param in dl_model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        dl_model.classifier.parameters(), lr=1e-3, weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    print("\nTraining classification head on frozen AST...")

    for epoch in range(1, FROZEN_EPOCHS + 1):
        train_loss, _, train_f1 = train_one_epoch(
            model=dl_model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        test_loss, _, test_f1 = evaluate_dl_model(
            model=dl_model, dataloader=test_loader, criterion=criterion, device=device
        )

        print(
            f"Epoch {epoch}/{FROZEN_EPOCHS} | "
            f"train_loss={train_loss:.4f}, train_f1={train_f1:.4f} | "
            f"test_loss={test_loss:.4f}, test_f1={test_f1:.4f}"
        )

    frozen_head_test_loss, frozen_head_test_acc, frozen_head_test_f1 = (
        evaluate_dl_model(
            model=dl_model, dataloader=test_loader, criterion=criterion, device=device
        )
    )

    print("\nFrozen AST + classification head final result:")
    print("Test loss:", frozen_head_test_loss)
    print("Test accuracy:", frozen_head_test_acc)
    print("Test macro-F1:", frozen_head_test_f1)

    # 7 Fine-tuning AST + classification head

    print("\nFine-tuning AST backbone + classification head...")

    for param in dl_model.backbone.parameters():
        param.requires_grad = True

    optimizer_finetune = torch.optim.AdamW(
        [
            {"params": dl_model.backbone.parameters(), "lr": 1e-5},
            {"params": dl_model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        train_loss, _, train_f1 = train_one_epoch(
            model=dl_model,
            dataloader=train_loader,
            optimizer=optimizer_finetune,
            criterion=criterion,
            device=device,
        )

        test_loss, _, test_f1 = evaluate_dl_model(
            model=dl_model, dataloader=test_loader, criterion=criterion, device=device
        )

        print(
            f"Fine-tune epoch {epoch}/{FINETUNE_EPOCHS} | "
            f"train_loss={train_loss:.4f}, train_f1={train_f1:.4f} | "
            f"test_loss={test_loss:.4f}, test_f1={test_f1:.4f}"
        )

    finetuned_test_loss, finetuned_test_acc, finetuned_test_f1 = evaluate_dl_model(
        model=dl_model, dataloader=test_loader, criterion=criterion, device=device
    )

    print("\nFine-tuned AST + classification head final result:")
    print("Test loss:", finetuned_test_loss)
    print("Test accuracy:", finetuned_test_acc)
    print("Test macro-F1:", finetuned_test_f1)

    # Repeat 3 and 4

    finetuned_ast_cached = load_features("finetuned_ast", cache_dir=CACHE_DIR)

    if finetuned_ast_cached is not None:
        print("\nLoading fine-tuned AST features from cache...")
        (
            X_train_finetuned_ast,
            y_train_finetuned_ast,
            X_test_finetuned_ast,
            y_test_finetuned_ast,
        ) = finetuned_ast_cached
    else:
        X_train_finetuned_ast, y_train_finetuned_ast = (
            extract_ast_features_from_dataset_with_model(
                backbone_model=dl_model.backbone,
                feature_extractor=ast_feature_extractor,
                dataset=train_data,
                dataset_name="train",
                label_column=LABEL_COLUMN,
                class_to_id=class_to_id,
                device=device,
                batch_size=BATCH_SIZE,
            )
        )

        X_test_finetuned_ast, y_test_finetuned_ast = (
            extract_ast_features_from_dataset_with_model(
                backbone_model=dl_model.backbone,
                feature_extractor=ast_feature_extractor,
                dataset=test_data,
                dataset_name="test",
                label_column=LABEL_COLUMN,
                class_to_id=class_to_id,
                device=device,
                batch_size=BATCH_SIZE,
            )
        )

        save_features(
            "finetuned_ast",
            X_train_finetuned_ast,
            y_train_finetuned_ast,
            X_test_finetuned_ast,
            y_test_finetuned_ast,
            cache_dir=CACHE_DIR,
        )

        print("\nFine-tuned AST features saved to cache.")

    print("\nFinal fine-tuned AST shapes:")
    print("X_train_finetuned_ast:", X_train_finetuned_ast.shape)
    print("y_train_finetuned_ast:", y_train_finetuned_ast.shape)
    print("X_test_finetuned_ast:", X_test_finetuned_ast.shape)
    print("y_test_finetuned_ast:", y_test_finetuned_ast.shape)

    visualize_pca(
        X_train_finetuned_ast,
        y_train_finetuned_ast,
        X_test_finetuned_ast,
        y_test_finetuned_ast,
        class_names,
        "Fine-tuned AST feature space visualization using PCA",
    )

    finetuned_ast_results = evaluate_classical_models(
        X_train_finetuned_ast,
        y_train_finetuned_ast,
        X_test_finetuned_ast,
        y_test_finetuned_ast,
        class_names,
        "Fine-tuned AST",
    )

    # 8 Closest and farthest audio recordings

    best_X = X_test_finetuned_ast
    best_y = y_test_finetuned_ast
    best_dataset = test_data
    best_space_name = "Fine-tuned AST"

    closest_pair, farthest_pair = find_closest_and_farthest_pairs(best_X)

    print(f"\nBest feature space: {best_space_name}")

    print("\nClosest pair:")
    print("Indexes:", closest_pair[0], closest_pair[1])
    print("Distance:", closest_pair[2])
    print(
        "Classes:",
        class_names[best_y[closest_pair[0]]],
        "|",
        class_names[best_y[closest_pair[1]]],
    )

    print("\nFarthest pair:")
    print("Indexes:", farthest_pair[0], farthest_pair[1])
    print("Distance:", farthest_pair[2])
    print(
        "Classes:",
        class_names[best_y[farthest_pair[0]]],
        "|",
        class_names[best_y[farthest_pair[1]]],
    )

    closest_i, closest_j, _ = closest_pair

    plot_mel_spectrogram(
        best_dataset[closest_i],
        f"Closest audio 1 | index={closest_i} | class={class_names[best_y[closest_i]]}",
    )

    plot_mel_spectrogram(
        best_dataset[closest_j],
        f"Closest audio 2 | index={closest_j} | class={class_names[best_y[closest_j]]}",
    )

    farthest_i, farthest_j, _ = farthest_pair

    plot_mel_spectrogram(
        best_dataset[farthest_i],
        f"Farthest audio 1 | index={farthest_i} | class={class_names[best_y[farthest_i]]}",
    )

    plot_mel_spectrogram(
        best_dataset[farthest_j],
        f"Farthest audio 2 | index={farthest_j} | class={class_names[best_y[farthest_j]]}",
    )

    # Summary

    print("\nAll experiments finished.")

    print("\nMFCC results:")
    print(mfcc_results)

    print("\nAST results:")
    print(ast_results)

    print("\nFine-tuned AST results:")
    print(finetuned_ast_results)


if __name__ == "__main__":
    main()
