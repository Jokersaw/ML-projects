# Results

## Feature shapes

| feature space | train shape | test shape |
|---|---:|---:|
| MFCC | (320, 80) | (80, 80) |
| AST | (320, 768) | (80, 768) |
| Fine-tuned AST | (320, 768) | (80, 768) |

## MFCC

PCA explained variance ratio: `[0.30868453, 0.22433507]`  
Total explained variance: `0.5330196`

| index | feature_space | model | accuracy | f1_macro |
|---:|---|---|---:|---:|
| 2 | MFCC | Random Forest | 0.9750 | 0.974603 |
| 1 | MFCC | SVM RBF | 0.9625 | 0.962019 |
| 0 | MFCC | Logistic Regression | 0.9375 | 0.937018 |
| 3 | MFCC | KNN | 0.9000 | 0.899937 |

## AST

PCA explained variance ratio: `[0.13327892, 0.10646404]`  
Total explained variance: `0.23974296`

| index | feature_space | model | accuracy | f1_macro |
|---:|---|---|---:|---:|
| 0 | AST | Logistic Regression | 0.9875 | 0.987340 |
| 2 | AST | Random Forest | 0.9875 | 0.987340 |
| 1 | AST | SVM RBF | 0.9750 | 0.974603 |
| 3 | AST | KNN | 0.9500 | 0.949717 |

## Frozen AST + classification head

| metric | value |
|---|---:|
| Test loss | 0.14198683034628629 |
| Test accuracy | 0.975 |
| Test macro-F1 | 0.9747474747474747 |

## Fine-tuned AST + classification head

| metric | value |
|---|---:|
| Test loss | 0.09747316810689881 |
| Test accuracy | 0.9875 |
| Test macro-F1 | 0.9873397689507833 |

## Fine-tuned AST features

PCA explained variance ratio: `[0.28858367, 0.06646151]`  
Total explained variance: `0.35504517`

| index | feature_space | model | accuracy | f1_macro |
|---:|---|---|---:|---:|
| 0 | Fine-tuned AST | Logistic Regression | 0.9875 | 0.98734 |
| 1 | Fine-tuned AST | SVM RBF | 0.9875 | 0.98734 |
| 2 | Fine-tuned AST | Random Forest | 0.9875 | 0.98734 |
| 3 | Fine-tuned AST | KNN | 0.9875 | 0.98734 |

## Best feature space

`Fine-tuned AST`

## Closest pair

| property | value |
|---|---|
| Indexes | 20, 47 |
| Distance | 0.06380975 |
| Classes | Acropternis orthonyx / Acropternis orthonyx |

## Farthest pair

| property | value |
|---|---|
| Indexes | 6, 22 |
| Distance | 1.5857309 |
| Classes | Acropternis orthonyx / Accipiter striatus |
