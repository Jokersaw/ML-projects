import matplotlib.pyplot as plt
from realization import *

def count_transforms(layers):
    count = 0
    for layer in flatten_layers(layers):
        if hasattr(layer, "W") or hasattr(layer, "C"):
            count += 1
        else:
            raise ValueError(f"Unknown layer: {layer}")
    return count

def init_rbf_centers(layers, X_train):
    current_X = X_train

    for layer in layers:
        if isinstance(layer, RBFLayer):
            n_centers = layer.C.shape[0]

            idx = np.random.choice(current_X.shape[0], n_centers, replace=False)
            layer.C = current_X[idx].copy()

        current_X = layer.forward(current_X)

def run_experiment(model_name, layers, X_train, Y_train, X_test, Y_test,
                   lr=0.01, epochs=300, optimizer_cls=Adam, verbose=False):
    optimizer = optimizer_cls() if optimizer_cls is not None else None

    train_losses, test_losses = train_model(
        layers,
        X_train, Y_train,
        X_test, Y_test,
        lr=lr,
        epochs=epochs,
        verbose=verbose,
        optimizer=optimizer
    )

    result = {
        "name": model_name,
        "layers": layers,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_acc": accuracy(layers, X_train, Y_train),
        "test_acc": accuracy(layers, X_test, Y_test),
        "n_params": count_params(layers),
        "n_transforms": count_transforms(layers),
    }
    return result

def run_models_bunch(models, X_train, Y_train, X_test, Y_test, needsInitCenters = False):
    results = []
    for name, layers in models.items():
        if needsInitCenters:
            init_rbf_centers(layers, X_train) 
        results.append(
            run_experiment(
                name, layers,
                X_train, Y_train,
                X_test, Y_test,
                lr=0.01,
                epochs=500,
                optimizer_cls=Adam,
                verbose=False
            )
        )
    return results

def draw_learning_curve(results, model_type_name) :
    plt.figure(figsize=(12, 6))
    for r in results:
        plt.plot(r["train_losses"], label=f"{r['name']} train")
        plt.plot(r["test_losses"], linestyle="--", label=f"{r['name']} test")
    plt.title(f'Learning curves: {model_type_name}')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.show()

def plot_family(results, criteria, model_type_name):
    xs = [r[criteria] for r in results]
    ys = [r["test_acc"] for r in results]
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    plt.plot(xs, ys, marker="o", label=model_type_name)

def print_results(title, results):
    print(f"\n{title}")
    for r in results:
        print(
            f"{r['name']}: "
            f"params={r['n_params']}, "
            f"transforms={r['n_transforms']}, "
            f"train_acc={r['train_acc']:.3f}, "
            f"test_acc={r['test_acc']:.3f}"
        )

# Model dictionaries

input_dim = X_train.shape[1]
num_classes = Y_train.shape[1]

hidden = 32
centers = 64
gamma = 0.05

# Linear models

linear_models = {
    "linear_d1_id": [
        LinearLayer(input_dim, num_classes, Identity()),
    ],
    "linear_d2_relu": [
        LinearLayer(input_dim, hidden, ReLU()),
        LinearLayer(hidden, num_classes, Identity()),
    ],
    "linear_d3_relu": [
        LinearLayer(input_dim, hidden, ReLU()),
        LinearLayer(hidden, hidden, ReLU()),
        LinearLayer(hidden, num_classes, Identity()),
    ],
    "linear_d4_relu": [
        LinearLayer(input_dim, hidden, ReLU()),
        LinearLayer(hidden, hidden, ReLU()),
        LinearLayer(hidden, hidden, ReLU()),
        LinearLayer(hidden, num_classes, Identity()),
    ],
}

# Linear models with Residual

linear_residual_models = {
    "residual_1block_relu": [
        LinearLayer(input_dim, hidden, ReLU()),
        Residual([
            LinearLayer(hidden, hidden, ReLU()),
            LinearLayer(hidden, hidden, Identity()),
        ]),
        LinearLayer(hidden, num_classes, Identity()),
    ],

    "residual_2block_relu": [
        LinearLayer(input_dim, hidden, ReLU()),
        Residual([
            LinearLayer(hidden, hidden, ReLU()),
            LinearLayer(hidden, hidden, Identity()),
        ]),
        Residual([
            LinearLayer(hidden, hidden, ReLU()),
            LinearLayer(hidden, hidden, Identity()),
        ]),
        LinearLayer(hidden, num_classes, Identity()),
    ],

    "residual_3block_relu": [
        LinearLayer(input_dim, hidden, ReLU()),
        Residual([
            LinearLayer(hidden, hidden, ReLU()),
            LinearLayer(hidden, hidden, Identity()),
        ]),
        Residual([
            LinearLayer(hidden, hidden, ReLU()),
            LinearLayer(hidden, hidden, Identity()),
        ]),
        Residual([
            LinearLayer(hidden, hidden, ReLU()),
            LinearLayer(hidden, hidden, Identity()),
        ]),
        LinearLayer(hidden, num_classes, Identity()),
    ],
}

# RBF models

rbf_models = {
    "rbf_d2": [
        RBFLayer(input_dim, centers, gamma=gamma),
        LinearLayer(centers, num_classes, Identity()),
    ],
    "rbf_d3": [
        RBFLayer(input_dim, centers, gamma=gamma),
        RBFLayer(centers, centers, gamma=gamma),
        LinearLayer(centers, num_classes, Identity()),
    ],
    "rbf_d4": [
        RBFLayer(input_dim, centers, gamma=gamma),
        RBFLayer(centers, centers, gamma=gamma),
        RBFLayer(centers, centers, gamma=gamma),
        LinearLayer(centers, num_classes, Identity()),
    ],
}

# RBF models with Residual

rbf_residual_models = {
    "rbf_residual_1block": [
        RBFLayer(input_dim, centers, gamma=gamma),
        Residual([
            RBFLayer(centers, centers, gamma=gamma),
        ]),
        LinearLayer(centers, num_classes, Identity()),
    ],

    "rbf_residual_2block": [
        RBFLayer(input_dim, centers, gamma=gamma),
        Residual([
            RBFLayer(centers, centers, gamma=gamma),
        ]),
        Residual([
            RBFLayer(centers, centers, gamma=gamma),
        ]),
        LinearLayer(centers, num_classes, Identity()),
    ],
}

# Combinations of ordinary and RBF transforms
mixed_models = {
    "mixed_rbf_linear_relu": [
        RBFLayer(input_dim, centers, gamma=gamma),
        LinearLayer(centers, hidden, ReLU()),
        LinearLayer(hidden, num_classes, Identity()),
    ],

    "mixed_linear_rbf": [
        LinearLayer(input_dim, hidden, ReLU()),
        RBFLayer(hidden, centers, gamma=gamma),
        LinearLayer(centers, num_classes, Identity()),
    ],

    "mixed_linear_rbf_linear": [
        LinearLayer(input_dim, hidden, ReLU()),
        RBFLayer(hidden, centers, gamma=gamma),
        LinearLayer(centers, hidden, ReLU()),
        LinearLayer(hidden, num_classes, Identity()),
    ],
}

mixed_residual_models = {
    "mixed_rbf_residual_relu": [
        RBFLayer(input_dim, centers, gamma=gamma),
        Residual([
            LinearLayer(centers, centers, ReLU()),
            LinearLayer(centers, centers, Identity()),
        ]),
        LinearLayer(centers, num_classes, Identity()),
    ],

    "mixed_linear_rbf_residual": [
        LinearLayer(input_dim, hidden, ReLU()),
        RBFLayer(hidden, centers, gamma=gamma),
        Residual([
            LinearLayer(centers, centers, ReLU()),
            LinearLayer(centers, centers, Identity()),
        ]),
        LinearLayer(centers, num_classes, Identity()),
    ],
}

# Run all experiments

linear_results = run_models_bunch(linear_models, X_train, Y_train, X_test, Y_test)

linear_residual_results = run_models_bunch(linear_residual_models, X_train, Y_train, X_test, Y_test)

rbf_results = run_models_bunch(rbf_models, X_train, Y_train, X_test, Y_test, needsInitCenters=True)

rbf_residual_results = run_models_bunch(rbf_residual_models, X_train, Y_train, X_test, Y_test, needsInitCenters=True)

mixed_results = run_models_bunch(mixed_models, X_train, Y_train, X_test, Y_test, needsInitCenters=True)

mixed_residual_results = run_models_bunch(mixed_residual_models, X_train, Y_train, X_test, Y_test, needsInitCenters=True)

# Print summary

print_results("LINEAR MODELS", linear_results)
print_results("LINEAR RESIDUAL MODELS", linear_residual_results)
print_results("RBF MODELS", rbf_results)
print_results("RBF RESIDUAL MODELS", rbf_residual_results)
print_results("MIXED MODELS", mixed_results)
print_results("MIXED RESIDUAL MODELS", mixed_residual_results)

# Learning curves

draw_learning_curve(linear_results, "linear models")
draw_learning_curve(linear_residual_results, "linear residual models")
draw_learning_curve(rbf_results, "RBF models")
draw_learning_curve(rbf_residual_results, "RBF Residual models")
draw_learning_curve(mixed_results, "mixed models")
draw_learning_curve(mixed_residual_results, "mixed residual models")

# Accuracy vs number of parameters

plt.figure(figsize=(10, 6))
plot_family(linear_results, "n_params", "Linear")
plot_family(linear_residual_results, "n_params", "Linear+Residual")
plot_family(rbf_results, "n_params", "RBF")
plot_family(rbf_residual_results, "n_params", "RBF+Residual")
plot_family(mixed_results, "n_params", "Mixed")
plot_family(mixed_residual_results, "n_params", "Mixed+Residual")
plt.xlabel("Number of parameters")
plt.ylabel("Test accuracy")
plt.title("Test accuracy vs number of parameters")
plt.legend()
plt.grid(True)
plt.show()

# Accuracy vs number of transformations

plot_family(linear_results, "n_transforms", "Linear")
plot_family(linear_residual_results, "n_transforms", "Linear+Residual")
plot_family(rbf_results, "n_transforms", "RBF")
plot_family(rbf_residual_results, "n_transforms", "RBF+Residual")
plot_family(mixed_results, "n_transforms", "Mixed")
plot_family(mixed_residual_results, "n_transforms", "Mixed+Residual")
plt.xlabel("Number of transformations")
plt.ylabel("Test accuracy")
plt.title("Test accuracy vs number of transformations")
plt.legend()
plt.grid(True)
plt.show()
