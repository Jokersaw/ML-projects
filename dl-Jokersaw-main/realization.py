import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(21)

X, y = load_digits(return_X_y=True)
# X (1797, 64) - пиксели
# y (1797,) - цифры от 0 до 9

# One-hot
num_classes = len(np.unique(y))
Y = np.eye(num_classes)[y]

# Normalization
scaler = StandardScaler()
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=21, stratify=y
)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


class LinearLayer:
    def __init__(self, in_features, out_features, activation):
        # Размер матрицы весов: (in_features, out_features)
        # ReLU - He initialization
        # Tanh/Id - Xavier initialization
        if isinstance(activation, ReLU):
            self.W = np.random.randn(in_features, out_features) * np.sqrt(
                2.0 / in_features
            )
        else:
            self.W = np.random.randn(in_features, out_features) * np.sqrt(
                1.0 / in_features
            )

        # Смещения инициализируем нулями
        self.b = np.zeros(out_features)
        # Функция активации слоя
        self.activation = activation

    def forward(self, x):
        self.x = x  # (N, in)
        self.z = x @ self.W + self.b  # (N, in) @ (in, out) -> (N, out)
        return self.activation.forward(self.z)  # вычисляем функцию активации

    def backward(self, grad_out):
        # grad_out = dL/dy
        grad_z = self.activation.backward(grad_out)  # dL/dz

        # градиенты по параметрам, N - batch_size, в данном случае весь датасет
        self.dW = self.x.T @ grad_z  # (in, N) @ (N, out) -> (in, out)
        self.db = np.sum(grad_z, axis=0)  # (out,)

        # градиент по входу слоя
        grad_x = grad_z @ self.W.T  # (N, out) @ (out, in) -> (N, in)
        return grad_x


class Identity:
    def forward(self, x):
        self.x = x
        return x

    def backward(self, grad):
        return grad


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * self.mask


class Tanh:
    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad):
        return grad * (1 - self.y**2)


class Residual:
    # Residual wrapper: y = x + F(x)
    # F(x) задаётся списком слоёв self.layers.

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        self.x = x
        out = x
        for layer in self.layers:
            out = layer.forward(out)

        if out.shape != x.shape:
            raise ValueError(
                f"ResidualClassError: "
                f"input shape {x.shape} != output shape {out.shape}"
            )
        self.out = out

        return self.x + self.out

    def backward(self, grad_out):
        # grad_out = dL/dy
        # y = x + out => dL/dx = grad_out (через skip) + dL/dout * dout/dx
        grad = grad_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad_out + grad


def flatten_layers(layers):
    flat = []
    for layer in layers:
        if isinstance(layer, Residual):
            flat.extend(flatten_layers(layer.layers))
        else:
            flat.append(layer)
    return flat


class SoftArgMaxCrossEntropy:
    # logits: (N, K) - выходы модели (логиты), любые действительные числа
    # target_one_hot: (N, K) - one-hot кодировка правильных классов
    # N - количество объектов (batch size)
    # K - количество классов
    def forward(self, logits, target_one_hot):
        self.target = target_one_hot

        # вычитаем максимум из каждой строки logits
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        sum_exp = np.sum(exp, axis=1, keepdims=True)

        # вероятности классов (softmax)
        self.probs = exp / sum_exp
        log_probs = shifted - np.log(sum_exp)

        # cross-entropy - выбираем log вероятности правильных классов
        loss = -np.mean(np.sum(target_one_hot * log_probs, axis=1))
        return loss

    def backward(self):
        # dL/dlogits = (probs - target) / N
        N = self.target.shape[0]
        return (self.probs - self.target) / N


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m = {}  # первый момент
        self.v = {}  # второй момент

    def begin_step(self):
        self.t += 1

    def step(self, param, grad, key):
        # param: параметр
        # grad:  градиент
        # key:   уникальный ключ параметра как строка
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad**2)

        # корректировка смещения
        m_hat = self.m[key] / (1 - self.beta1**self.t)
        v_hat = self.v[key] / (1 - self.beta2**self.t)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def forward_model(layers, X):
    out = X
    for layer in layers:
        out = layer.forward(out)
    return out


def backward_model(layers, grad):
    for layer in reversed(layers):
        grad = layer.backward(grad)


def train_model(
    layers,
    X_train,
    Y_train,
    X_test,
    Y_test,
    lr=0.05,
    epochs=500,
    verbose=False,
    optimizer=None,
):

    train_loss_fn = SoftArgMaxCrossEntropy()
    test_loss_fn = SoftArgMaxCrossEntropy()
    train_losses = []
    test_losses = []

    for epoch in range(epochs):

        # forward
        train_logits = forward_model(layers, X_train)
        train_loss = train_loss_fn.forward(train_logits, Y_train)
        train_losses.append(train_loss)

        # backward
        grad = train_loss_fn.backward()
        backward_model(layers, grad)

        # update
        if optimizer is None:
            for layer in flatten_layers(layers):
                if hasattr(layer, "W"):
                    layer.W -= lr * layer.dW
                    layer.b -= lr * layer.db
                if hasattr(layer, "C"):
                    layer.C -= lr * layer.dC
        else:
            optimizer.begin_step()  # обновляю раз в эпоху, т.к. full batch закидываю
            for i, layer in enumerate(flatten_layers(layers)):
                if hasattr(layer, "W"):
                    optimizer.step(layer.W, layer.dW, key=f"W{i}")
                    optimizer.step(layer.b, layer.db, key=f"b{i}")
                if hasattr(layer, "C"):
                    optimizer.step(layer.C, layer.dC, key=f"C{i}")

        if verbose and (epoch + 1) % 100 == 0:
            train_acc = accuracy(layers, X_train, Y_train)
            test_acc = accuracy(layers, X_test, Y_test)
            print(
                f"epoch {epoch+1} | loss={train_loss:.4f} | "
                f"train_acc={train_acc:.3f} | test_acc={test_acc:.3f}"
            )

        # test loss делаю после, чтобы не перезаписать значения тренировочной
        test_logits = forward_model(layers, X_test)
        test_loss = test_loss_fn.forward(test_logits, Y_test)
        test_losses.append(test_loss)

    return train_losses, test_losses


def accuracy(layers, X, Y):
    logits = forward_model(layers, X)
    preds = np.argmax(logits, axis=1)
    true = np.argmax(Y, axis=1)
    return np.mean(preds == true)


def count_params(layers):
    total = 0
    for layer in flatten_layers(layers):
        if hasattr(layer, "W"):
            total += layer.W.size + layer.b.size
        if hasattr(layer, "C"):
            total += layer.C.size
    return total


class RBFLayer:
    # RBF-слой: z[i,j] = exp(-gamma * ||x_i - C_j||^2)
    # C_j - обучаемые центры (n_centers x n_features)
    def __init__(self, n_features, n_centers, gamma=1.0):
        self.gamma = gamma
        # центры: (n_centers, n_features)
        self.C = np.random.randn(n_centers, n_features)

    def forward(self, x):
        # x: (batch_size, n_features)
        self.x = x
        # diff: (batch_size, n_centers, n_features)
        self.diff = x[:, None, :] - self.C[None, :, :]
        # dist2: (batch_size, n_centers)
        self.dist2 = np.sum(self.diff**2, axis=2)
        # z: (batch_size, n_centers)
        self.z = np.exp(-self.gamma * self.dist2)
        return self.z

    def backward(self, grad_out):
        # grad_out = dL/dz, shape: (batch_size, n_centers)
        # Нужно вернуть dL/dx: (batch_size, n_features)
        # И посчитать dL/dC: (n_centers, n_features) в self.dC
        # factor = dL/d(dist2) * d(dist2)/d(diff)
        # z = exp(-gamma*dist2)
        # dz/ddist2 = -gamma*z
        # dist2 = sum(diff^2) -> d(dist2)/d(diff) = 2*diff
        # => dL/ddiff = grad_out * (-gamma*z) * 2*diff
        factor = (-2.0 * self.gamma) * (grad_out * self.z)  # (B,K) −2yz
        factor = factor[:, :, None]  # (B,K,1)

        # dL/dx = sum_j factor[i,j]*diff[i,j,:]
        grad_x = np.sum(factor * self.diff, axis=1)  # (B,D) −2γz * (x - c)

        # dL/dC: знак минус, потому что diff = x - C
        self.dC = -np.sum(factor * self.diff, axis=0)  # (K,D) 2γz * (x - c)

        return grad_x
