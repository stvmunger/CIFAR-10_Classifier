import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation='relu'):
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        self.weights.append(
            np.random.randn(input_size, hidden_size1) * np.sqrt(2 / input_size)  # 使用He初始化
        )
        self.weights.append(
            np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2 / hidden_size1)
        )
        self.weights.append(
            np.random.randn(hidden_size2, output_size) * np.sqrt(1 / hidden_size2)
        )
        self.biases = [
            np.zeros(hidden_size1),
            np.zeros(hidden_size2),
            np.zeros(output_size)
        ]
        
        if activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: np.where(x <= 0, 0, 1)
        else:
            raise ValueError("Unsupported activation")

    def forward(self, X):
        a = [X]
        z = []
        for i in range(3):
            z_i = a[-1] @ self.weights[i] + self.biases[i]
            if i < 2:  # 隐藏层用ReLU，输出层用Softmax
                a_i = self.activation(z_i)
            else:
                exp_scores = np.exp(z_i - np.max(z_i, axis=1, keepdims=True))
                a_i = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            a.append(a_i)
            z.append(z_i)
        return a[-1], a, z

    def compute_loss(self, X, y, reg_strength):
        probs, _, _ = self.forward(X)
        N = X.shape[0]
        loss = -np.sum(np.log(probs[range(N), y])) / N
        # L2正则化
        reg_loss = 0.5 * reg_strength * (
            np.sum(self.weights[0] ** 2) +
            np.sum(self.weights[1] ** 2) +
            np.sum(self.weights[2] ** 2)
        )
        return loss + reg_loss

    def evaluate(self, X, y):
        probs, _, _ = self.forward(X)
        preds = np.argmax(probs, axis=1)
        accuracy = np.mean(preds == y)
        loss = self.compute_loss(X, y, 0.0)  # 验证时不加正则化
        return loss, accuracy

    def backward(self, X, y, probs, a, z, reg_strength):
        N = X.shape[0]
        grads = {}
        # 输出层梯度
        delta = probs.copy()
        delta[range(N), y] -= 1
        delta /= N
        dW3 = a[2].T @ delta + reg_strength * self.weights[2]
        db3 = np.sum(delta, axis=0)
        grads['W3'], grads['b3'] = dW3, db3
        
        # 隐藏层2梯度
        delta2 = delta @ self.weights[2].T * self.activation_derivative(z[1])
        dW2 = a[1].T @ delta2 + reg_strength * self.weights[1]
        db2 = np.sum(delta2, axis=0)
        grads['W2'], grads['b2'] = dW2, db2
        
        # 隐藏层1梯度
        delta1 = delta2 @ self.weights[1].T * self.activation_derivative(z[0])
        dW1 = a[0].T @ delta1 + reg_strength * self.weights[0]
        db1 = np.sum(delta1, axis=0)
        grads['W1'], grads['b1'] = dW1, db1
        
        return grads

    def save_weights(self, path):
        np.savez_compressed(path, 
            W1=self.weights[0], b1=self.biases[0],
            W2=self.weights[1], b2=self.biases[1],
            W3=self.weights[2], b3=self.biases[2])

    def load_weights(self, path):
        data = np.load(path)
        self.weights = [data['W1'], data['W2'], data['W3']]
        self.biases = [data['b1'], data['b2'], data['b3']]