import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def calculate_input(x, w, b):
    return np.dot(x, w) + b

def step_trigger(x):
    return 1 if x >= 0 else 0

def bipolar_trigger(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def sigmoid_curve(x):
    return 1 / (1 + np.exp(-x))

def relu_filter(x):
    return max(0, x)

def leaky_relu_filter(x):
    return x if x > 0 else 0.01 * x

def calculate_error(prediction, target):
    return prediction - target

def train_model(X, y, activation_func, learning_rate=0.05, max_epochs=1000):
    weights = np.random.uniform(-0.5, 0.5, X.shape[1])
    bias = np.random.uniform(-0.5, 0.5)
    error_history = []

    for epoch in range(max_epochs):
        cumulative_error = 0

        for i in range(len(X)):
            signal = calculate_input(X[i], weights, bias)
            output = activation_func(signal)

            gap = calculate_error(output, y[i])
            cumulative_error += gap**2

            weights = weights - learning_rate * gap * X[i]
            bias = bias - learning_rate * gap

        mse = cumulative_error / len(X)
        error_history.append(mse)

        if mse <= 0.002:
            break

    return weights, bias, error_history, epoch + 1

logic_inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
and_results = np.array([0,0,0,1])
xor_results = np.array([0,1,1,0])

w_and, b_and, and_errors, and_epochs = train_model(logic_inputs, and_results, step_trigger)

plt.plot(and_errors)
plt.title("A2: AND Gate Error")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()

activation_options = {
    "Bipolar": bipolar_trigger,
    "Sigmoid": sigmoid_curve,
    "ReLU": relu_filter
}

comparison_results = {}
for name, func in activation_options.items():
    _, _, _, ep = train_model(logic_inputs, and_results, func)
    comparison_results[name] = ep

rates_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
epoch_counts = []

for r in rates_to_test:
    _, _, _, ep = train_model(logic_inputs, and_results, step_trigger, learning_rate=r)
    epoch_counts.append(ep)

plt.plot(rates_to_test, epoch_counts, marker='o')
plt.title("A4: Learning Rate vs Epochs")
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")
plt.show()

w_xor, b_xor, xor_errors, xor_epochs = train_model(logic_inputs, xor_results, step_trigger)

customer_data = np.array([
    [20,6,2,386], [16,3,6,289], [27,6,2,393], [19,1,2,110], [24,4,2,280],
    [22,1,5,167], [15,4,2,271], [18,4,2,274], [21,1,4,148], [16,2,4,198]
])
customer_buys = np.array([1,1,1,0,1,0,1,1,0,0])

def normalize_data(data):
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

scaled_customers = normalize_data(customer_data)
w_c, b_c, err_c, ep_c = train_model(scaled_customers, customer_buys, sigmoid_curve)

def fast_math_solve(X, y):
    X_plus_bias = np.c_[np.ones(len(X)), X]
    return np.linalg.pinv(X_plus_bias).dot(y)

weights_shortcut = fast_math_solve(scaled_customers, customer_buys)

def train_smart_network(X, y, lr=0.05, epochs=1000):
    np.random.seed(0)
    weights_hidden = np.random.uniform(-0.05, 0.05, (2, 2))
    weights_output = np.random.uniform(-0.05, 0.05, (2, 1))
    error_log = []

    for epoch in range(epochs):
        total_err = 0
        for i in range(len(X)):
            hidden_layer = sigmoid_curve(np.dot(X[i].reshape(1, -1), weights_hidden))
            final_guess = sigmoid_curve(np.dot(hidden_layer, weights_output))

            error = y[i] - final_guess
            total_err += error**2

            delta_out = error * final_guess * (1 - final_guess)
            delta_hidden = delta_out.dot(weights_output.T) * hidden_layer * (1 - hidden_layer)

            weights_output += lr * hidden_layer.T.dot(delta_out)
            weights_hidden += lr * X[i].reshape(-1, 1).dot(delta_hidden)

        mse = total_err / len(X)
        error_log.append(mse)
        if mse <= 0.002: break

    return weights_hidden, weights_output, error_log

w_h, w_o, xor_mlp_errors = train_smart_network(logic_inputs, xor_results)

def encode_output(y):
    return np.array([[1,0] if val==0 else [0,1] for val in y])

y_and_encoded = encode_output(and_results)

sk_model = MLPClassifier(
    hidden_layer_sizes=(4,),
    solver='lbfgs',
    activation='logistic',
    max_iter=1000,
    random_state=0
)
sk_model.fit(logic_inputs, and_results)

sk_customer_model = MLPClassifier(
    hidden_layer_sizes=(6,),
    solver='lbfgs',
    max_iter=1000,
    random_state=0
)
sk_customer_model.fit(scaled_customers, customer_buys)