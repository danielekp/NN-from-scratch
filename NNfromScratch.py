import numpy as np
import matplotlib.pyplot as plt
import argparse

class MSELoss:

    def forward(self, y, target):
        self.diff = diff = y - target
        c = np.sum(np.square(diff)) / diff.size
        return c

    def backward(self):
        assert hasattr(self, 'diff')
        r = 2*self.diff / self.diff.size
        return r

class Tanh:

    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, dy):
        assert hasattr(self, 'x')
        if np.isnan(self.x[0][0]):
            print("Learning rate is too high, it diverges!")
            exit(-1)
        r = 1 - np.square(np.tanh(self.x))
        return np.multiply(r,dy)

class myReLU:

    def forward(self, x):
        self.x = x
        return np.maximum(0,x)

    def backward(self, dy):
        assert hasattr(self, 'x')
        if np.isnan(self.x[0][0]):
            print("Learning rate is too high, it diverges!")
            exit(-1)
        dx = np.where(self.x > 0, dy, 0)
        return dx

class Sigmoid:

    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self, dy):
        assert hasattr(self, 'x')
        if np.isnan(self.x[0][0]):
            print("Learning rate is too high, it diverges!")
            exit(-1)
        return dy * (1 / (1 + np.exp(-self.x))) * (1 - (1 / (1 + np.exp(-self.x))))



def linear_forward_batch(x, W, b):
    r = np.empty(shape=(x.shape[0],W.shape[0]))
    for i in range(x.shape[0]):
        u = np.dot(W,x[i]) + b
        for y in range(u.shape[0]):
            r[i][y] = u[y]
    return r

def linear_backward_batch(dy, x, W, b):
    assert dy.ndim == 2 and dy.shape[1] == W.shape[0]
    u = np.empty(shape=(x.shape[0],x.shape[1]))
    for i in range(x.shape[0]):
        r = np.dot(dy[i],W)
        for y in range(r.shape[0]):
            u[i][y] = r[y]
    d = np.dot(x.T, dy).T
    t = dy[0]
    return u, d, t

class LinearBatch:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Initialization of the weights
        bound = 3 / np.sqrt(in_features)
        self.W = np.random.uniform(-bound, bound, (out_features, in_features))
        bound = 1 / np.sqrt(in_features)
        self.b = np.random.uniform(-bound, bound, out_features)

        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        return linear_forward_batch(x, self.W, self.b)

    def backward(self, dy):
        assert hasattr(self, 'x')
        assert dy.ndim == 2 and dy.shape[1] == self.W.shape[0]
        dx, self.grad_W, self.grad_b = linear_backward_batch(dy, self.x, self.W, self.b)
        return dx

class MLPBatch:

    def __init__(self, hidden, neurons, activation):
        if activation=='Tanh':
            self.act = []
            for e in range(hidden):
                self.act.append(Tanh())
        elif activation=='ReLU':
            self.act = []
            for e in range(hidden):
                self.act.append(myReLU())
        else:
            self.act = []
            for e in range(hidden):
                self.act.append(Sigmoid())

        if hidden>0:
            self.l = []
            self.l.append(LinearBatch(1, neurons))
            for i in range(hidden-1):
                self.l.append(LinearBatch(neurons,neurons))
            self.l.append(LinearBatch(neurons,1))
        else:
            self.l = [LinearBatch(1,1)]

    def forward(self, x):
        for e in range(len(self.l)-1):
            x = self.l[e].forward(x)
            x = self.act[e].forward(x)
        return self.l[len(self.l)-1].forward(x)

    def backward(self, dy):
        for e in range(len(self.l)-1,0,-1):
            dy = self.l[e].backward(dy)
            dy = self.act[e-1].backward(dy)
        return self.l[0].backward(dy)

def main():

    parser = argparse.ArgumentParser(description="NN from scratch")
    parser.add_argument("--epochs", type=int, default=200, metavar='N', help="Number of epochs (default 200)")
    parser.add_argument("--lr", type=float, default=0.15, metavar='N', help="Learning rate (default 0.15)")
    parser.add_argument("--seed", type=int, default=2, metavar='N', help="Seed for data generation (default 2)")
    parser.add_argument("--complexity", choices=['1', '2', '3', '4', '5'], default='3', help="Complexity of generated data (from 1 to 5, default 3)")
    parser.add_argument("--hidden", type=int, default=3, help="Number of hidden layers (default 3)")
    parser.add_argument("--neurons", type=int, default=12, help="Number of neurons per hidden layer (default 12)")
    parser.add_argument("--activation", choices=['Tanh', 'ReLU', 'Sigmoid'], default='Tanh', help="Activation function between hidden units (Tanh, ReLU or Sigmoid, default Tanh)")
    parser.add_argument("--points", type=int, default=300, help="Number of data points (default 300)")

    args = parser.parse_args()


    np.random.seed(args.seed)
    x = np.random.randn(args.points, 1)
    x = np.sort(x, axis=0)

    if args.complexity=='1':
        targets = x * 4
    elif args.complexity=='3':
        targets = np.sin(x * 2 * np.pi / 3)
    elif args.complexity=='4':
        targets = np.cos(x) * np.arctan(x) / (x*2)
    elif args.complexity=='2':
        targets = np.square(x)*2
    elif args.complexity=='5':
        targets = np.sin(x) * np.arctan(x) / np.pi*np.sin(x)


    targets = targets + 0.2 * np.random.randn(*targets.shape)

    mlp = MLPBatch(args.hidden, args.neurons, args.activation)  # Create MLP network
    loss = MSELoss()  # Create loss

    fig, ax = plt.subplots(1)
    ax.plot(x, targets, '.')
    learning_rate = args.lr
    n_epochs = args.epochs
    for i in range(n_epochs):
        # Forward computations
        y = mlp.forward(x)
        c = loss.forward(y, targets)

        # Backward computations
        dy = loss.backward()
        dx = mlp.backward(dy)

        # Gradient descent update
        learning_rate *= 0.99  # Learning rate annealing
        for module in mlp.__dict__.values():
            for r in module:
                if hasattr(r, 'W'):
                    r.W = r.W - r.grad_W * learning_rate
                    r.b = r.b - r.grad_b * learning_rate

        ax.clear()
        ax.plot(x, targets, '.')
        ax.plot(x, y, 'r-')
        ax.grid(True)
        ax.set_title('Iteration %d/%d' % (i+1, n_epochs))
        path = "fitting/"+str(i)+".jpeg"
        plt.savefig(path)
        plt.pause(0.005)

if __name__=="__main__":
    main()
