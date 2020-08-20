import numpy as np

class RVFLNN:
    def __init__(self):
        self.N_i = None  # 输入节点数
        self.N_o = None  # 输出节点数
        self.N_h = 200  # 隐层节点数
        self.sigma = 1
        self.beta = 1e-6

    def train(self, X, Y):
        self.N_i = X.shape[0]
        self.N_o = Y.shape[0]
        self.W_i = np.random.uniform(-self.sigma, self.sigma, (self.N_h, self.N_i + 1))

        Z = np.tanh(self.W_i.dot(np.vstack([X, np.ones([1, X.shape[1]])])))
        H = np.vstack([Z, X])

        self.W_o = Y.dot(H.T.dot(np.linalg.inv(H.dot(H.T) + self.beta * np.eye(self.N_h + self.N_i))))

    def predict(self, X):
        Z = np.tanh(self.W_i.dot(np.vstack([X, np.ones([1, X.shape[1]])])))
        H = np.vstack([Z, X])
        return self.W_o.dot(H)

np.random.seed(42)

model = RVFLNN()

n = 2  # 使用 n 个历史点作为输入

num_train = 8000
x_train = np.vstack([select_samples(x,0+i,num_train) for i in range(n)])
y_train = select_samples(x,n,num_train)

model.train(x_train, y_train)

num_test = 2000
test_start = 8000
P = np.empty((3,num_test))
Q = select_samples(x,test_start, num_test)
p = np.vstack([select_samples(x,test_start-n+i,1) for i in range(n)])


for i in range(num_test):
    p_next = model.predict(p)
    P[:,i] = np.squeeze(p_next)
    p = np.vstack([p,p_next])[3:]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot()
plt.plot(*P, 'r')
plt.plot(*Q, 'g')

plt.figure()
dim = ['x','y','z']
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(P[i,:].T, label='prediction')
    plt.plot(Q[i,:].T, label='true')
    plt.ylabel(dim[i])
    plt.legend(loc='upper right')
