# LinearRegression

## 知识准备

### 基础数学知识

1. 最小二乘法（least square）

    最小二乘法是一种数学优化技术。它通过最小化误差的平方和寻找数据的最佳函数匹配。
    利用最小二乘法可以简便地求得未知的数据，并使得这些求得的数据与实际数据之间误差的平方和为最小.

$$ min \sum_{i=1}^m (y_\theta(x)-y)^2 $$

2. likehood 似然（最大似然评估）

    likehood function：
    $L(\theta \mid x) = f(x \mid \theta)$

    在上式中，如果x已经给出，预测 $\theta$ 的值，则该函数为似然函数。如果 $\theta$ 已经给出，预测x的值，那么这就是普通的概率函数。

    而最大似然估计就是取一定的 $\theta$值，使得似然函数最大，即：
    $$ arg  \;max \;L(\theta \mid X)$$


3. 矩阵的迹及其部分等式

    定义：矩阵的迹 $tr(A) = \sum_{i=1}^m a_{ii}$ 为标量且A为标量或方阵
    一些公式：
      $$ tr(ABC) = tr(CAB) = tr(BCA)$$
      $$ tr(A+B) = tr(A) + tr(B)$$
      $$\Delta_A tr(AB) = B^T $$
      $$ \Delta_{A^T}f(A) = (\Delta_Af(A))^T $$
      $$ \Delta_Atr(ABA^TC) = CAB+C^TAB^T$$
      $$ \Delta_A|A| = |A|(A^{-1})^T$$

    等式的证明见参考资料链接
4. 矩阵求导公式
    矩阵A对矩阵B求导，就是矩阵A中的每个值对矩阵B中的每个值求导，向量同理。

    一些重要的公式：
    $$ \frac{d X}{d X^T } = I $$
    $$ \frac{d X^T}{d X } = I $$
    $$ \frac{d AX}{d X^T } = A $$
    $$ \frac{d AX^T}{d X } = A $$
    $$ \frac{d U^TV}{d X } = \frac{d U^T}{d X}V + \frac{d V^T}{d X }U$$

    等式的证明见参考资料链接
### 基础机器学习知识

1. 学习过程
2. hypothesis
3. classification、regression
4. loss function(error function)

## LinearRegression

### 理论模型
>在统计学中，线性回归是利用称为线性回归方程的最小二乘函数对一个或多个自变量和因变量之间关系进行建模的一种回归分析。

在线性回归中，数据使用线性预测函数来建模，并且未知的模型参数也是通过数据来估计。这些模型被叫做线性模型。在机器学习算法中，我们使用如下的线性函数来预测y：
$$ h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 ...$$
即：
$$ h_\theta(X) = \theta^T x$$
其中：
$$ \theta = \begin{bmatrix} \theta_0\\ \theta_1\\... \end{bmatrix}
   x =  \begin{bmatrix} x_0\\ x_1\\... \end{bmatrix}
$$
也可写成：
$$ y = Wx + b$$

上面式子中,$\theta$ 和W都是一个意思，称为参数或者权重(weight)。其中 $\theta_0$和b表示偏差（bias），也可称为误差项。之后我们统一使用$h_\theta(X) = \theta^T x$

为了评估线性模型的性能，线性回归使用最小二乘法来作为损失函数（loss function | cost function）：
$$ J(\theta) = 1/2m\sum_{i=1}^m (h(x^i)-y^i)^2$$

至于为什么如此选择，后面再进行解释。

### minimize J(θ)

为了提升预测准确度或者时候模型性能，我们需要选择 $\theta$ 来最小化损失函数即：
$$ \theta = arg \; min \; J(\theta)$$

一般有两种方法求解：

#### gradient descent

为了选择 $\theta$ 最小化  $J(\theta)$,先给 $\theta$一个初始值，然后重复的改变它,直到收敛得到一个 $\theta$,使得 $J(\theta)$ 最小。

gradient descent就是这样一个算法，从一个初始 $\theta$开始，然后重复的执行如下更新：
$$  \theta_j = \theta_j - \alpha \frac{\delta}{\delta\theta_j}J(\theta) $$

表达式中 $\alpha$代表学习速率，表示 $\theta$值收敛步长。$\frac{\delta}{\delta\theta_j}J(\theta_j)$ 代表 $J(\theta_j)$ 对 $\theta$的偏导。

对于这个偏导，我们可以很容易的求解：
$$
  \begin{align}
    \frac{\delta}{\delta\theta_j}J(\theta) & = \frac{\delta}{\delta\theta_j}J(\theta) \\
    & =   \frac{\delta}{\delta\theta_j}(1/2m\sum_{i=1}^m (h(x^i)-y^i)^2) \\
    & =  1/m(\sum_{i=1}^m (h(x^i)-y^i)x_j^i)
  \end{align}

$$
向量化表示：
$$
    \frac{\delta}{\delta\theta_j}J(\theta)  = 1/m(\sum_{i=1}^m (h(x^i)-y^i)x_j^i)
$$
$$  
     \begin{align}
        \frac{\delta}{\delta\theta}J(\theta) &=
        \begin{bmatrix}
           \frac{\delta}{\delta\theta_1}J(\theta) \\ \frac{\delta}{\delta\theta_2}J(\theta) \\
           ...\\
           \frac{\delta}{\delta\theta_n}J(\theta)
        \end{bmatrix} \\

      & = 1/m \;\sum_{i=1}^m
      \begin{bmatrix}  (h(x^i)-y^i)x_1^i \\
        (h(x^i)-y^i)x_2^i\\
      ...\\
      (h(x^i)-y^i)x_n^i
       \end{bmatrix} \\

      & = 1/m\sum_{i=1}^m((h(x^i)-y^i)x^i) \\

      & = 1/m\;X^T(h(X)-y) \\
      & = 1/m\;X^T(X\theta - y)

      \end{align}
$$

所以根据gradient descent：
$$
  for \;i \;in \;range(niteration):\\
      \qquad \theta = \theta - \alpha/m\;X^T(X\theta - y)
$$


#### normal equation(least square)

normal equation 其实就是解方程，使得梯度为0即达到最优值，上面我们已经求出梯度表达式：
$$
1/m\;X^T(X\theta - y)
$$
令它为0可求：
$$
\begin{align}

 1/m\;X^T(X\theta - y) &= 0\\
 X^TX\theta & = X^Ty \\
 \theta &= (X^TX)^{-1}X^Ty
\end{align}
$$
如此一来，不需迭代，直接根据输入的X和y求得预测函数

除了上面这种解法求梯度，还有两种方法：

#### 直接矩阵求导
$$
\begin{align}

  J(\theta) &= 1/2m(X\theta-y)^T(X\theta-y) \\
  \frac{\delta}{\delta\theta}J(\theta) &= 1/2m(2\frac{\delta}{\delta\theta}(X\theta-y)^T) (X\theta-y)\\
  & = 1/mX^T(X\theta-y)

\end{align}
$$

#### 利用矩阵的迹求导
$$
  J(\theta) = 1/2m(X\theta-y)^T(X\theta-y)
$$
这是一个标量，所以它的迹即为它本身，所以我们可以直接用迹来求解：
$$
\begin{align}
tr J(\theta) &= 1/2m tr(X\theta-y)^T(X\theta-y)\\
             &= 1/2mtr(\theta^TX^T-y^T)(X\theta-y)\\
             &= 1/2mtr(\theta^TX^TX\theta - y^TX\theta-\theta^TX^Ty-y^Ty) \\
=> \\
\frac{\delta}{\delta\theta}J(\theta)
  &= 1/2m\frac{\delta}{\delta\theta}tr(\theta^TX^TX\theta - y^TX\theta-\theta^TX^Ty-y^Ty) \\
  &= 1/2m\frac{\delta}{\delta\theta}tr(\theta^TX^TX\theta - 2y^TX\theta)\\
  &= 1/2m(X^TX\theta +X^TX\theta -2X^Ty)  \\
  &= 1/m(X^TX\theta - X^Ty)
\end{align}
$$
## python代码实现

### 基本numpy函数解释
```python

1. np.insert(X,0,1,axis=1):
    X: numpy narray
    0: 插入的位置
    1：初始化的值
    axis：0为row，1为col
2. np.random.random((x,y))
    (row,col) 随机生成值为(0,1)，大小为row*col的数组

3. X.dot：矩阵乘法
   X.T ： 矩阵转置
   np.linalg.pinv:矩阵求伪逆

```
### 代码

```python
import numpy as np
class MyLinearRegression:
    """
    params:    
    n_iterations:
    learning_rate:
    gradient_descent:
    """
    def __init__(self, n_iterations=100, learning_rate=0.01, gradient_descent=True ):
        self.w = None
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent

    def fit(self,X,y):
        # X :m*n => m*(n+1) for bias weight
        X = np.insert(X,0,1,axis=1)
        # gradient descent
        if self.gradient_descent :
            n_features = np.shape(X)[1]
            self.w = np.random.random((n_features,))  # 注意这里col还不能填1
            for _ in range(self.n_iterations):
                w_gradient = X.T.dot(X.dot(self.w)-y)
                self.w = self.w - self.learning_rate * w_gradient
        # normal equation  
        else:
            self.w = (np.linalg.pinv(X.T.dot(X))).dot(X.T.dot(y))


    def predict(self,X):
        X = np.insert(X,0,1,axis=1)
        y_pred = X.dot(self.w)
        return y_pred

```

### 验证：
```python
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
X,y = datasets.make_regression(n_features=1,n_samples=200,bias=100,noise=5)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#clf = MyLinearRegression(gradient_descent=False)
clf = MyLinearRegression(n_iterations=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
mse = mean_squared_error(y_pred,y_test)
print(mse)
clf2 = linear_model.LinearRegression()
clf2.fit(X_train,y_train)
y_pred2 = clf2.predict(X_test)
mse2 = mean_squared_error(y_pred2,y_test)
print(mse2)

clf3 = DecisionTreeRegressor(max_depth=20)
clf3.fit(X_train,y_train)
mse3 = mean_squared_error(clf3.predict(X_test),y_test)
print(mse3)

```
可以从结果中看出，在数据集较小的情况下三个效果差不多，说明我们的算法没有问题。

## 扩展

### why choose least squares

### Locally weighted linear regression

### non-parametric algorithm and parametric algorithm

这三个问题比较上面略微复杂，直接看参考资料中 Andrew Ng的讲义。


## 参考资料

1. [wiki：最小二乘法](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95)
2. [wiki：似然函数](https://en.wikipedia.org/wiki/Likelihood_function)
3. [矩阵的迹公式证明1](http://dawenl.github.io/files/mat_deriv.pdf)
4. [矩阵的迹公式证明2](http://math.stackexchange.com/questions/277151/prove-that-gradient-of-operatornametra-cdot-b-cdot-at-cdot-c-with-r?rq=1)
5. [矩阵求导](https://wenku.baidu.com/view/a90e4c61453610661ed9f479)
6. [github 上linearRegession的实现](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/supervised_learning/linear_regression.py)
7. [Andrew Ng 关于Linear regression和logistic Regression的讲义](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
