# General Guidance
## Framework of ML
- Training data: $\{(\mathbf{x}^{(i)}, \hat{y}^{(i)} \}, i =1, \dots, N$
- Testing data: $\{\mathbf{x}^{(i)}\}, i =N+1, \dots, N+M$

## Guidance

<img src="https://zhj-0830.oss-cn-hangzhou.aliyuncs.com/img/image-20220213165924721.png" alt="image-20220213165924721" style="zoom: 67%;" />

### Model Bias

Model is too simple to describe the function

<img src="https://zhj-0830.oss-cn-hangzhou.aliyuncs.com/img/image-20220213171539147.png" alt="image-20220213171539147" style="zoom:50%;" />

- *find a needle in a haystack but there is no needle*
- **Solution**: Redesign model to make it more flexible, such as use more features, use deep learning and so on.

### Optimization Issue

Large loss may also means local minimum.

<img src="https://zhj-0830.oss-cn-hangzhou.aliyuncs.com/img/image-20220213171726005.png" alt="image-20220213171726005" style="zoom: 50%;" />

### Model Bias vs. Optimization Issue

#### Example[^1]

<img src="https://zhj-0830.oss-cn-hangzhou.aliyuncs.com/img/image-20220214095712973.png" alt="image-20220214095712973" style="zoom: 67%;" />

- On testing data, the loss on 56-layer net is bigger than 20-layer net. *But is this overfitting?*
- In the meanwhile, on training data, however, the loss on 56-layer net is also bigger than 20-layer net. 
- In theory, the training loss on 56-layer net is smaller than 20-layer net. If this is overfitting caused by model bias, the training loss of 56-layer net should be smaller than 20-layer net.
- So, what happened on 56-layer net is **optimization issue**.

#### Advice

- Gaining Insights from comparison between training loss and testing loss
- Start from the shallower networks which is easy to optimize
- If deeper networks don’t obtain smaller training loss, there is a optimization issue

#### Solution

Use more powerful optimization technology

### Overfitting

#### An extreme example

$$
f(x) = \begin{cases}
\hat{y}_i, &\exist x_i = x\\
\text{random}, &\text{otherwise}
\end{cases}
$$

$f$ has **zero training loss**, but **large testing loss**

#### Solution:

- More training data & Data augmentation
- Constrain model
- Less parameters, sharing parameters
- Less features
- Early stopping
- Regularzation
- Dropout

#### Bias-Complexity Trade-off

<img src="https://zhj-0830.oss-cn-hangzhou.aliyuncs.com/img/image-20220214101847946.png" alt="image-20220214101847946" style="zoom: 50%;" />

### Cross Validation

- split data into training set and validation set
- choose model according to loss on validation set
- N-fold Cross Validation

### Mismatch

- Training data and testing data have different distributions. ==Be aware of how data is generrated==
- Overfitting can be overcome by using more training data. However, mismatch can not.

[^1]: Ref: http://arxiv.org/abs/1512.03385

