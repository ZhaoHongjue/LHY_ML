# Introduction of ML & DL
## ML
- **Machine Learning**$\approx$==Looking for Function==
	- *Speech Recognition*: voice$\to$text
	- *Image Recognition*: image$\to$class
- **Different Types of function**
	1. *Regression*: Outputs a scalar
	2. *Classification*: Given options(*classes*), the function outputs the correct one.
	3. *Structured Learning*: **create** something structure(image, document...)
- **Steps of Training**:
	1. **Function** with Unknown Parameters
		- based on ==domain knowledge==
		- Model, feature, weight, bias
	2. Define **Loss** from Training Data
		- Loss is a function of parameters $L(b, w)$
		- how good a set of values are
		- **MAE**: absolute error, $e = |y - \hat{y}|$
		- **MSE**: square error, $e = (y-\hat y)^2$
	3. **Optimization**: $w^*, b^* = \mathop{\arg\min}_w L \to$ usually use *Gradient Descent*
		1. (Randomly) pick an initial value $w^0, b^0$
		2. Compute $\frac{\partial L}{\partial w}|_{w = w^0}$
		3. Update $w$ iteratively. $w \leftarrow w - \eta \frac{\partial L}{\partial w}$, $\eta$: learning rate, need to be set by yourself$\to$hyperparameters
		**Question**: may not find global minima, usually find local minima.
## DL
- Linear Model may be too simple, which can not get features of data.

- All Piecewise Linear Curves = Constant + sum of a set of <img src="https://zhj-0830.oss-cn-hangzhou.aliyuncs.com/img/image-20220211222249765.png" alt="image-20220211222249765" style="zoom:33%;" />

- **Sigmoid**:
  $$
  y = c \frac{1}{1 + \exp(-(b+wx_1))}
  $$
  <img src="https://zhj-0830.oss-cn-hangzhou.aliyuncs.com/img/image-20220211222423920.png" alt="image-20220211222423920" style="zoom:50%;" />

- Therefore:
  $$
  y = b +\sum_{i}c_i \text{sigmoid}(b_i + \sum_jw_{ij}x_j)
  $$
  sigmoid is called *Activation Function*

  Other activation functions: such as ReLU(*Rectified Linear Unit*)
  $$
  \max(0, b+wx)
  $$

  - RelU is better than sigmoid

## Summary

- Steps of Train: 
  1. Function with Unknown Parameters;
  2. Define Loss from Training Data
  3. Optimization (usually use Gradient Descent)
- A kind of explanation of Activation Function, which may be associated with Math Analysis.