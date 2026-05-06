# 🔢 Linear Algebra for Machine Learning — Notes


>
> Linear algebra is the **core mathematical foundation** for machine learning.
> Most datasets and models are represented using vectors and matrices,
> enabling efficient computation, data manipulation, and optimization.

---

## Table of Contents

1. [Why Linear Algebra in ML](#1-why-linear-algebra-in-ml)
2. [Fundamental Concepts](#2-fundamental-concepts)
3. [Operations in Linear Algebra](#3-operations-in-linear-algebra)
4. [Linear Transformations](#4-linear-transformations)
5. [Matrix Operations](#5-matrix-operations)
6. [Eigenvalues and Eigenvectors](#6-eigenvalues-and-eigenvectors)
7. [Solving Linear Systems](#7-solving-linear-systems)
8. [Applications in Machine Learning](#8-applications-in-machine-learning)
9. [Cheat Sheet](#10-cheat-sheet)

---

## 1. Why Linear Algebra in ML

Linear algebra is a core mathematical foundation for machine learning, as most datasets and models are represented using vectors and matrices. It allows efficient computation, data manipulation, and optimization, making complex tasks manageable.

| Concept | Role in ML |
|---------|-----------|
| **Vectors** | Represent data features |
| **Matrices** | Represent entire datasets |
| **Dot product** | Measure similarity between data points |
| **Eigenvalues/Eigenvectors** | Power dimensionality reduction (PCA) |
| **Matrix decompositions** | Enable optimization and training |

```
Data Flow in ML using Linear Algebra:

  Raw Data
     ↓
  Feature Vectors   →  v = [age, salary, score]
     ↓
  Dataset Matrix    →  X (n_samples × n_features)
     ↓
  Transformations   →  Xβ = Y  (Linear Regression)
     ↓
  Predictions       →  ŷ = X · weights
```

---

## 2. Fundamental Concepts

### 2.1 Scalars

Scalars are single numerical values, without direction — magnitude only. In machine learning, they are used to adjust things like the weights in a model or the learning rate during training.

```
k = 3        ← a scalar

Scalar × Vector:
k · v = 3 · [2, -1, 4] = [6, -3, 12]
         ↑ each element multiplied by k
```

**In ML:** Learning rate (α = 0.01), regularisation strength (λ), loss values.

---

### 2.2 Vectors

Vectors are quantities that have both magnitude and direction, often represented as arrows in space.

```
Column vector:          Row vector:
    ┌ 2 ┐
v = │-1 │         v = [2, -1, 4]
    └ 4 ┘

Magnitude (norm):
  ||v|| = √(2² + (-1)² + 4²) = √21 ≈ 4.58
```

**In ML:** A single data sample (one row) = one feature vector.
```
Student record:  v = [age=22, score=85, hours=6]
```

---

### 2.3 Matrices

Matrices are rectangular arrays of numbers, arranged in rows and columns. Matrices are used to represent linear transformations, systems of linear equations, and data transformations in machine learning.

```
Dataset Matrix X (3 students × 3 features):

         age   score  hours
         ┌              ┐
Student1 │ 22    85     6 │
Student2 │ 25    90     8 │
Student3 │ 20    78     5 │
         └              ┘
Shape: (3 × 3)   →   (n_samples × n_features)

Special Matrices:
  Identity (I)   →   diagonal 1s, zeros elsewhere
  Square Matrix  →   rows == columns
  Symmetric (A)  →   A = Aᵀ  (used in covariance matrices)
```

---

## 3. Operations in Linear Algebra

### 3.1 Vector Addition & Subtraction

Add or subtract corresponding elements of vectors/matrices.

```
u = [2, -1, 4]     v = [3, 0, -2]

u + v = [2+3,  -1+0,  4+(-2)] = [5, -1,  2]
u - v = [2-3,  -1-0,  4-(-2)] = [-1, -1,  6]
```

---

### 3.2 Scalar Multiplication

Multiply each element by a scalar.

```
k = 3,   v = [2, -1, 4]

k · v = [3×2, 3×(-1), 3×4] = [6, -3, 12]
```

---

### 3.3 Dot Product

Measures similarity of directions by multiplying matching elements and summing.

```
u · v = u₁v₁ + u₂v₂ + u₃v₃

Example:
  u = [2, -1, 4]   v = [3, 0, -2]
  u · v = (2×3) + (-1×0) + (4×-2)
        = 6 + 0 + (-8)
        = -2
```

**In ML:**
- Dot product of features and weights → model prediction
- Dot product of two vectors → cosine similarity measurement

---

### 3.4 Cross Product

For 3D vectors, produces a new vector perpendicular to both.

```
u × v = [u₂v₃ - u₃v₂,  u₃v₁ - u₁v₃,  u₁v₂ - u₂v₁]
```

**In ML:** Used in computer vision and 3D geometry tasks.

---

## 4. Linear Transformations

Linear transformations are basic operations in linear algebra that change vectors and matrices while keeping important properties like straight lines and proportionality. In machine learning, they are key for tasks like preparing data, creating features, and training models.

### Definition

A transformation T is linear if it satisfies:

```
Additivity:    T(u + v) = T(u) + T(v)
Homogeneity:   T(kv)   = k · T(v)
```

### Common Types in ML

| Transformation | Description | ML Use Case |
|---------------|-------------|-------------|
| **Translation** | Shift data by subtracting the mean | Data centering |
| **Scaling** | Normalize features to same range | Prevent one feature dominating |
| **Rotation** | Turn/rotate data in space | Computer vision, robotics |

```
Example — Data Centering (Translation):
  Original: [22, 85, 6]
  Mean:     [22.3, 84.3, 6.3]
  Centered: [-0.3, 0.7, -0.3]

Example — Feature Scaling (Normalization):
  x_scaled = (x - min) / (max - min)   ← MinMax
  x_scaled = (x - μ) / σ               ← Standard scaling
```

---

## 5. Matrix Operations

Matrix operations are central to linear algebra and widely used in machine learning for data handling, transformations, and model training.

### 5.1 Matrix Multiplication

Combines two matrices by taking the dot product of rows and columns. Used in feature transformations, parameter computation, and neural network operations.

```
     ┌ 2  1 ┐         ┌ 3  0 ┐
A =  │      │    B =  │      │
     └ 1  2 ┘         └ 1  2 ┘

         ┌ (2×3+1×1)  (2×0+1×2) ┐   ┌ 7  2 ┐
A × B =  │                       │ = │      │
         └ (1×3+2×1)  (1×0+2×2) ┘   └ 5  4 ┘

Rule: A(m×n) × B(n×p) = C(m×p)
      (inner dimensions must match)
```

**In ML:**
```
Neural network layer:  output = W · input + bias
Linear Regression:     predictions = X · weights
```

---

### 5.2 Transpose

Flips a matrix across its diagonal — rows become columns. Denoted by Aᵀ.

```
    ┌ 1  2  3 ┐           ┌ 1  4 ┐
A = │         │    Aᵀ =   │ 2  5 │
    └ 4  5  6 ┘           └ 3  6 ┘

Shape: A (2×3) → Aᵀ (3×2)
```

**In ML:** Used in the normal equation for Linear Regression: `XᵀX β = XᵀY`

---

### 5.3 Inverse

The matrix A⁻¹ satisfies A · A⁻¹ = I. Exists only if det(A) ≠ 0. Used in solving equations and optimization.

```
A · A⁻¹ = I   (Identity matrix)

If  A = ┌ 2  1 ┐    then   A⁻¹ = ┌  2/3  -1/3 ┐
        └ 1  2 ┘                   └ -1/3   2/3 ┘

Check: A · A⁻¹ = I ✓
```

**In ML:** Normal Equation: `β = (XᵀX)⁻¹ Xᵀ Y`

---

### 5.4 Determinant

A scalar value indicating whether a matrix is invertible. If det(A) = 0, the matrix cannot be inverted.

```
For 2×2 matrix:
    ┌ a  b ┐
A = │      │     det(A) = ad - bc
    └ c  d ┘

Example:
    ┌ 2  1 ┐
A = │      │     det(A) = (2×2) - (1×1) = 3  ≠ 0  → invertible ✓
    └ 1  2 ┘
```

---

## 6. Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors describe how matrices transform space, making them fundamental in many ML algorithms.

### Definition

```
A · v  =  λ · v

Where:
  A  =  square matrix (transformation)
  v  =  eigenvector  (direction that doesn't change)
  λ  =  eigenvalue   (how much it stretches/compresses)
```

| Term | Symbol | Meaning |
|------|--------|---------|
| **Eigenvalue** | λ | Scalar showing how much transformation stretches or compresses along a direction |
| **Eigenvector** | v | Non-zero vector that only scales — does not change direction — under transformation |

### Worked Example

For matrix A = [[2,1],[1,2]], solving det(A - λI) = 0 gives λ₁ = 1, λ₂ = 3.

```
    ┌ 2  1 ┐
A = │      │
    └ 1  2 ┘

Step 1 — Characteristic equation:
  det(A - λI) = 0
  det ┌ 2-λ   1  ┐ = 0
      └  1   2-λ ┘
  (2-λ)(2-λ) - (1)(1) = 0
  λ² - 4λ + 3 = 0
  (λ - 1)(λ - 3) = 0

Step 2 — Eigenvalues:
  λ₁ = 1    λ₂ = 3

Step 3 — Eigenvectors:
  λ₁ = 1  →  v₁ = [ 1, -1]
  λ₂ = 3  →  v₂ = [ 1,  1]
```

### Eigen Decomposition

```
A  =  Q · Λ · Q⁻¹

Where:
  Q  =  matrix of eigenvectors (columns)
  Λ  =  diagonal matrix of eigenvalues
  Q⁻¹ = inverse of eigenvector matrix
```

### Applications in ML

In Dimensionality Reduction (PCA), it keeps directions with the largest eigenvalues (most variance). In Matrix Factorization (SVD, NMF), it breaks large datasets into smaller, structured parts for feature extraction.

```
PCA Process:
  1. Compute covariance matrix of data
  2. Find eigenvalues and eigenvectors
  3. Sort eigenvalues (largest = most variance)
  4. Keep top k eigenvectors as principal components
  5. Project data onto new reduced space

Large eigenvalue → direction captures lots of variance → keep it
Small eigenvalue → direction adds little info → discard it
```

---

## 7. Solving Linear Systems

Linear systems are common in machine learning for parameter estimation and optimization.

### System of Linear Equations

```
General form:  Ax = b

Example:
  2x + y = 5     ┌ 2  1 ┐ ┌ x ┐   ┌ 5 ┐
  x + 2y = 4  →  │      │ │   │ = │   │
                  └ 1  2 ┘ └ y ┘   └ 4 ┘
```

### Method 1 — Gaussian Elimination

Transforms a matrix into row-echelon form using row operations.

```
Steps:
  1. Forward Elimination  → make entries below diagonal zero
  2. Back Substitution    → solve variables from last row upward
  3. Pivoting             → swap rows to avoid division by zero

Augmented matrix:
  ┌ 2  1 | 5 ┐   →   ┌ 2  1  | 5   ┐   →   x = 2, y = 1
  └ 1  2 | 4 ┘       └ 0  1.5| 1.5 ┘
```

### Method 2 — LU Decomposition

Splits a matrix into Lower (L) and Upper (U) triangular matrices. Solves systems efficiently using forward and back substitution.

```
A  =  L · U

     ┌ 1  0 ┐   ┌ 2  1 ┐
A =  │      │ × │      │
     └ 0.5  1┘   └ 0  1.5┘
       L           U

Use: More efficient than Gaussian for multiple right-hand sides
```

### Method 3 — QR Decomposition

Splits a matrix into Orthogonal (Q) and Upper triangular (R). Useful for least squares problems and eigenvalue computation.

```
A  =  Q · R

Q = orthogonal matrix (columns are perpendicular unit vectors)
R = upper triangular matrix

Use in ML: Solving Ax = b in least squares regression
```

---

## 8. Applications in Machine Learning

### 8.1 PCA — Principal Component Analysis

Reduces dimensionality by computing covariance, eigenvalues/eigenvectors and projecting data onto principal components.

```
Algorithm:
  1. Standardize data  →  X_scaled
  2. Covariance matrix →  C = (1/n) XᵀX
  3. Eigen decomposition of C
  4. Sort eigenvectors by eigenvalue (descending)
  5. Project data:  X_pca = X · top_k_eigenvectors

Before PCA:  100 features  →  After PCA:  10 components
             95% variance retained
```

### 8.2 SVD — Singular Value Decomposition

Factorizes a matrix into A = UΣVᵀ, used for dimensionality reduction, compression, and noise filtering.

```
A  =  U · Σ · Vᵀ

Where:
  U  =  left singular vectors  (m × m)
  Σ  =  diagonal singular values (m × n)
  Vᵀ =  right singular vectors (n × n)

Use cases:
  → Recommender systems (matrix factorization)
  → Image compression
  → Noise reduction in data
  → Natural language processing (LSA)
```

### 8.3 Linear Regression

Models relationships via matrix form Y = Xβ + ε, solved using the normal equation XᵀXβ = XᵀY.

```
Matrix form:
  Y  =  X · β  +  ε

  Y  =  target values          (n × 1)
  X  =  feature matrix         (n × p)
  β  =  weights/coefficients   (p × 1)
  ε  =  error/residuals

Normal Equation (closed-form solution):
  β  =  (XᵀX)⁻¹ · Xᵀ · Y
```

### 8.4 SVM — Support Vector Machines

Uses the kernel trick and optimization to find decision boundaries for classification and regression.

```
Decision boundary:  w · x + b = 0

Where:
  w  =  weight vector (perpendicular to boundary)
  x  =  input features
  b  =  bias term

Kernel trick:
  K(x, z) = φ(x) · φ(z)   ← avoids explicit transformation
  Linear:     K = xᵀz
  RBF:        K = exp(-γ||x-z||²)
```

### 8.5 Neural Networks

Neural networks depend on matrix multiplications, gradient descent, and weight initialization for training deep models.

```
Forward Pass (one layer):
  output = activation( W · input + b )

  W     = weight matrix  (neurons_out × neurons_in)
  input = input vector   (neurons_in × 1)
  b     = bias vector    (neurons_out × 1)

Backpropagation:
  Gradients computed using chain rule + matrix calculus
  W_new = W - α · ∇L     (gradient descent update)
```

### ML Applications Summary Table

| Algorithm | Linear Algebra Used |
|-----------|-------------------|
| **PCA** | Covariance matrix, eigendecomposition |
| **SVD** | Matrix factorization A = UΣVᵀ |
| **Linear Regression** | Matrix equation Y = Xβ, normal equation |
| **SVM** | Dot products, kernel functions |
| **Neural Networks** | Matrix multiplications, gradient descent |
| **K-Means** | Vector distances (norms) |
| **Recommendation Systems** | Matrix factorization, SVD |

---


## 9. Cheat Sheet

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LINEAR ALGEBRA FOR ML — QUICK REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FUNDAMENTALS
  Scalar   →  single number          k = 3
  Vector   →  array of numbers       v = [2, -1, 4]
  Matrix   →  2D array of numbers    A (m × n)

  VECTOR OPERATIONS
  Addition       u + v  =  element-wise sum
  Subtraction    u - v  =  element-wise difference
  Scalar mult    k · v  =  each element × k
  Dot product    u · v  =  Σ uᵢvᵢ    (scalar result)
  Magnitude      ||v||  =  √(Σ vᵢ²)

  MATRIX OPERATIONS
  Multiply       A × B  →  dot product rows × columns
  Transpose      Aᵀ     →  rows become columns
  Inverse        A⁻¹    →  A · A⁻¹ = I  (if det ≠ 0)
  Determinant    det(A) →  scalar; 0 = not invertible

  EIGENDECOMPOSITION
  A · v = λ · v          (definition)
  det(A - λI) = 0        (find eigenvalues)
  A = Q · Λ · Q⁻¹        (eigen decomposition)

  LINEAR SYSTEMS
  Ax = b → x = A⁻¹b      (direct inverse)
  Normal Equation:         β = (XᵀX)⁻¹ Xᵀ Y

  SVD
  A = U · Σ · Vᵀ

  KEY ML APPLICATIONS
  PCA              →  eigenvalues of covariance matrix
  SVD              →  matrix factorization A = UΣVᵀ
  Linear Regression→  β = (XᵀX)⁻¹ Xᵀ Y
  Neural Networks  →  W · x + b  (layer computation)
  SVM              →  w · x + b = 0  (decision boundary)

  NUMPY COMMANDS
  np.dot(u, v)           →  dot product
  np.linalg.norm(v)      →  vector magnitude
  A @ B                  →  matrix multiplication
  A.T                    →  transpose
  np.linalg.inv(A)       →  inverse
  np.linalg.det(A)       →  determinant
  np.linalg.eig(A)       →  eigenvalues & eigenvectors
  np.linalg.svd(A)       →  SVD decomposition
  np.linalg.solve(A, b)  →  solve Ax = b

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Further Reading

- 🌐 [Linear Algebra for ML — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/ml-linear-algebra-operations/)
---

*Notes compiled from GeeksforGeeks | Linear Algebra for Machine Learning | May 2026*
