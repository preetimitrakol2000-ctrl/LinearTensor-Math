# LinearTensor-Math

A low-level implementation of a deep learning multi-dimensional matrix framework written in pure C99. This framework avoids multi-pointer dynamic fragmentation (`double**`) by storing spatial weight layers inside flat, contiguous 1D arrays to maximize hardware cache hits during execution loops.

## 📈 Functional Scope
* **Row-Major Memory Transformation:** Maps index calculations via `(i * columns + j)`.
* **Sigmoid Activation Mapping:** Manual execution of element-wise transformations.
