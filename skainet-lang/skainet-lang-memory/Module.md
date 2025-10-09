# skainet-lang-memory

## Overview

## memory layout

row major **BCWH**

$$
\text{stride}[i] = \prod_{j=i+1}^{n-1} \text{dims}[j]
$$

## Classes

DenseXXXTensorDataMemory implement dense linear memoty holding tensors data. 