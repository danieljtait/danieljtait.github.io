---
layout: post
title: "Pydygp and Scikit-Learn Kernels"
---

## Introduction

$$
\operatorname{Cov}\left\{f(\mathbf{x}), \frac{\partial f(\mathbf{y})}{\partial y_p } \right\} = \frac{\partial k(\mathbf{x}, \mathbf{y})}{\partial y_p }, \qquad \operatorname{Cov}\left\{\frac{\partial f(\mathbf{x})}{\partial x_p}, \frac{\partial f(\mathbf{y})}{\partial y_q } \right\} = \frac{\partial^2 k(\mathbf{x}, \mathbf{y})}{\partial x_p \partial y_q },
$$