## sl-BAMM
This repo contains code accompaning the paper, Averaged Method of Multipliers for Bi-Level Optimization without Lower-Level Strong Convexity (Liu et al., ICML 2023). It includes code for running the numerical example under LL Merely convex assumption and LL Stronly Convex assumption.


### Abstract
Gradient methods have become mainstream techniques for Bi-Level Optimization (BLO) in learning fields. The validity of existing works heavily rely on either a restrictive Lower- Level Strong Convexity (LLSC) condition or on solving a series of approximation subproblems with high accuracy or both. In this work, by averaging the upper and lower level objectives, we propose a single loop Bi-level Averaged Method of Multipliers (sl-BAMM) for BLO that is simple yet efficient for large-scale BLO and gets rid of the limited LLSC restriction. We further provide non-asymptotic convergence analysis of sl-BAMM towards KKT stationary points, and the comparative advantage of our analysis lies in the absence of strong gradient boundedness assumption, which is always required by others. Thus our theory safely captures a wider variety of applications in deep learning, especially where the upper-level objective is quadratic w.r.t. the lower-level variable. Experimental results demonstrate the superiority of our method.

### Dependencies
This code mainly requires the following:
- Python 3.*
- Pytorch
- [Torchmeta](https://github.com/tristandeleu/pytorch-meta) 

### Citation

You are encouraged to cite the following paper:
- Risheng Liu, Yaohua Liu, Wei Yao, Shangzhi Zeng, Jin Zhang. ["Averaged Method of Multipliers for Bi-Level Optimization without Lower-Level Strong Convexity"](https://arxiv.org/abs/2302.03407). ICML, 2023.

### License 

MIT License

Copyright (c) 2021 Vision Optimizaion Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
