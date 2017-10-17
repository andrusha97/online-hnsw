Online HNSW
===========
An implementation of the HNSW index for approximate nearest neighbors search for C++14,
that supports incremental insertion and removal of elements.

License
=======
Original parts of this project are licensed under the terms of the Apache 2.0 license.
This project also includes:
* A copy of
[the hopscotch-map project](https://github.com/Tessil/hopscotch-map), which is licensed under the MIT license.
It resides in `include/hnsw/containers/hopscotch-map-1.4.0`.
* Parts of [nmslib](https://github.com/searchivarius/nmslib) which are also licensed under the Apache 2.0 license.

Individual source files have corresponding copyright and license info.

References
==========
* Malkov, Y.A., Yashunin, D.A. [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)
