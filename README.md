# `GaussianKDEs.jl`: Gaussian Kernel Density Estimators in Julia

Why `GaussianKDEs.jl` and not `KernelDensity.jl`?  If you want a one-dimensional
KDE, it is probably best to use `KernelDensity.jl` (though `GaussianKDEs.jl`
also has 1D KDEs).  However, if you want a multi-dimensional KDE, then
`KernelDensity.jl` is limited to at most 2D, and even in 2D uses axis-aligned
kernels to estimate the density; this package can construct KDEs in arbitrary
dimension.

`GaussianKDEs.jl` has some other features that could be attractive:

* It presents a unified interface for both 1D and multi-dimensional KDEs.
* It allows control over the kernel bandwidth in all dimensions.
* It implements a `BoundedKDE` type for 1D KDEs on bounded domains.

## Installation

```julia
using Pkg
Pkg.add("https://github.com/farr/GaussianKDEs.jl.git")
```

## Usage

```julia
using Distributions
using GaussianKDEs

k1 = KDE(randn(1024))
pdf(k1, 0) # Should be close to 1/sqrt(2*pi), the PDF for a unit-normal at the origin

npts = 1024
ndim = 4
k4 = KDE(randn(ndim, npts))
pdf(k4, zeros(ndim)) # Should be close to 1/(2*pi)^(ndim/2), the PDF for a unit-normal at the origin

ku = BoundedKDE(rand(1024), lower=0, upper=1)
pdf(ku, 0.5) # Should be close to 1, the PDF for a uniform distribution on [0,1]
pdf(ku, 0) # Should be close to 1, the PDF for a uniform distribution on [0,1]
almost_uniform = rand(ku, 1024) # Draw from the bounded distribution implied by the KDE.
```