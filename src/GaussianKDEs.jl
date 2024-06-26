module GaussianKDEs

using Distributions
using LinearAlgebra
using Random
using StatsFuns

export KDE, BoundedKDE, bw_opt_kde

square(x) = x*x

# Required because KernelDensity.jl uses axis-aligned KDEs.

struct KDEnD{T} <: ContinuousMultivariateDistribution where T <: Number
    pts::Matrix{T}
    chol_bw::Cholesky{T, Matrix{T}}
end
struct KDE1D{T} <: ContinuousUnivariateDistribution where T <: Number
    pts::Vector{T}
    bw::T
end
struct BoundedKDE{T} <: ContinuousUnivariateDistribution where T <: Number
    kde::KDE1D{T}
    lb::Union{T, Nothing}
    ub::Union{T, Nothing}
end

raw"""
    KDE(pts; bandwidth_factor=1, lower=nothing, upper=nothing)

Return a kernel density object out of the given points.

`pts` should have shape `(ndim, npts)` or `(npts,)` for a 1-dimensional KDE.  If
bounds are given, the resulting object will implement a reflective boundary
condition at the given bounds.

`bandwidth_factor` is a scaling factor for the bandwidth of the KDE, relative to
Scott's rule, with the KDE covariance matrix given by 

``\Sigma_\mathrm{kde} = \frac{f^2}{N^{2/(d+4)}} \Sigma_\mathrm{data}``

where ``N`` is the number of points in the KDE, ``d`` is the number of
dimensions, and ``f`` is the bandwidth factor.
"""
function KDE(pts::Matrix{T}; bandwidth_factor=1) where T <: Number
    nd, np = size(pts)
    S = (bandwidth_factor*bandwidth_factor) .* cov(pts') ./ np^(2/(nd+4))

    KDEnD(pts, cholesky(S))
end

function KDE(pts::Vector{T}; bandwidth_factor=1) where T <: Number
    KDE1D(pts, bandwidth_factor * std(pts) / length(pts)^(1/5))
end

function BoundedKDE(pts::Vector{T}; bandwidth_factor=1, lower=nothing, upper=nothing) where T <: Number
    k = KDE(pts; bandwidth_factor=bandwidth_factor)
    if lower === nothing && upper === nothing
        error("Bounded KDE created with no bounds!")
    elseif lower === nothing
        BoundedKDE(k, lower, convert(T, upper))
    elseif upper === nothing
        BoundedKDE(k, convert(T, lower), upper)
    else
        BoundedKDE(k, convert(T, lower), convert(T, upper))
    end
end

"""The number of dimensions in the KDE."""
function ndim(k::KDEnD)
    size(k.pts, 1)
end
"""The number of points stored in the KDE."""
function npts(k::KDEnD)
    size(k.pts, 2)
end
function npts(k::KDE1D)
    length(k.pts)
end
function npts(k::BoundedKDE)
    npts(k.kde)
end

"""The cholesky factor of the KDE bandwidth matrix."""
function chol_bw(k::KDEnD)
    k.chol_bw
end
function bw(k::KDE1D)
    k.bw
end
function bw(k::BoundedKDE)
    bw(k.kde)
end

Distributions.length(k::KDEnD) = ndim(k)
Distributions.sampler(k::KDEnD) = k
Distributions.eltype(::KDEnD{T}) where T <: Number = T
function Distributions._rand!(rng::AbstractRNG, k::KDEnD{T}, x::AbstractVector{T}) where T <: Number
    i = rand(rng, 1:npts(k))
    x .= k.pts[:, i] .+ k.chol_bw.L * randn(rng, T, ndim(k))
end
function Distributions._logpdf(k::KDEnD{T}, x::AbstractArray{T}) where T <: Number
    lp = -Inf
    for j in 1:npts(k)
        r = x .- k.pts[:, j]
        lp = logaddexp(lp, -(r' * (k.chol_bw \ r))/2)
    end

    lp = lp - sum(log.(sqrt(2*pi).*diag(k.chol_bw.L))) - log(npts(k))
    lp
end

function Distributions.rand(rng::AbstractRNG, k::KDE1D{T}) where T <: Number
    i = rand(rng, 1:npts(k))
    k.pts[i] + k.bw * randn(rng, T)
end
function Distributions.sampler(k::KDE1D{T}) where T <: Number
    k
end
function Distributions.logpdf(k::KDE1D{T}, x::T) where T <: Number
    lognorm = -log(2*pi)/2 - log(k.bw)
    lp = -Inf
    for p in k.pts
        lp = logaddexp(lp, -0.5*square((x-p)/k.bw))
    end
    lp + lognorm - log(npts(k))
end
function Distributions.logpdf(k::KDE1D{T}, x::S) where T <: Number where S <: Number
    logpdf(k, convert(T, x))
end
function Distributions.minimum(k::KDE1D{T}) where T <: Number
    -Inf
end
function Distributions.maximum(k::KDE1D{T}) where T <: Number
    Inf
end
function Distributions.insupport(k::KDE1D{T}, x) where T <: Number
    true
end

function Distributions.rand(rng::AbstractRNG, k::BoundedKDE{T}) where T <: Number
    x = rand(rng, k.kde)
    while true
        if k.lb !== nothing && x < k.lb
            x = 2*k.lb - x
        elseif k.ub !== nothing && x > k.ub
            x = 2*k.ub - x
        else
            return x
        end
    end
end
Distributions.sampler(k::BoundedKDE) = k
function Distributions.logpdf(k::BoundedKDE{T}, x::S) where T<:Number where S<:Number
    lp = logpdf(k.kde, x)
    if k.lb !== nothing
        lp = logaddexp(lp, logpdf(k.kde, 2*k.lb - x))
    end
    if k.ub !== nothing
        lp = logaddexp(lp, logpdf(k.kde, 2*k.ub - x))
    end
    lp
end
function Distributions.minimum(k::BoundedKDE{T}) where T <: Number
    if k.lb !== nothing
        k.lb
    else
        -Inf
    end
end
function Distributions.maximum(k::BoundedKDE{T}) where T <: Number
    if k.ub !== nothing
        k.ub
    else
        Inf
    end
end
function Distributions.insupport(k::BoundedKDE{T}, x) where T <: Number
    (x > minimum(k)) && (x < maximum(k))
end

function golden_search(f, a, z)
    phi = (1 + sqrt(5))/2

    r = 2*phi + 1
    d = (z-a)/r
    b = a + phi*d
    c = b + d

    fa = f(a)
    fz = f(z)
    fb = f(b)
    fc = f(c)

    while abs(z-a) > 1e-8
        if fb < fc
            z, fz = c, fc
            c, fc = b, fb
            b = a + (z-c)
            fb = f(b)
        else
            a, fa = b, fb
            b, fb = c, fc
            c = z - (b - a)
            fc = f(c)
        end 
    end

    return (a+z)/2
end

"""
    bw_opt_kde(pts, test_pts; test_low=-1, test_high=1)

Will return a KDE based on `pts` whose bandwidth factor maximizes the likelihood
of `test_pts`.

`test_low` and `test_high` are the bounds on the natural log of the bandwidth
factor for the optimizing search. 
"""
function bw_opt_kde(pts, test_pts; test_low=-1, test_high=1)

    function neg_logp(log_bw)
        kde = KDE(pts; bandwidth_factor=exp(log_bw))
        -sum([logpdf(kde, test_pts[:,j]) for j in axes(test_pts,2)])
    end

    bw = exp(golden_search(neg_logp, test_low, test_high))

    KDE(pts; bandwidth_factor=bw)
end

end # module GaussianKDE
