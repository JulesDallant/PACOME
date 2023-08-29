module Simus

using Distributions
using Random
using Random: GLOBAL_RNG
using EasyFITS
import Random: rand, rand!
using Statistics
using Utils: split_on_threads

struct SumSquaredRectifiedGaussian
    ndof::Int
    mean::Vector{AbstractFloat}
    dev::Vector{AbstractFloat}
end

OwnPDF = Union{SumSquaredRectifiedGaussian}

"""
   SumSquaredRectifiedGaussian

PACOME cost function distribution in the absence of source.

"""
function SumSquaredRectifiedGaussian(ndof::Int,
                                mean::T,
                                dev::T) where {T<:AbstractFloat}
   return SumSquaredRectifiedGaussian(ndof, ones(T, ndof)*mean,
                                       ones(T, ndof)*dev)
end

function SumSquaredRectifiedGaussian(ndof::Int)
   return SumSquaredRectifiedGaussian(ndof, zeros(AbstractFloat, ndof),
                                       ones(AbstractFloat, ndof))
end

"""
    ndof(pdf)

yields the number of degrees of freedom of the distribution `pdf`.

"""
ndof(pdf::OwnPDF) = getfield(pdf, :ndof)
mean(pdf::OwnPDF) = getfield(pdf, :mean)
dev(pdf::OwnPDF) = getfield(pdf, :dev)

"""
    relu(x)

yields the *ReLU* (Rectified Linear Unit) function of `x`, that is
`max(x,zero(x))` but computed more efficiently and accounting for NaNs.

"""
relu(x::Number) = ifelse(x < zero(x), zero(x), x)

"""
    rand([rng=Random.GLOBAL_RNG,] [T=Float64,] pdf::SumSquaredRectifiedGaussian, [dims...])

yields a pseudo-random realization of the distribution `pdf`.

"""
rand(pdf::OwnPDF) = rand(GLOBAL_RNG, pdf)
rand(T::Type{<:AbstractFloat}, pdf::OwnPDF) = rand(GLOBAL_RNG, T, pdf)
rand(rng::AbstractRNG, pdf::OwnPDF) = rand(rng, Float64, pdf)

function rand(rng::AbstractRNG, T::Type{<:AbstractFloat}, pdf::SumSquaredRectifiedGaussian)
    s = zero(T)
    for i in 1:ndof(pdf)
        # x = randn(rng, T)
        x = rand(Normal(mean(pdf)[i],dev(pdf)[i]))
        y = relu(x)^2
        s += y
    end
    return s
end

# Dimensions specified individually.
function rand(pdf::OwnPDF, dims::Integer...)
    return rand(pdf, dims)
end
function rand(T::Type{<:AbstractFloat}, pdf::OwnPDF,
              dims::Integer...)
    return rand(T, pdf, dims)
end
function rand(rng::AbstractRNG, pdf::OwnPDF,
                   dims::Integer...)
    return rand(rng, pdf, dims)
end
function rand(rng::AbstractRNG, T::Type{<:AbstractFloat},
              pdf::OwnPDF, dims::Integer...)
    return rand!(rng, T, pdf, dims)
end

# Dimensions specified as a tuple.
for type in (:Int, :Integer)
    @eval begin
        function rand(pdf::OwnPDF,
                      dims::NTuple{N,$type}) where {N}
            return rand(GLOBAL_RNG, pdf, dims)
        end
        function rand(T::Type{<:AbstractFloat},
                      pdf::OwnPDF,
                      dims::NTuple{N,$type}) where {N}
            return rand(GLOBAL_RNG, T, pdf, dims)
        end
        function rand(rng::AbstractRNG,
                      pdf::OwnPDF,
                      dims::NTuple{N,$type}) where {N}
            return rand(rng, Float64, pdf, dims)
        end
        function rand(rng::AbstractRNG,
                      T::Type{<:AbstractFloat},
                      pdf::OwnPDF,
                      dims::NTuple{N,$type}) where {N}
            return rand!(rng, Array{T}(undef, dims), pdf)
        end
    end
end

"""
    rand!([rng=Random.GLOBAL_RNG], A, pdf::OwnPDF) -> A

populates array `A` with pseudo-random values drawn from distribution `pdf`.

"""
rand!(A::AbstractArray{<:AbstractFloat}, pdf::OwnPDF) =
    rand!(GLOBAL_RNG, A, pdf)

function rand!(rng::AbstractRNG,
               A::AbstractArray{<:AbstractFloat},
               pdf::OwnPDF)
    @inbounds @simd for i in eachindex(A)
        A[i] = rand(rng, eltype(A), pdf)
    end
    return A
end

"""
    quantile(pdf, p; N) -> q

builds the pseudo-random empirical distribution of a squared rectified Normal
distribution `pdf` to estimate the lower bound of the quantile of probability
`p`. The number of points drawn from the random distribution is automatically
tuned such that an average of `N` points are found above the quantile, it is set
to `N=1000` by default.

"""
function quantile(pdf::OwnPDF, p::AbstractFloat; N::Int=1000)

    x = rand(pdf,round(Int, N/(1-p)))
    return Distributions.quantile(x, p)
end

"""
    confint(d, width; tail)
    confint(d; α, tail)


Return a confidence interval of the distribution `d` (can be theoretical of
empirical).

Use the positional argument `level` or the keyword-argument `α`
to set the width of the interval, where `level=1-α`.

For a one-sided interval, where one "tail" extends to infinity, set
which tail should extend to infinity with the keyword argument `tail`.
Possible values are `:none` (default value), `:left` and `:right`.

"""
function confint(d::Union{Distribution, Vector{T}};
                 α::T, tail::Symbol=:none) where {T<:AbstractFloat}
    !(0 ≤ α ≤ 1) && error("The given interval width ($(1-α)) is not between 0 and 1.")
    if tail == :none
        return (quantile(d, α / 2), quantile(d, 1 - α / 2))
    elseif tail == :left
        return (-Inf, quantile(d, 1 - α))
    elseif tail == :right
        return (quantile(d, α), Inf)
    else
        error("Tail not recognized. Choose from :none, :left or :right.")
    end
end

function confint(d::Union{Distribution, Vector{T}},
                 level::T; kwds...) where {T<:AbstractFloat}
    return confint(d; α=1-level, kwds...)
end

"""
    model_conflevel(dof, ρ) -> conflevel

computes an estimation of the empirical quantile of the distribution of `dof`
degrees of freedom for confidence level `ρ`.

"""
const model_param_file = "./data/abacus_conflevel_distrib_model_param.fits"

model_param = read(FitsArray, model_param_file)

function model_conflevel(dof::Int, ρ::T) where {T<:AbstractFloat}
    @assert 1 ≤ dof ≤ 100
    a, n0, Un0, r = model_param[:,dof]
    return a^(-log10(ρ)-n0)*(Un0-r)+r
end

"""
    model_param_INJECTIONS(ρ) -> conflevel

computes an estimation of the empirical quantile of the distribution of the
simulated data degrees of freedom for confidence level `ρ`.

"""
const model_param_SIMU_DATA_file = "./data/conflevel_distrib_model_param_SIMU_DATA.fits"

model_param_SIMU_DATA = read(FitsArray, model_param_SIMU_DATA_file)

function model_conflevel_SIMU_DATA(ρ::T) where {T<:AbstractFloat}
    a, n0, Un0, r = model_param_SIMU_DATA
    return a^(-log10(ρ)-n0)*(Un0-r)+r
end

end # module
