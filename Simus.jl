module Simus

using Distributions
using Random
using Random: GLOBAL_RNG
using EasyFITS
import Random: rand, rand!
using Statistics
using Utils: split_on_threads


struct CosineUniform <: Sampleable{Univariate,Continuous}
    a::AbstractFloat
    b::AbstractFloat
    CosineUniform(a, b) = a > b ? error("out or order, $a > $b") : new(a,b)
end

function Distributions.pdf(d::CosineUniform, x::AbstractFloat)
    cosa, cosb = cos(d.a), cos(d.b)
    if cosa < cosb
        return sin(x)/(cosb-cosa)
    else
        return sin(x)/(cosa-cosb)
    end
end

function rand(rng::AbstractRNG, d::CosineUniform)
    cosa, cosb = cos(d.a), cos(d.b)
    if cosa < cosb
        return acos(rand(Uniform(cosa, cosb)))
    else
        return acos(rand(Uniform(cosb, cosa)))
    end
end

struct CosineDegUniform <: Sampleable{Univariate,Continuous}
    a::AbstractFloat
    b::AbstractFloat
    CosineDegUniform(a, b) = a > b ? error("out or order, $a > $b") : new(a,b)
end

function Distributions.pdf(d::CosineDegUniform, x::AbstractFloat)
    cosa, cosb = cos(deg2rad(d.a)), cos(deg2rad(d.b))
    if cosa < cosb
        return sin(deg2rad(x))/(cosb-cosa)
    else
        return sin(deg2rad(x))/(cosa-cosb)
    end
end

function rand(rng::AbstractRNG, d::CosineDegUniform)
    cosa, cosb = cos(deg2rad(d.a)), cos(deg2rad(d.b))
    if cosa < cosb
        return rad2deg(acos(rand(Uniform(cosa, cosb))))
    else
        return rad2deg(acos(rand(Uniform(cosb, cosa))))
    end
end

struct SumSquaredRectifiedGaussian
    ndof::Int
    mean::Vector{AbstractFloat}
    dev::Vector{AbstractFloat}
end

struct SumSquaredRectifiedSignedGaussian
    ndof::Int
    mean::Vector{AbstractFloat}
    dev::Vector{AbstractFloat}
end

struct SignedMultiEpochSNR
    ndof::Int
    mean::Vector{AbstractFloat}
    dev::Vector{AbstractFloat}
end

struct MedianGaussian
    ndof::Int
    mean::Vector{AbstractFloat}
    dev::Vector{AbstractFloat}
end

# struct MeanGaussian
#     ndof::Int
#     mean::Vector{AbstractFloat}
#     dev::Vector{AbstractFloat}
# end

OwnPDF = Union{SumSquaredRectifiedGaussian,
               SumSquaredRectifiedSignedGaussian,
               SignedMultiEpochSNR, MedianGaussian}

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

function MedianGaussian(ndof::Int)
   return MedianGaussian(ndof, zeros(AbstractFloat, ndof),
                               ones(AbstractFloat, ndof))
end

function MedianGaussian(ndof::Int,
                        mean::T,
                        dev::T) where {T<:AbstractFloat}
   return MedianGaussian(ndof, ones(T, ndof)*mean,
                               ones(T, ndof)*dev)
end

function SumSquaredRectifiedSignedGaussian(ndof::Int)
   return SumSquaredRectifiedSignedGaussian(ndof, zeros(AbstractFloat, ndof),
                               ones(AbstractFloat, ndof))
end

function SumSquaredRectifiedSignedGaussian(ndof::Int,
                        mean::T,
                        dev::T) where {T<:AbstractFloat}
   return SumSquaredRectifiedSignedGaussian(ndof, ones(T, ndof)*mean,
                               ones(T, ndof)*dev)
end

function SignedMultiEpochSNR(ndof::Int)
   return SignedMultiEpochSNR(ndof, zeros(AbstractFloat, ndof),
                               ones(AbstractFloat, ndof))
end

function SignedMultiEpochSNR(ndof::Int,
                        mean::T,
                        dev::T) where {T<:AbstractFloat}
   return SignedMultiEpochSNR(ndof, ones(T, ndof)*mean,
                               ones(T, ndof)*dev)
end

# function MeanGaussian(ndof::Int)
#    return MeanGaussian(ndof, zeros(AbstractFloat, ndof),
#                                ones(AbstractFloat, ndof))
# end
#
# function MeanGaussian(ndof::Int,
#                         mean::T,
#                         dev::T) where {T<:AbstractFloat}
#    return MeanGaussian(ndof, ones(T, ndof)*mean,
#                                ones(T, ndof)*dev)
# end

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

function rand(rng::AbstractRNG, T::Type{<:AbstractFloat}, pdf::SumSquaredRectifiedSignedGaussian)
    s = zero(T)
    for i in 1:ndof(pdf)
        x = rand(Normal(mean(pdf)[i],dev(pdf)[i]))
        s += sign(x) * x^2
    end
    return s
end

function rand(rng::AbstractRNG, T::Type{<:AbstractFloat}, pdf::SignedMultiEpochSNR)
    s = zero(T)
    for i in 1:ndof(pdf)
        x = rand(Normal(mean(pdf)[i],dev(pdf)[i]))
        s += sign(x)*x^2
    end
    return sign(s) * sqrt(abs(s))
end

function rand(rng::AbstractRNG, T::Type{<:AbstractFloat}, pdf::MedianGaussian)
    s = Vector{T}(undef, ndof(pdf))
    for i in 1:ndof(pdf)
        s[i] = rand(Normal(mean(pdf)[i],dev(pdf)[i]))
    end
    return median(s)
end

# function rand(rng::AbstractRNG, T::Type{<:AbstractFloat}, pdf::MeanGaussian)
#     s = 0
#     for i in 1:ndof(pdf)
#         s += rand(Normal(mean(pdf)[i],dev(pdf)[i]))
#     end
#     return s/ndof(pdf)
# end

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

function quantile2(pdf::OwnPDF, pmax::T;
                   N::Int=1000, p0::T=1e-1) where{T<:AbstractFloat}

    (pmax > p0) && error("pmax = $pmax should be smaller than p0 = $p0")
    nthreads = Threads.nthreads()

    ps = Vector{T}()
    push!(ps, p0)
    i = round(Int,abs(log10(p0)))
    while round(last(ps)/10, digits=i+21) > pmax
        push!(ps, last(ps)/10)
        i += 1
    end
    push!(ps, pmax)

    Nps = ceil.(Int, N./ps)

    qs = fill!(Vector{T}(undef, length(ps)), NaN)

    Np_used = Vector{T}(undef, length(ps))

    D = Vector{T}(undef, Nps[1])
    I = split_on_threads(Nps[1], nthreads)
    Threads.@threads for n in 1:nthreads
        for k in first(I[n]):last(I[n])
            D[k] = rand(pdf)
        end
    end
    qs[1] = Distributions.quantile(D, 1-p0)
    D = sort(D[findall(x -> x > qs[1], D)])
    Np_used[1] = length(D)

    length(ps) == 1 ? (return ps, qs) : nothing

    pidx = 1
    while pidx < length(ps)
        Ds = Vector{Vector{T}}([Vector{T}() for n in 1:nthreads])
        Ds[1] = D
        Threads.@threads for Np in Nps[pidx]+1:Nps[pidx+1]
            r = rand(pdf)
            (r > qs[pidx]) && push!(Ds[Threads.threadid()], r)
        end

        D = sort(vcat(Ds...))

        N0 = Nps[pidx+1] - length(D)
        # ival = (1 - ps[pidx+1]) * Nps[pidx+1]
        # il = floor(Int, ival)
        # iu = il + 1
        # qs[pidx+1] = (D[il-N0]*(iu-ival)+D[iu-N0]*(ival-il))/(iu-il)
        h = (Nps[pidx+1]-1/3)*ps[pidx+1] + 1/3 - N0 # Hyndman Hyp. 8
        qs[pidx+1] = D[floor(Int,h)] + (h-floor(h))*(D[ceil(Int,h)]-D[floor(Int,h)])

        D = D[findfirst(x -> x > qs[pidx+1], D):end]
        Np_used[pidx+1] = length(D)

        pidx += 1
    end

    return ps, qs #, Np_used
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
const model_param_file = "/home/jules/algo_PACOME/src/" *
                         "abacus_conflevel_distrib_model_param.fits"

model_param = read(FitsArray, model_param_file)

function model_conflevel(dof::Int, ρ::T) where {T<:AbstractFloat}
    @assert 1 ≤ dof ≤ 100
    a, n0, Un0, r = model_param[:,dof]
    return a^(-log10(ρ)-n0)*(Un0-r)+r
end

end # module
