#
# orbits.jl --
#
# Julia methods for dealing with simple Keplerian orbits.
#
module Orbits

using OptimPackNextGen: fzero
using Parameters
using Measurements

"""
    twopi(T = Float64) -> 2π :: T

yields 2π converted to floating-point type `T`.

"""
twopi(::Type{T} = Float64) where {T<:AbstractFloat} = 2*T(π)

"""
    Orbit(a, e, i, τ, ω, Ω, K, P)

yields a structure storing orbital parameters `a` the *semi-major axis*, `e` the
*eccentricity*, `i` the *inclination of orbit* (in degrees), `τ` the *epoch of
periapsis passage* expressed as a fraction of the orbital period, `ω` the
*argument of periapsis* (in degrees), `Ω` the *longitude of ascending node* (in
degrees), `K` the kepler's constant a^3/P^2 (in [a]^3.yr^-2) and `P` the
*orbital period* (in years).

Orbital parameters can be specified by keywords:

    Orbit(T=Float64; a=…, e=…, i=…, τ=…, ω=…, Ω=…, K=…)

where all keywords must be provided, optional parameter `T` is the
floating-point type for storing the orbital parameters.  To modify some
orbital parameters, call the constructor as:

    Orbit(A; a=semimajor_axis(A), e=eccentricity(A), i=inclination(A),
          τ=periapsis_epoch(A), ω=periapsis_argument(A),
          Ω=ascending_node_longitude(A), K=kepler_constant(A))

with given orbital parameters `A` specified as positional argument and keywords
for the parameters to modify.

The floating-point type `T` for storing the parameters can be specified as a
type parameter:

    Orbit{T}(args...; kwds...)

The `convert` method and the constructor can be used to change the
floating-point type.  For instance, assuming `A` is an instance of `Orbit`,
then:

    Orbit{Float32}(A)
    convert(Orbit{Float32}, A)

yields single precision orbital parameters.

"""
@with_kw struct Orbit{T<:AbstractFloat}
   a::T  # semi-major axis
   e::T  # eccentricity
   i::T  # inclination of orbit (deg)
   τ::T  # fraction of epoch of periapsis
   ω::T  # longitude of periapsis (deg)
   Ω::T  # position angle of ascending node (deg)
   K::T  # kepler's constant ([a]^3/yr^2)
   P::T=sqrt(a^3/K) # period (yr)
end
semimajor_axis(A::Orbit)           = getfield(A, :a)
eccentricity(A::Orbit)             = getfield(A, :e)
inclination(A::Orbit)              = getfield(A, :i)
periapsis_epoch(A::Orbit)          = getfield(A, :τ)
periapsis_argument(A::Orbit)       = getfield(A, :ω)
ascending_node_longitude(A::Orbit) = getfield(A, :Ω)
kepler_constant(A::Orbit)          = getfield(A, :K)
period(A::Orbit)                   = getfield(A, :P)

# Constructors.
# Orbit(::Type{T}=Float64; kwds...) where {T<:AbstractFloat} = Orbit{T}(; kwds...)
Orbit(A::Orbit{T}; kwds...) where {T} = Orbit{T}(A; kwds...)
function Orbit{T}(a::Real, e::Real, i::Real, τ::Real,
                  ω::Real, Ω::Real, K::Real) where {T<:AbstractFloat}
    return Orbit{T}(a, e, i, τ, ω, Ω, K)
end
function Orbit{T}(A::Orbit;
                  a::Real = semimajor_axis(A),
                  e::Real = eccentricity(A),
                  i::Real = inclination(A),
                  τ::Real = periapsis_epoch(A),
                  ω::Real = periapsis_argument(A),
                  Ω::Real = ascending_node_longitude(A),
                  K::Real = kepler_constant(A)) where {T<:AbstractFloat}
    return Orbit{T}(a, e, i, τ, ω, Ω, K)
end

# Conversion.
Base.convert(::Type{T}, A::T) where {T<:Orbit} = A
Base.convert(::Type{Orbit{T}}, A::Orbit) where {T} =
    Orbit{T}(semimajor_axis(A), eccentricity(A), inclination(A),
             periapsis_epoch(A), periapsis_argument(A),
             ascending_node_longitude(A), kepler_constant(A), period(A))

# Base.show(io::IO, A::Orbit{T}) where {T} = begin
#     x = (("a", string(semimajor_axis(A)), "semi-major axis"),
#          ("e", string(eccentricity(A)), "eccentricity"),
#          ("i", string(inclination(A)), "inclination (deg)"),
#          ("τ", string(periapsis_epoch(A)), "epoch of periapsis fraction"),
#          ("ω", string(periapsis_argument(A)), "argument of periapsis (deg)"),
#          ("Ω", string(ascending_node_longitude(A)),
#           "longitude of ascending node (deg)"),
#          ("K", string(kepler_constant(A)), "kepler's constant ([a]^3/yr^2)"),
#          ("P", string(period(P)), "period (yr)"))
#     hdr = "Orbit{$T}("
#     spc = repeat(" ", length(hdr))
#     maxlen = 0
#     for (lab, val, com) in x
#         maxlen = max(maxlen, length(lab) + length(val))
#     end
#     n = length(x)
#     for i in 1:n
#         lab, val, com = x[i]
#         cnt = 1 + maxlen - length(lab) - length(val)
#         print(io, (i == 1 ? hdr : spc), lab, " =", repeat(" ", cnt), val)
#         if i < n
#             println(io, ", # ", com)
#         else
#             print(io, ") # ", com)
#         end
#     end
# end

"""
    Grid(a, e, i, τ, ω, Ω, K)

gives a structure of arrays `a`, `e`, `i`, `τ`, `ω`, `Ω`, `K` containing the
values over which each orbital elements is sampled. It encodes the search grid
The total number of orbits covered by the grid is `norbits`.

Grid parameters can be specified by keywords:

    Grid(; a=…, e=…, i=…, τ=…, ω=…, Ω=…, K=…) -> grid

where all keywords must be provided except for `norbits` which is computed
automatically given the seven arrays. The result, `grid`, is a grid structure.

A `grid` structure can also be initialized from a .txt file with :

    Grid{T}(file_path) -> grid

where `file_path` is a `String` encoding the path where the .txt file is stored.
The .txt file must be formatted following the example given in the documentation.

"""
struct Grid{T<:AbstractFloat}
   a::AbstractArray{T,1}  # semi-major axis (arbitrary)
   e::AbstractArray{T,1}  # eccentricity
   i::AbstractArray{T,1}  # inclination of orbit (deg)
   τ::AbstractArray{T,1} # epoch of periapsis fraction
   ω::AbstractArray{T,1}  # longitude of periapsis (deg)
   Ω::AbstractArray{T,1}  # position angle of ascending node (deg)
   K::AbstractArray{T,1}  # period ([a]^3/yr^2)
   norbits::Int           # number of orbits to cover
end

function Grid{T}(; a::AbstractArray{T,1},
                   e::AbstractArray{T,1}, i::AbstractArray{T,1},
                   τ::AbstractArray{T,1}, ω::AbstractArray{T,1},
                   Ω::AbstractArray{T,1},
                   K::AbstractArray{T,1}) where {T<:AbstractFloat}
    return Grid{T}(a, e, i, τ, ω, Ω, K, length(a)*length(e)*length(i)*
                   length(τ)*length(ω)*length(Ω)*length(K))
end

function Grid{T}(file_path::String) where {T<:AbstractFloat}

    isfile(file_path) || error("File does not exist !")

    s = Array{T,2}(undef, (7,3))

    lines = readlines(file_path)
    for (k, line) in enumerate(lines[32:38])

        start, stop, len = split(line,"\t")[3:end]
        s[k,:] .= parse(T,start), parse(T,stop), parse(Int,len)
    end

    grid = Grid{T}(a  = LinRange{T}(s[1,1],s[1,2],s[1,3]),
                   e  = LinRange{T}(s[2,1],s[2,2],s[2,3]),
                   i  = LinRange{T}(s[3,1],s[3,2],s[3,3]),
                   τ  = LinRange{T}(s[4,1],s[4,2],s[4,3]),
                   ω  = LinRange{T}(s[5,1],s[5,2],s[5,3]),
                   Ω  = LinRange{T}(s[6,1],s[6,2],s[6,3]),
                   K  = LinRange{T}(s[7,1],s[7,2],s[7,3]))
    return grid
end

Grid(file_path::String) = Grid{Float32}(file_path)

"""
    KeplerEquation(e, M) -> eq

yields and instance of Kepler's equation for eccentricity `e` and mean anomaly
`M`.

Kepler's equation relates the *eccentric anomaly* `E` (in radians), the
eccentricity `e` and the *mean anomaly* `M` (both unitless) by:

    E - e*sin(E) = M

where at time `t`:

    M = 2π*(t/P - τ) mod 2π

with `P` the orbital period and `τ` the epoch of periapsis fraction.

Object `eq` is callable:

    eq(E) -> E - M - e*sin(E)

Call `eccentric_anomaly(eq)` to solve Kepler's equation `eq`.

""" KeplerEquation

struct KeplerEquation{T<:AbstractFloat}
    e::T # eccentricity
    M::T # mean anomaly
end
eccentricity(eq::KeplerEquation) = getfield(eq, :e)
mean_anomaly(eq::KeplerEquation) = getfield(eq, :M)

# Make an instance of Kepler's equation callable so that it can be used to
# solve the equation.
(eq::KeplerEquation)(E::Real) =
    E - mean_anomaly(eq) - eccentricity(eq)*sin(E)

KeplerEquation(::Type{T}=Float64; kwds...) where {T} =
    KeplerEquation{T}(; kwds...)
KeplerEquation(eq::KeplerEquation{T}; kwds...) where {T} =
    KeplerEquation{T}(eq; kwds...)
KeplerEquation{T}(; e::Real, M::Real) where {T<:AbstractFloat} =
    KeplerEquation{T}(e, M)
function KeplerEquation{T}(eq::KeplerEquation;
                           e::Real = eccentricity(eq),
                           M::Real = mean_anomaly(eq)) where {T<:AbstractFloat}
    return KeplerEquation{T}(e, M)
end

Base.convert(::Type{T}, eq::T) where {T<:KeplerEquation} = eq
Base.convert(::Type{KeplerEquation{T}}, eq::KeplerEquation) where {T} =
    KeplerEquation{T}(eccentricity(eq), mean_anomaly(eq))

"""
    eccentric_anomaly(eq; kwds...) -> E

yields the solution of Kepler's equation `eq` (see [`KeplerEquation`](@ref)).
Keywords `kwds...` are those accepted by `OptimPackNextGen.fzero`.

The parameters of Kepler's equation, the *eccentricity* `e` and the *mean
anomaly* `M`, can be directly specified:

    eccentric_anomaly(e, M; kwds...) -> E

Another possibility is to provide orbital parameters `A` and epoch `t` (in
years):

    eccentric_anomaly(A, t; kwds...) -> E

"""
eccentric_anomaly(eq::KeplerEquation; kwds...) =
    #eccentric_anomaly_cordic(eq; kwds...)
    eccentric_anomaly_brent(eq; kwds...)

eccentric_anomaly(A::Orbit, t::Real; kwds...) =
    eccentric_anomaly(eccentricity(A), mean_anomaly(A, t); kwds...)

eccentric_anomaly(e::Real, M::Real; kwds...) = begin
    T = float(promote_type(typeof(e), typeof(M)))
    eccentric_anomaly(KeplerEquation{T}(e, M); kwds...)
end

eccentric_anomaly(e::T, M::T; kwds...) where {T<:AbstractFloat} =
    eccentric_anomaly(KeplerEquation(e, M); kwds...)

function eccentric_anomaly_brent(f::KeplerEquation{T};
                                 kwds...) where {T}
    # A crude bracketing of the solution of Kepler's equation is given by:
    # E ∈ [M - e, M + e].  For elliptic orbits, the eccentricity is such
    # that 0 ≤ e < 1 and f(E) = E - M - e*sin(E) is a strictly
    # non-decreasing function as f'(E) = 1 - e*cos(E) ≥ 1 - e > 0.  A
    # narrower interval can be determined in that case which makes the
    # search a bit faster.
    e = eccentricity(f)
    M = mean_anomaly(f)
    0 ≤ e < 1 || error("not an elliptic orbit")
    a = M # initial guess
    fa = f(a)
    fa == 0 && return a
    #(fa == 0 || e < 1e-5) && return a

    b = (fa > 0 ? a - e : a + e)
    fb = f(b)

    E, = fzero(T, f, a, fa, b, fb; kwds...)
    return E
end

function eccentric_anomaly_cordic(eq::KeplerEquation{T};
                                  kwds...) where {T}
    E, = cordic_ecs(Float64(eccentricity(eq)),
                    Float64(mean_anomaly(eq)); kwds...)
    return T(E)
end

"""
    cordic_ecs(e, M; n=) -> E, cosE, sinE

solves Kepler's equation `E - e*sin(E) = M` by `n` iterations of Zechmeister's
CORDIC-like method and yields the solution `E` (in radians), its cosine and its
sine.

This method is faster than Brent's method applied to Kepler's equation.

Reference: M. Zechmeister, "CORDIC-like method for solving Kepler’s equation",
A&A, vol. 619, A218 (2018).

"""
function cordic_ecs(e::T, M::T; n::Int = 55) where {T<:Float64}
    # NOTE: The recurrence to compute the sine and cosine of the eccentric
    # anomaly `E` introduces rounding errors and the precision is ~ √n⋅ϵ with
    # `n` the number of iterations.  As suggested in the paper, we could switch
    # to Newton-Raphson or Halley methods for polishing the solution.
    #
    # NOTE: Another refinement could be to keep the best solution so far.
    tbl = cordic_table(n)
    E = twopi(T)*round(M/twopi(T))
    expiE = Complex{T}(1, 0)
    for k in 1:n
        a, expia = tbl[k]
        f = E - M - e*expiE.im
        if f > 0
            E -= a
            expiE *= conj(expia)
        elseif f < 0
            E += a
            expiE *= expia
        else
            break
        end
    end
    return E, expiE.re, expiE.im
end

const _CORDIC_TABLE = Tuple{Float64,Complex{Float64}}[]

function cordic_table(n::Int)
    # NOTE: The last entries (for k ≥ 29) of the table are (a, 1 +1im*a) and
    # don't need to be stored or, at least, computed.
    if length(_CORDIC_TABLE) < n
        m = length(_CORDIC_TABLE)
        resize!(_CORDIC_TABLE, n)
        a = Float64(π)
        @inbounds for k in m+1:n
            a = Float64(π)/2^k
            _CORDIC_TABLE[k] = (a, expi(a))
        end
    end
    return _CORDIC_TABLE
end

"""
    mean_anomaly(A, t) -> M

yields the mean anomaly `M` for orbit `A` at epoch `t`.  The returned value is
in the range `[0,2π)`.  Another possibility is to compute the mean anomaly from
Kepler's equation `eq` (see [`KeplerEquation`](@ref)):

    mean_anomaly(eq) -> M

"""
mean_anomaly(A::Orbit{T}, t::T) where {T} = begin
    # NOTE: mod2pi(x) is faster than mod(x,q) whatever q.
   # println("P=$(sqrt(semimajor_axis(A)^3/kepler_constant(A)))")
   mod2pi(twopi(T)*(t/period(A)-mod(periapsis_epoch(A),1)))
end

mean_anomaly(A::Orbit{T1}, t::T2) where {T1,T2<:Real} = begin
    T = promote_type(T1, T2)
    # println("P=$(sqrt(semimajor_axis(A)^3/kepler_constant(A)))")
    mod2pi(twopi(T)*(T(t)/T(period(A))-T(mod(periapsis_epoch(A),1))))
end

"""
    projected_position(A, t; polar=false, kwds...) -> x, y

yields the projected position for orbital parameters `A` at epoch `t` (in
years) as the relative tangential right ascension (`x = ΔRA`) and declination
(`y = ΔDec`) in same units as the semi-major axis stored in `A`.

Specify keyword `polar=true` to get the projected position in polar
coordinates:

    projected_position(A, t; polar=true, kwds...) -> ρ, θ

with `ρ` is the projected separation (in same units as the semi-major axis
stored in `A`) and `θ` is the position angle (in degrees) in the reference
frame.

Keyword `polar` must be specified (to avoid ambiguities).  Other keywords
`kwds...` are used to determine the eccentric anomaly.

"""
function projected_position(A::Orbit{T}, t::T;
                            polar::Bool=false, kwds...) where {T<:AbstractFloat}

    # Extract orbital parameters (angles in radians).
    a = semimajor_axis(A)
    e = eccentricity(A)
    i = deg2rad(inclination(A))
    ω = deg2rad(periapsis_argument(A))
    Ω = deg2rad(ascending_node_longitude(A))

    # Determine eccentric anomaly `E`.
    E = eccentric_anomaly(A, t; kwds...)
    M = mean_anomaly(A, t; kwds...)

    # True separation `r`.
    r = a*(1 - e*cos(E))

    # True anomaly `ν` (in radians).
    ν = 2*atan(sqrt((1 + e)/(1 - e))*tan(E/2))

    # ∂X_∂r = sin(Ω)*cos(ν+ω) + cos(Ω)*sin(ν+ω)*cos(i)
    # ∂Y_∂r = cos(Ω)*cos(ν+ω) - sin(Ω)*sin(ν+ω)*cos(i)
    # ∂r_∂a = 1 - e*cos(E)
    # ∂XY_∂a = [∂X_∂r*∂r_∂a, ∂Y_∂r*∂r_∂a]

    # Position angle `θ` (in radians) and projected separation `ρ`.
    sinνpω, cosνpω = sincos(ν + ω)
    cosi = cos(i)

    # Return projected position.
    if polar
        # Position angle θ and projected speration ρ.
        θ = Ω + atan(cosi*sinνpω, cosνpω)
        ρ = r*hypot(cosνpω, cosi*sinνpω)
        return ρ, rad2deg(θ)
    else
        # Compute projected separation ρ times the cosine and sine of the
        # position angle θ.  Then return sky plane coordinates (x,y) such that
        # x = +ΔRA and y = +ΔDec.
        sinΩ, cosΩ = sincos(Ω)
        ρsinθ = r*(sinΩ*cosνpω + cosΩ*(sinνpω*cosi))
        ρcosθ = r*(cosΩ*cosνpω - sinΩ*(sinνpω*cosi))
        return ρsinθ, ρcosθ
    end
end

projected_position(A::Orbit{T1}, t::T2; kwds...) where {T1,T2<:Real} = begin
    T = promote_type(T1, T2)
    return projected_position(convert(Orbit{T}, A), T(t); kwds...)
end

function projected_position3D(A::Orbit{T}, t::T;
                              polar::Bool=false, kwds...) where {T<:AbstractFloat}

    # Extract orbital parameters (angles in radians).
    a = semimajor_axis(A)
    e = eccentricity(A)
    i = deg2rad(inclination(A))
    ω = deg2rad(periapsis_argument(A))
    Ω = deg2rad(ascending_node_longitude(A))

    # Determine eccentric anomaly `E`.
    E = eccentric_anomaly(A, t; kwds...)

    # True separation `r`.
    r = a*(1 - e*cos(E))

    # True anomaly `ν` (in radians).
    ν = 2*atan(sqrt((1 + e)/(1 - e))*tan(E/2))

    # Position angle `θ` (in radians) and projected separation `ρ`.
    sinνpω, cosνpω = sincos(ν + ω)
    cosi = cos(i)

    # Return projected position.
    if polar
        # Position angle θ and projected speration ρ.
        θ = Ω + atan(cosi*sinνpω, cosνpω)
        ρ = r*hypot(cosνpω, cosi*sinνpω)
        return ρ, rad2deg(θ)
    else
        # Compute projected separation ρ times the cosine and sine of the
        # position angle θ.  Then return sky plane coordinates (x,y) such that
        # x = +ΔRA and y = +ΔDec.
        sinΩ, cosΩ = sincos(Ω)
        ρsinθ = r*(sinΩ*cosνpω + cosΩ*(sinνpω*cosi))
        ρcosθ = r*(cosΩ*cosνpω - sinΩ*(sinνpω*cosi))
        return ρsinθ, ρcosθ, r*sinνpω*sin(i)
    end
end

projected_position3D(A::Orbit{T1}, t::T2; kwds...) where {T1,T2<:Real} = begin
    T = promote_type(T1, T2)
    return projected_position3D(convert(Orbit{T}, A), T(t); kwds...)
end

"""
projected_position_derivs(A, t; kwds...) -> X, Y, partialderivs

yields the projected position for orbital parameters `A` at epoch `t` (in
years) as the relative tangential right ascension (`X = ΔRA`) and declination
(`Y = ΔDec`) in same units as the semi-major axis stored in `A`. It also
updates the 7-element array of tuples `partialderivs` with the partial
derivatives of the positions with respect to the 7 orbital elements.

The output `Jac_XY` is a 2x7 array that encodes the Jacobian matrix of the
positions XY with respect to the orbital parameters μ. It takes the form :
--> [[∂X_∂a, ∂X_∂e, ∂X_∂i, ∂X_∂t0, ∂X_∂ω, ∂X_∂Ω, ∂X_∂P],
     [∂Y_∂a, ∂Y_∂e, ∂Y_∂i, ∂Y_∂t0, ∂Y_∂ω, ∂Y_∂Ω, ∂Y_∂P]]

"""

function projected_position_derivs(A::Orbit{T}, t::T,
                                   kwds...) where {T<:AbstractFloat}

    # Extract orbital parameters (angles in radians).
    a = semimajor_axis(A)
    e = eccentricity(A)
    i = deg2rad(inclination(A))
    τ = periapsis_epoch(A)
    ω = deg2rad(periapsis_argument(A))
    Ω = deg2rad(ascending_node_longitude(A))
    P = period(A)
    K = kepler_constant(A)

    # Determine eccentric anomaly `E`.
    E = eccentric_anomaly(A, t; kwds...)
    M = mean_anomaly(A, t; kwds...)

    sini, cosi = sincos(i)
    sinΩ, cosΩ = sincos(Ω)
    sinE, cosE = sincos(E)

    # True separation `r`.
    r = a*(1 - e*cosE)

    # True anomaly `ν` (in radians).
    ν = 2*atan(sqrt((1 + e)/(1 - e))*tan(E/2))

    # Return projected position.
    # Compute projected separation ρ times the cosine and sine of the
    # position angle θ.  Then return sky plane coordinates (x,y) such that
    # x = +ΔRA and y = +ΔDec.
    sinνpω, cosνpω = sincos(ν + ω)

    X = r*(sinΩ*cosνpω + cosΩ*(sinνpω*cosi)) # CHECK
    Y = r*(cosΩ*cosνpω - sinΩ*(sinνpω*cosi)) # CHECK

    # Compute the derivatives of the positions with respect to
    # the orbital elements.
    ∂r_∂a = 1 - e*cosE -3*π*t*e*sinE/(1-e*cosE) * sqrt(K/a^3) # CHECK
    ∂r_∂e = a*(e-cosE)/(1-e*cosE) # CHECK
    ∂r_∂K = e*sinE*π*t/(sqrt(a*K)*(1-e*cosE)) # CHECK
    ∂r_∂τ = -twopi(T)*a*e*sinE/(1-e*cosE)# CHECK

    ∂ν_∂e = sinE/(1-e*cosE) * (1/sqrt(1-e^2) + sqrt(1-e^2)/(1-e*cosE)) # CHECK
    ∂ν_∂K = sqrt(1-e^2)*π*t/((1-e*cosE)^2*sqrt(K*a^3)) # CHECK
    ∂ν_∂a = -3*π*t*sqrt(K*(1-e^2)/a^5)/(1-e*cosE)^2 # CHECK
    ∂ν_∂τ = -twopi(T)*sqrt(1-e^2)/(1-e*cosE)^2 # CHECK

    ∂X_∂r = sinΩ*cosνpω + cosΩ*(sinνpω*cosi) # CHECK
    ∂X_∂ν = -r*(sinΩ*sinνpω - cosΩ*cosνpω*cosi) # CHECK
    ∂Y_∂r = cosΩ*cosνpω - sinΩ*(sinνpω*cosi) # CHECK
    ∂Y_∂ν = -r*(cosΩ*sinνpω + sinΩ*cosνpω*cosi) # CHECK

    Jac_XY = Array{T,2}(undef, (2,7))
    Jac_XY[1,1] = ∂X_∂r*∂r_∂a+∂X_∂ν*∂ν_∂a
    Jac_XY[1,2] = ∂X_∂r*∂r_∂e+∂X_∂ν*∂ν_∂e
    Jac_XY[1,3] = -r*cosΩ*sinνpω*sini
    Jac_XY[1,4] = ∂X_∂r*∂r_∂τ+∂X_∂ν*∂ν_∂τ
    Jac_XY[1,5] = ∂X_∂ν
    Jac_XY[1,6] = r*∂Y_∂r
    Jac_XY[1,7] = ∂X_∂r*∂r_∂K+∂X_∂ν*∂ν_∂K
    Jac_XY[2,1] = ∂Y_∂r*∂r_∂a+∂Y_∂ν*∂ν_∂a
    Jac_XY[2,2] = ∂Y_∂r*∂r_∂e+∂Y_∂ν*∂ν_∂e
    Jac_XY[2,3] = r*sinΩ*sinνpω*sini
    Jac_XY[2,4] = ∂Y_∂r*∂r_∂τ+∂Y_∂ν*∂ν_∂τ
    Jac_XY[2,5] = ∂Y_∂ν
    Jac_XY[2,6] = -r*∂X_∂r
    Jac_XY[2,7] = ∂Y_∂r*∂r_∂K+∂Y_∂ν*∂ν_∂K

    return [X,Y], Jac_XY
end

"""
    KeplerStellarMass(orb, par) -> Mstar

yields an approximation of the mass of a star `Mstar` orbited by a companion
of orbital elements `orb` and with parallax `par` (in mas). `orb` can either be
of type `Orbit` or `Array`

"""
function KeplerStellarMass(orb::Orbit{T}, par::T) where {T<:AbstractFloat}
    G = T(39.47524008717717);   # Gravitational constant [au3 Msun-1 yr-2]
    return(orb.K/G*twopi(T)^2*(1/par)^3)
end

function KeplerStellarMass(K::T, par::T) where {T<:AbstractFloat}
    G = T(39.47524008717717);   # Gravitational constant [au3 Msun-1 yr-2]
    return(K/G*twopi(T)^2*(1/par)^3)
end

function KeplerStellarMass(orb::Array{T,1}, par::T) where {T<:AbstractFloat}
    @assert length(orb)==7
    G = T(39.47524008717717);   # Gravitational constant [au3 Msun-1 yr-2]
    return(orb[end]/G*twopi(T)^2*(1/par)^3)
end


"""
    KeplerConstant(Mstar, par) -> K

yields an approximation of Kepler's constant `K` given the stellar mass `Mstar`
and the parallax `par`.

"""
function KeplerConstant(Mstar::T, par::T) where {T<:AbstractFloat}
    G = T(39.47524008717717);   # Gravitational constant [au3 Msun-1 yr-2]
    return(Mstar*G/(twopi(T)^2)*par^3)
end

function KeplerConstant(Mstar::Measurement{T},
                        par::Measurement{T}) where {T<:AbstractFloat}
    G = T(39.47524008717717);   # Gravitational constant [au3 Msun-1 yr-2]
    return(Mstar*G/(twopi(T)^2)*par^3)
end

"""
    KeplerPeriod(orb) -> P

yields the period (in years) of the object whose orbital elements are encoded in
the array `orb`.

"""
function KeplerPeriod(orb::Array{T,1}) where {T<:AbstractFloat}
    @assert length(orb)==7
    return sqrt(orb[1]^3/orb[end])
end


"""
    expi(θ) -> exp(1i*θ)

yields the complex `exp(1i*θ)` for `θ` real in radians.  This is a cheap way to
compute and store the sine and cosine of `θ`.

"""
expi(θ::Real) = begin
    sinθ, cosθ = sincos(θ)
    return Complex(cosθ, sinθ)
end


"""
    orb_comb(a, e, i, τ, ω, Ω, K) -> a_k, e_k, i_k, τ_k, ω_k, Ω_k, K_k

creates a unique orbital combination given the lists of orbital
parameters `a`, `e`, `i`, `τ`, `ω`, `Ω`, `K` and the index `k`.

Note : index `k` must be between 1 and the total number of orbits to explore.
Note2 : Execution time = 78ns so for 1E8 orbits to explore, it only adds ~8s
        to the calculation.
"""
function orb_comb(k::Int,
                  a::AbstractArray{T,1},
                  e::AbstractArray{T,1},
                  i::AbstractArray{T,1},
                  τ::AbstractArray{T,1},
                  ω::AbstractArray{T,1},
                  Ω::AbstractArray{T,1},
                  K::AbstractArray{T,1}) where {T<:AbstractFloat}

    la = getfield(a,:len)
    le = getfield(e,:len)
    li = getfield(i,:len)
    lτ = getfield(τ,:len)
    lω = getfield(ω,:len)
    lΩ = getfield(Ω,:len)
    lK = getfield(K,:len)

    k -= 1
    K_k = K[(k%lK)+1]
    Ω_k = Ω[(k÷(lK))%lΩ+1]
    ω_k = ω[(k÷(lK*lΩ))%lω+1]
    τ_k = τ[(k÷(lK*lΩ*lω))%lτ+1]
    i_k = i[(k÷(lK*lΩ*lω*lτ))%li+1]
    e_k = e[(k÷(lK*lΩ*lω*lτ*li))%le+1]
    a_k = a[(k÷(lK*lΩ*lω*lτ*li*le))%la+1]

    return a_k, e_k, i_k, τ_k, ω_k, Ω_k, K_k
end

function orb_comb(k::Int,
                  grid::Grid{T}) where {T<:AbstractFloat}

    a = getfield(grid,:a)
    e = getfield(grid,:e)
    i = getfield(grid,:i)
    τ = getfield(grid,:τ)
    ω = getfield(grid,:ω)
    Ω = getfield(grid,:Ω)
    K = getfield(grid,:K)
    return orb_comb(k, a, e, i, τ, ω, Ω, K)
end

"""
    arr2orb(arr) -> orb

converts an array `arr` containing the 7 orbital elements ordered as
`arr` = [a, e, i, τ, ω, Ω, K] into a Orbit structure `orb`.

"""
function arr2orb(arr::AbstractArray{T,1}) where {T<:AbstractFloat}
    return Orbit{T}(a  = arr[1], e  = arr[2], i  = arr[3], τ = arr[4],
                    ω  = arr[5], Ω  = arr[6], K  = arr[7])
end

"""
    orb2arr(orb) -> arr

converts an Orbit structure `orb` into an array `arr` containing the 7 orbital
elements ordered as `arr` = [a, e, i, τ, ω, Ω, K].

"""
function orb2arr(orb::Orbit{T}) where {T<:AbstractFloat}
    return [orb.a, orb.e, orb.i, orb.τ, orb.ω, orb.Ω, orb.K]
end

"""
    wrap_orb(orb) -> orb

wraps the orbital elements of the Orbit structure `orb`.

"""
function wrap_orb(orb::Orbit{T}) where {T<:AbstractFloat}
    orb_arr = orb2arr(orb)
    wrap_orb!(orb_arr)
    return arr2orb(orb_arr)
end

function wrap_orb!(orb::Vector{T}) where {T<:AbstractFloat}
    i = mod(abs(orb[3]),360)
    i > 180. ? (orb[3] = 360-i) : (orb[3] = i)
    orb[4] = mod(orb[4],1) # τ
    orb[5] = mod(orb[5],360) # ω
    orb[6] = mod(orb[6],360) # Ω
end

function wrap_orb!(orbs::AbstractArray{T,2}) where {T<:AbstractFloat}
    @assert size(orbs,1) == 7
    no = size(orbs,2)
    for o in 1:no
        i = mod(abs(orbs[3,o]),360)
        i > 180. ? (orbs[3,o] = 360-i) : (orbs[3,o] = i)
        orbs[3,o] = mod(orbs[3,o],180) # i
        orbs[4,o] = mod(orbs[4,o],1) # τ
        orbs[5,o] = mod(orbs[5,o],360) # ω
        orbs[6,o] = mod(orbs[6,o],360) # Ω
    end
end

end # module
