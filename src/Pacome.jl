module Pacome

using EasyFITS
using InterpolationKernels
using TwoDimensional
using Glob
using Printf
using LinearAlgebra
using Parameters
using Mmap
using Dates
using FiniteDiff
using Revise, OptimPackNextGen
using Setfield
using Distributions
using Base: @propagate_inbounds
using ProgressMeter
using InterpolationKernels: compute_offset_and_weights
using SpecialFunctions: erf
using FiniteDifferences: grad, jacobian, forward_fdm, backward_fdm, central_fdm
using Orbits:  Orbit, projected_position, projected_position_derivs,
               projected_position3D,
               orb_comb, arr2orb, orb2arr, period, wrap_orb, wrap_orb!,
               Grid, eccentric_anomaly, twopi
using Utils: guess_paco_hduname, fix_paco_image!, central_pixel,
             rewrite_paco_file, mjd_to_epoch, formatTimeSeconds, SPHERE_COR_DIAM,
             split_on_threads

struct PacomeData{T<:AbstractFloat,N}
    dates::Vector{String}     # Date of observation (in ISO 8601 standard)
    epochs::Vector{T}         # Date of observation (in Julian years)
    pixres::Vector{T}         # Pixel resolution at each epoch
    centers::Vector{Point{T}} # Image center in index units at each epoch
    dims::NTuple{N,Int}       # Size of images
    a::Vector{Array{T,N}}     # Denominator map at each epoch
    b::Vector{Array{T,N}}     # Numerator map at each epoch
    iflts::Vector{String}     # Infra-Red filter combination name
    icors::Vector{String}     # Infra-Red coronagraphic combination name
end
Base.length(A::PacomeData) = length(A.epochs)

"""
    PacomeData{T,N}(pat, outliers=nothing; mode, asdi_apriori, asdi_masked, asdi_BB) -> dat

loads all the numerator and denominator maps produced by PACO reduction and
stored in FITS files found in directories matching glob-style pattern `pat`.
Optional argument `outliers` is to specify a list of directories to omit.

Keyword argument `mode` specified whether the data was reduced in `"adi"` or in
`"asdi"` mode. The ASDI PACO prior index is given in `asdi_apriori`. `asdi_masked`
is a boolean specifying if the masked ASDI data should be considered (see PACO).
Finally, `asdi_BB` is to include (or not) broad band ADI observations along with
the other ASDI observations.

Parameter `T` is the floating-point type of the data; if omitted,
`T=Float32` is assumed.

Parameter `N` is the number of dimensions of the PACO maps; if omitted,
`N=3` is assumed.

"""
PacomeData(pat::AbstractString, args...; kwds...) =
    PacomeData{Float32}(pat, args...; kwds...)

PacomeData{T}(pat::AbstractString, args...; kwds...) where {T<:AbstractFloat} =
    PacomeData{T,3}(pat, args...; kwds...)

function PacomeData{T,N}(pat::AbstractString,
                         outliers = String[];
                         mode::String = "adi",
                         asdi_apriori::Int = 1,
                         asdi_masked::Bool = false,
                         asdi_BB::Bool = false) where {T<:AbstractFloat,N}

    @assert N ≥ 2

    print("Loading PACO data ")

    # Build sorted list of directories.
    listdir = String[]
    for path in (startswith(pat, "/") ? glob(pat[2:end], "/") : glob(pat))
        if isdir(path) && !(basename(path) ∈ outliers)
            push!(listdir, path)
        end
    end

    # Assigns the right PACO path depending on the mode (ADI or ASDI) and the
    # right numerator and denominator patterns
    mode = lowercase(mode)
    if mode == "adi"
        paco_path = "1_reduction_paco/adi_results/all_fits"
        num_pat = "*_num_aligned_*"
        den_pat = "*_denom_aligned_*"
    elseif mode == "asdi"
        paco_path = "1_reduction_paco/asdi_results/all_fits"
        if asdi_masked
            num_pat = "*_wwm_num_aligned_wwms$(asdi_apriori)_*"
            den_pat = "*_wwm_denom_aligned_wwms$(asdi_apriori)_*"
        else
            num_pat = "*_ww_num_aligned_wwms$(asdi_apriori)_*"
            den_pat = "*_ww_denom_aligned_wwms$(asdi_apriori)_*"
        end
    else
        error("Unsupported PACO mode! Should be `adi` or `asdi`.")
    end

    # Collect data for all epochs.
    epochs = T[]            # epochs of observation
    pixres = T[]            # pixel resolution for each epoch
    sizes = NTuple{N,Int}[] # size of the images at every epoch
    B = Array{T,N}[]        # numerators of each epoch
    A = Array{T,N}[]        # denominators of each epoch
    centers = Point{T}[]    # image centers
    filters = String[]
    dates_obs = String[]
    coronagraphs = String[]
    for dir in listdir
        isdir(joinpath(dir, paco_path)) ? nothing : continue
        print(".")
        file_num = glob(num_pat, joinpath(dir, paco_path))
        file_den = glob(den_pat, joinpath(dir, paco_path))
        length(file_num) == 1 ? file_num = first(file_num) :
                                          error("several numerators were found in dir : $(dir)")
        length(file_den) == 1 ? file_den = first(file_den) :
                                        error("several denominators were found in dir : $(dir)")
        b = read(FitsImage{T}, file_num)
        a = read(FitsImage{T}, file_den)
        size(a) == size(b) ||
            error("numerator and denominator maps have different sizes")
        if mode == "asdi"
            arr_a, arr_b = get(Array, a), get(Array, b)
            a = FitsImage(reshape(arr_a, (size(a)...,1)), get(FitsHeader, a))
            b = FitsImage(reshape(arr_b, (size(b)...,1)), get(FitsHeader, b))
        end
        length(size(a)) == N || error("expecting $N-dimensional maps")
        haskey(a, "HDUNAME") || fix_paco_image!(a, guess_paco_hduname(file_den))
        haskey(b, "HDUNAME") || fix_paco_image!(b, guess_paco_hduname(file_num))
        if ((a["HDUNAME"]::String,
             b["HDUNAME"]::String) ∈ (("PACO_DENOM_ALIGNED",
                                       "PACO_NUM_ALIGNED"),
                                      ("PACO_ROBUST_DENOM_ALIGNED",
                                       "PACO_ROBUST_NUM_ALIGNED"),
                                       ("PACO2_ROBUST_WW_DENOM_ALIGNED",
                                        "PACO2_ROBUST_WW_NUM_ALIGNED"),
                                       ("PACO2_ROBUST_WWM_DENOM_ALIGNED",
                                        "PACO2_ROBUST_WWM_NUM_ALIGNED"))) == false
            error("incompatible numerator ($(b.HDUNAME)) ",
                  "and denominator ($(a.HDUNAME))")
        end
        # Check pixel scale.
        pixtoarc = a["PIXTOARC"]::Float64
        b["PIXTOARC"]::Float64 == pixtoarc ||
            error("numerator and denominator maps have different pixel sizes")
        for (key,val) in (("CDELT1", -pixtoarc), ("CDELT2", +pixtoarc))
            a[key]::Float64 ≈ val || error("inconsistent denominator $key")
            b[key]::Float64 ≈ val || error("inconsistent numerator $key")
        end
        # Check date of observation (ISO 8601 standard).
        date_obs = a["DATE-OBS"]::String
        b["DATE-OBS"]::String == date_obs ||
        error("numerator and denominator maps have different observation dates")
        # Check date of observation (MJD).
        mjd = a["MJD-OBS"]::Float64
        b["MJD-OBS"]::Float64 == mjd ||
            error("numerator and denominator maps have different dates")
        # Check filter combination name.
        iflt = a["ESO INS COMB IFLT"]::String
        b["ESO INS COMB IFLT"]::String == iflt ||
            error("numerator and denominator maps have different filters")
        # Check infrared coronograph combination name
        icor = a["ESO INS COMB ICOR"]::String
        b["ESO INS COMB ICOR"]::String == icor || error("numerator and
                           denominator maps have different coronagraphic masks")
        # Check center.
        center = central_pixel(a)
        central_pixel(b) == center ||
            error("numerator and denominator maps have different centers")
        # Update list of data.
        push!(sizes, size(a))
        push!(B, Array(b))
        push!(A, Array(a))
        push!(dates_obs, date_obs)
        push!(epochs, mjd_to_epoch(mjd))
        push!(pixres, pixtoarc)
        push!(centers, center)
        push!(filters, iflt)
        push!(coronagraphs, icor)
    end

    if mode=="asdi" && asdi_BB
        add_BB_data!(listdir, dates_obs, epochs, pixres, centers, sizes, A, B,
                          filters, coronagraphs)
    end

    if !all((x) -> x == sizes[1], sizes)
        error("the loaded images are not all of the same size!")
    end

    i = sortperm(epochs)
    # return listdir[i]
    println(" DONE!")
    return PacomeData(dates_obs[i], epochs[i], pixres[i], centers[i], sizes[1],
                      A[i], B[i], filters[i], coronagraphs[i])
end

function add_BB_data!(listdir::Vector{String}, dates_obs::Vector{String},
                      epochs::Vector{T},
                      pixres::Vector{T}, ctrs::Vector{Point{T}},
                      sizes::Vector{NTuple{N,Int}},
                      A::Vector{Array{T,N}}, B::Vector{Array{T,N}},
                      flts::Vector{String},
                      cors::Vector{String}) where {T<:AbstractFloat, N}

   paco_path = "1_reduction_paco/adi_results/all_fits"
   num_pat = "*_num_aligned_*"
   den_pat = "*_denom_aligned_*"

   for dir in listdir
      isdir(joinpath(dir, paco_path)) ? nothing : continue
      file_hdr = glob("*.fits", joinpath(dir, paco_path))
      isempty(file_hdr) ? continue : hdr = read(FitsHeader, first(file_hdr))
      occursin("BB_", hdr["ESO INS COMB IFLT"]) ? nothing : continue
      print(".")

      file_num = glob(num_pat, joinpath(dir, paco_path))
      file_den = glob(den_pat, joinpath(dir, paco_path))
      length(file_num) == 1 ? file_num = first(file_num) :
                                      error("several numerators were found")
      length(file_den) == 1 ? file_den = first(file_den) :
                                    error("several denominators were found")
      b = read(FitsImage{T}, file_num)
      a = read(FitsImage{T}, file_den)
      size(a) == size(b) ||
          error("numerator and denominator maps have different sizes")

      arr_a, arr_b = get(Array, a)[:,:,1], get(Array, b)[:,:,1]
      a = FitsImage(reshape(arr_a, (size(arr_a)...,1)), get(FitsHeader, a))
      b = FitsImage(reshape(arr_b, (size(arr_b)...,1)), get(FitsHeader, b))

      length(size(a)) == N || error("expecting $N-dimensional maps")
      haskey(a, "HDUNAME") || fix_paco_image!(a, guess_paco_hduname(file_den))
      haskey(b, "HDUNAME") || fix_paco_image!(b, guess_paco_hduname(file_num))
      if ((a["HDUNAME"]::String,
           b["HDUNAME"]::String) ∈ (("PACO_DENOM_ALIGNED",
                                     "PACO_NUM_ALIGNED"),
                                    ("PACO_ROBUST_DENOM_ALIGNED",
                                     "PACO_ROBUST_NUM_ALIGNED"),
                                     ("PACO2_ROBUST_WW_DENOM_ALIGNED",
                                      "PACO2_ROBUST_WW_NUM_ALIGNED"),
                                     ("PACO2_ROBUST_WWM_DENOM_ALIGNED",
                                      "PACO2_ROBUST_WWM_NUM_ALIGNED"))) == false
          error("incompatible numerator ($(b.HDUNAME)) ",
                "and denominator ($(a.HDUNAME))")
      end
      # Check pixel scale.
      pixtoarc = a["PIXTOARC"]::Float64
      b["PIXTOARC"]::Float64 == pixtoarc ||
          error("numerator and denominator maps have different pixel sizes")
      for (key,val) in (("CDELT1", -pixtoarc), ("CDELT2", +pixtoarc))
          a[key]::Float64 ≈ val || error("inconsistent denominator $key")
          b[key]::Float64 ≈ val || error("inconsistent numerator $key")
      end
      # Check date of observation (ISO 8601 standard).
      date_obs = a["DATE-OBS"]::String
      b["DATE-OBS"]::String == date_obs ||
      error("numerator and denominator maps have different observation dates")
      # Check date of observation (MJD).
      mjd = a["MJD-OBS"]::Float64
      b["MJD-OBS"]::Float64 == mjd ||
          error("numerator and denominator maps have different dates")
      # Check filter combination name.
      iflt = a["ESO INS COMB IFLT"]::String
      b["ESO INS COMB IFLT"]::String == iflt ||
          error("numerator and denominator maps have different filters")
      # Check infrared coronograph combination name
      icor = a["ESO INS COMB ICOR"]::String
      b["ESO INS COMB ICOR"]::String == icor || error("numerator and
                         denominator maps have different coronagraphic masks")
      # Check center.
      center = central_pixel(a)
      central_pixel(b) == center ||
          error("numerator and denominator maps have different centers")
      # Update list of data.
      push!(sizes, size(a))
      push!(B, Array(b))
      push!(A, Array(a))
      push!(dates_obs, date_obs)
      push!(epochs, mjd_to_epoch(mjd))
      push!(pixres, pixtoarc)
      push!(ctrs, center)
      push!(flts, iflt)
      push!(cors, icor)
   end
end

"""
    mask_sources!(dat, orb; rad, val) -> nothing

applies a mask in PACOME's b terms contained in data `dat` to hide the sources
whose projected positions on the detector are given by the orbit `orb`. The
parameter `orb` can be a single orbit or a collection of orbits.
Keyword argument `rad` is the radius of the mask, it is set to `rad=10` by
default and `val` is the numerical value used to mask the sources, default is
`val=0`.

"""
function mask_sources!(dat::PacomeData{T,3},
                       orb::Orbit{T};
                       rad::Int=10,
                       val::T=0.) where {T<:AbstractFloat}

    nt = length(dat)
    nx, ny, nλ = dat.dims

    coord_x = ones(Int, nx)' .* (1:nx)
    coord_y = (1:ny)' .* ones(Int, ny)

    for t in 1:nt
        ΔRA, ΔDec = projected_position(orb, dat.epochs[t]; polar=false)
        pt = round(Int, dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t])

        dist = sqrt.((coord_x .- pt.x).^2+(coord_y .- pt.y).^2)
        idx_m = findall(x-> x < rad, dist)

        for λ in 1:nλ
            dat.b[t][idx_m,λ] .= val
        end
    end
end

function mask_sources!(dat::PacomeData{T,3},
                       orbs::Vector{Orbit{T}};
                       kwds...) where {T<:AbstractFloat}

   for orb in orbs
       mask_sources!(dat, orb; kwds...)
   end
end

"""
    mask_centers!(dat, orb; rad, val) -> nothing

applies a mask in PACOME's b terms contained in data `dat` at the center of the
images. The size of the mask for each epoch is directly adapted to the size of
the coranagraphic mask. Keyword argument `val` is the  numerical value used to
mask the sources, default is `val=0`.

"""
function mask_centers!(dat::PacomeData{T,3}; kwds...) where {T<:AbstractFloat}

    rad = [SPHERE_COR_DIAM[dat.icors[t]]/dat.pixres[t]/2 for t in 1:length(dat)]
    mask_centers!(dat, rad; kwds...)
end

function mask_centers!(dat::PacomeData{T,3}, rad::AbstractVector{T};
    val::T=0., inside::Bool=true, frac::T=1.) where {T<:AbstractFloat}

    @assert 0 ≤ frac ≤ 1
    nt = length(dat)
    @assert length(rad) == nt
    nx, ny, nλ = dat.dims

    coord_x = ones(Int, nx)' .* (1:nx)
    coord_y = (1:ny)' .* ones(Int, ny)

    for t in 1:nt
        pt = dat.centers[t]
        dist = sqrt.((coord_x .- pt.x).^2+(coord_y .- pt.y).^2)
        if inside
            idx_m = findall(x-> x < rad[t]*frac, dist)
        else
            idx_m = findall(x-> x >= rad[t]*frac, dist)
        end

        for λ in 1:nλ
            dat.b[t][idx_m,λ] .= val
        end
    end
end

function mask_centers!(dat::PacomeData{T,3}, rad::T;
                       kwds...) where {T<:AbstractFloat}

    mask_centers!(dat, repeat([rad],length(dat)); kwds...)
end

"""
    integrate!(a, b, A, B, ker, pt) -> a,b

integrates in `a` and `b` the values of the maps `A` and `B` interpolated by
kernel `ker` at position `pt`.

---
    integrate!(a, b, dat, orb, ker; kwds...) -> a, b

integrates in `a` and `b` the values of the maps `A` and `B` at the epochs
stored by the data `dat` for the orbital parameters `orb` interpolated by
kernel `ker` (a linear spline is used by default).  Arguments `a` and `b` are
resized if needed and initially zero-filled.

---
    integrate!(ab, dat, orb, ker; kwds...) -> a, b

for cases where data `dat` is uncalibrated. Integrates in `a` and `b` the
values of the maps `A` and `B` at the epochs stored by the data `dat` for the
orbital parameters `orb` interpolated by kernel `ker` (a linear spline is used
by default).  Arguments `a` and `b` are resized if needed and initially
zero-filled.

---
    integrate!(a, b, dat, orb, ker, field; kwds...) -> a, b

integrates in `a` and `b` the values of the maps `A` and `B` at the epochs
stored by the data `dat` for the orbital parameters `orb` interpolated by
kernel `ker` (a linear spline is used by default).  Arguments `a` and `b` are
resized if needed and initially zero-filled. Additionally, it fills a `field`
map (of the same size as `A` and `B`) at all pixellic interpolated positions.

---
    integrate!(ab, dat, orb, ker, field; kwds...) -> a, b

for cases where data `dat` is uncalibrated. Integrates in `a` and `b` the
values of the maps `A` and `B` at the epochs stored by the data `dat` for the
orbital parameters `orb` interpolated by kernel `ker` (a linear spline is used
by default).  Arguments `a` and `b` are resized if needed and initially
zero-filled. Additionally, it fills a `field` map (of the same size as `A` and
`B`) at all pixellic interpolated positions.

"""
function integrate!(a::AbstractVector{T},
                    b::AbstractVector{T},
                    dat::PacomeData{T,3},
                    orb::Orbit{T},
                    ker::Kernel{T};
                    kwds...) where {T<:AbstractFloat}
    dims = dat.dims
    nλ = dims[end]
    length(a) == nλ || resize!(a, nλ)
    length(b) == nλ || resize!(b, nλ)
    fill!(a, 0)
    fill!(b, 0)
    for t in 1:length(dat.epochs)
        # Projected position at the time of observation.
        ΔRA, ΔDec = projected_position(orb, dat.epochs[t]; polar=false, kwds...)

        # Convert tangential Right Ascension and Declination (ra,dec) into
        # pixel position in the image.
        pixpos = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]

        # Integrate numerator and denominator of SNR estimator.
        integrate!(a, b, dat.a[t], dat.b[t], ker, pixpos)
    end
    return a, b
end

function integrate!(a::AbstractVector{T},
                    b::AbstractVector{T},
                    dat::PacomeData{T,3},
                    orb::Orbit{T},
                    ker::Kernel{T},
                    field::Array{T,2};
                    kwds...) where {T<:AbstractFloat}
    dims = dat.dims
    nλ = dims[end]
    length(a) == nλ || resize!(a, nλ)
    length(b) == nλ || resize!(b, nλ)
    fill!(a, 0)
    fill!(b, 0)
    for t in 1:length(dat.epochs)
        # Projected position at the time of observation.
        ΔRA, ΔDec = projected_position(orb, dat.epochs[t]; polar=false, kwds...)

        # Convert tangential Right Ascension and Declination (ra,dec) into
        # pixel position in the image.
        pixpos = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]

        field[round(Int,pixpos.x), round(Int,pixpos.y)] += 1
        # Integrate numerator and denominator of SNR estimator.
        integrate!(a, b, dat.a[t], dat.b[t], ker, pixpos)
    end
    return a, b
end

function integrate!(ab::AbstractVector{T},
                    dat::PacomeData{T,3},
                    orb::Orbit{T},
                    ker::Kernel{T};
                    kwds...) where {T<:AbstractFloat}
    dims = dat.dims
    nλ = dims[end]
    length(ab) == nλ || resize!(ab, nλ)
    fill!(ab, 0)
    for t in 1:length(dat.epochs)
        # Projected position at the time of observation.
        ΔRA, ΔDec = projected_position(orb, dat.epochs[t]; polar=false, kwds...)

        # Convert tangential Right Ascension and Declination (ra,dec) into
        # pixel position in the image.
        pixpos = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]

        # Integrate numerator and denominator of SNR estimator.
        integrate!(ab, dat.a[t], dat.b[t], ker, pixpos)
    end
    return ab
end

function integrate!(ab::AbstractVector{T},
                    dat::PacomeData{T,3},
                    orb::Orbit{T},
                    ker::Kernel{T},
                    field::Array{T,2};
                    kwds...) where {T<:AbstractFloat}
    dims = dat.dims
    nλ = dims[end]
    length(ab) == nλ || resize!(ab, nλ)
    fill!(ab, 0)
    for t in 1:length(dat.epochs)
        # Projected position at the time of observation.
        ΔRA, ΔDec = projected_position(orb, dat.epochs[t]; polar=false, kwds...)

        # Convert tangential Right Ascension and Declination (ra,dec) into
        # pixel position in the image.
        pixpos = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]

        field[round(Int,pixpos.x), round(Int,pixpos.y)] += 1
        # Integrate numerator and denominator of SNR estimator.
        integrate!(ab, dat.a[t], dat.b[t], ker, pixpos)
    end
    return ab
end

function integrate!(a::AbstractVector{T},
                    b::AbstractVector{T},
                    A::AbstractArray{T,3},
                    B::AbstractArray{T,3},
                    ker::Kernel{T},
                    x::Real,
                    y::Real) where {T<:AbstractFloat}
    integrate!(a, b, A, B, ker, Point{T}(x, y))
end

function integrate!(ab::AbstractVector{T},
                    A::AbstractArray{T,3},
                    B::AbstractArray{T,3},
                    ker::Kernel{T},
                    x::Real,
                    y::Real) where {T<:AbstractFloat}
    integrate!(ab, A, B, ker, Point{T}(x, y))
end

function integrate!(a::AbstractVector{T},
                    b::AbstractVector{T},
                    A::AbstractArray{T,3},
                    B::AbstractArray{T,3},
                    ::BSpline{1,T},
                    pt::Point{T}) where {T<:AbstractFloat}
    @assert !Base.has_offset_axes(a, b, A, B)
    nx, ny, nλ = size(A)
    @assert size(B) == (nx, ny, nλ)
    @assert length(a) == nλ
    @assert length(b) == nλ
    i = round(Int, pt.x)
    j = round(Int, pt.y)
    if 1 ≤ i ≤ nx && 1 ≤ j ≤ ny
        @inbounds @simd for k in 1:nλ
            a[k] += A[i,j,k]
            b[k] += B[i,j,k]
        end
    end
    return a, b
end

function integrate!(ab::AbstractVector{T},
                    A::AbstractArray{T,3},
                    B::AbstractArray{T,3},
                    ::BSpline{1,T},
                    pt::Point{T}) where {T<:AbstractFloat}
    @assert !Base.has_offset_axes(ab, A, B)
    nx, ny, nλ = size(A)
    @assert size(B) == (nx, ny, nλ)
    @assert length(ab) == nλ
    i = round(Int, pt.x)
    j = round(Int, pt.y)
    if 1 ≤ i ≤ nx && 1 ≤ j ≤ ny
        @inbounds @simd for k in 1:nλ
            A[i,j,k] > 0 ? ab[k] += B[i,j,k]/sqrt(A[i,j,k]) : nothing
        end
    end
    return ab
end

function integrate!(a::AbstractVector{T},
                    b::AbstractVector{T},
                    A::AbstractArray{T,3},
                    B::AbstractArray{T,3},
                    ::BSpline{2,T},
                    pt::Point{T}) where {T<:AbstractFloat}
    @assert !Base.has_offset_axes(a, b, A, B)
    nx, ny, nλ = size(A)
    @assert size(B) == (nx, ny, nλ)
    @assert length(a) == nλ
    @assert length(b) == nλ
    fx = floor(pt.x)
    fy = floor(pt.y)
    if (1 ≤ fx < nx) & (1 ≤ fy < ny)
        i = Int(fx)
        j = Int(fy)
        u1 = pt.x - fx
        v1 = pt.y - fy
        u0 = one(T) - u1
        v0 = one(T) - v1
        @inbounds for k in 1:nλ
            a[k] += (v0*(u0*A[i,j,k] + u1*A[i+1,j,k]) +
                     v1*(u0*A[i,j+1,k] + u1*A[i+1,j+1,k]))
        end
        @inbounds for k in 1:nλ
            b[k] += (v0*(u0*B[i,j,k] + u1*B[i+1,j,k]) +
                     v1*(u0*B[i,j+1,k] + u1*B[i+1,j+1,k]))
        end
    end
    return a, b
end

function integrate!(ab::AbstractVector{T},
                    A::AbstractArray{T,3},
                    B::AbstractArray{T,3},
                    ::BSpline{2,T},
                    pt::Point{T}) where {T<:AbstractFloat}
    @assert !Base.has_offset_axes(ab, A, B)
    nx, ny, nλ = size(A)
    @assert size(B) == (nx, ny, nλ)
    @assert length(ab) == nλ
    fx = floor(pt.x)
    fy = floor(pt.y)
    if (1 ≤ fx < nx) & (1 ≤ fy < ny)
        i = Int(fx)
        j = Int(fy)
        u1 = pt.x - fx
        v1 = pt.y - fy
        u0 = one(T) - u1
        v0 = one(T) - v1
        @inbounds for k in 1:nλ
            bk = v0*(u0*B[i,j,k] + u1*B[i+1,j,k]) +
                 v1*(u0*B[i,j+1,k] + u1*B[i+1,j+1,k])
            ak = v0*(u0*A[i,j,k] + u1*A[i+1,j,k]) +
                 v1*(u0*A[i,j+1,k] + u1*A[i+1,j+1,k])

            ak > 0 ? ab[k] += bk /sqrt(ak) : nothing
        end
    end
    return ab
end

function integrate!(a::AbstractVector{T},
                    b::AbstractVector{T},
                    A::AbstractArray{T,3},
                    B::AbstractArray{T,3},
                    ker::Kernel{T,4},
                    pt::Point{T}) where {T<:AbstractFloat}
    @assert !Base.has_offset_axes(a, b, A, B)
    nx, ny, nλ = size(A)
    @assert size(B) == (nx, ny, nλ)
    @assert length(a) == nλ
    @assert length(b) == nλ

    x_off, x_wgt = compute_offset_and_weights(ker, pt.x)
    y_off, y_wgt = compute_offset_and_weights(ker, pt.y)

    # x_off + 1:4 and y_off + 1:4 must be in bounds
    if (0 ≤ x_off)&(x_off + 4 ≤ nx)&(0 ≤ y_off)&(y_off + 4 ≤ ny)
        @inbounds for k in 1:nλ
            a[k] += compute_interp_ker4(A, Int(x_off), x_wgt,
                                           Int(y_off), y_wgt, k)
        end
        @inbounds for k in 1:nλ
            b[k] += compute_interp_ker4(B, Int(x_off), x_wgt,
                                           Int(y_off), y_wgt, k)
        end
    end
    return a, b
end

function integrate!(ab::AbstractVector{T},
                    A::AbstractArray{T,3},
                    B::AbstractArray{T,3},
                    ker::Kernel{T,4},
                    pt::Point{T}) where {T<:AbstractFloat}
    @assert !Base.has_offset_axes(ab, A, B)
    nx, ny, nλ = size(A)
    @assert size(B) == (nx, ny, nλ)
    @assert length(ab) == nλ

    x_off, x_wgt = compute_offset_and_weights(ker, pt.x)
    y_off, y_wgt = compute_offset_and_weights(ker, pt.y)

    # x_off + 1:4 and y_off + 1:4 must be in bounds
    if (0 ≤ x_off)&(x_off + 4 ≤ nx)&(0 ≤ y_off)&(y_off + 4 ≤ ny)
        @inbounds for k in 1:nλ
            Bk = compute_interp_ker4(B, Int(x_off), x_wgt, Int(y_off), y_wgt, k)
            Ak = compute_interp_ker4(A, Int(x_off), x_wgt, Int(y_off), y_wgt, k)
            Ak > 0 ? ab[k] += Bk / sqrt(Ak) : nothing
        end
    end
    return ab
end

"""
   interpolate!(a, b, A, B, ker, pt, t) -> a, b

used for cases where the data is calibrated in flux.
interpolates the PACO maps `A` and `B` at subpixellic position `pt` and time
index `t` with interpolation kernel `ker` and adds the results in `a` and `b`.
If `A` and `B` store multi-channel data, a vector for `a` and `b` is returned.

---
   interpolate!(ab, A, B, ker, pt, t) -> ab

used for cases where the data is not calibrated in flux.
interpolates the PACO maps `A` and `B` at subpixellic position `pt` and time
index `t` with interpolation kernel `ker` and adds the results in `ab`.
If `A` and `B` store multi-channel data, a vector for `ab` is returned.

---
   interpolate(dat, orb, ker; kwds...) -> a, b

interpolates all individual temporal and spectral `a` and `b` PACO products from
data `dat` with interpolation kernel `ker` (of degree 4) for an orbit `orb`.

---
   interpolate(A, ker, pt)

interpolates the map `A` at position `pt` with interpolation kernel `ker` (of
degree 4). If `A` stores multi-channel data (3D array), the interpolation is
be performed at the same position for all channels and the result is a vector.

---
   interpolate(A, ker, pt, k)

interpolates the multi-channel map `A` at channel index `k` and position `pt`
with interpolation kernel `ker` (of degree 4).

"""
function interpolate!(a::AbstractMatrix{T},
                      b::AbstractMatrix{T},
                      A::AbstractArray{T,3},
                      B::AbstractArray{T,3},
                      ker::Kernel{T,4},
                      pt::Point{T},
                      t::Int) where {T<:AbstractFloat}
    @assert !Base.has_offset_axes(a, b, A, B)
    nx, ny, nλ, nt = size(A)..., last(size(a))
    @assert size(B) == (nx, ny, nλ)
    @assert size(a) == (nλ, nt)
    @assert size(b) == (nλ, nt)
    @assert 1 ≤ t ≤ nt

    x_off, x_wgt = compute_offset_and_weights(ker, pt.x)
    y_off, y_wgt = compute_offset_and_weights(ker, pt.y)

    if (0 ≤ x_off)&(x_off + 4 ≤ nx)&(0 ≤ y_off)&(y_off + 4 ≤ ny)
        @inbounds for k in 1:nλ
            a[k,t] = compute_interp_ker4(A, Int(x_off), x_wgt,
                                            Int(y_off), y_wgt, k)
        end
        @inbounds for k in 1:nλ
            b[k,t] = compute_interp_ker4(B, Int(x_off), x_wgt,
                                            Int(y_off), y_wgt, k)
        end
    else
        @inbounds for k in 1:nλ
            a[k,t] = zero(T)
        end
        @inbounds for k in 1:nλ
            b[k,t] = zero(T)
        end
    end
    return a, b
end

function interpolate!(ab::AbstractMatrix{T},
                      A::AbstractArray{T,3},
                      B::AbstractArray{T,3},
                      ker::Kernel{T,4},
                      pt::Point{T},
                      t::Int) where {T<:AbstractFloat}
    @assert !Base.has_offset_axes(ab, A, B)
    nx, ny, nλ, nt = size(A)..., last(size(a))
    @assert size(B) == (nx, ny, nλ)
    @assert size(ab) == (nλ, nt)
    @assert 1 ≤ t ≤ nt

    x_off, x_wgt = compute_offset_and_weights(ker, pt.x)
    y_off, y_wgt = compute_offset_and_weights(ker, pt.y)

    if (0 ≤ x_off)&(x_off + 4 ≤ nx)&(0 ≤ y_off)&(y_off + 4 ≤ ny)
        @inbounds for k in 1:nλ
            ab[k,t] = compute_interp_ker4(B, Int(x_off), x_wgt,
                                             Int(y_off), y_wgt, k) /
                      sqrt(compute_interp_ker4(A, Int(x_off), x_wgt,
                                                  Int(y_off), y_wgt, k))
        end
    else
        @inbounds for k in 1:nλ
            ab[k,t] = zero(T)
        end
    end
    return ab
end

function interpolate(dat::PacomeData{T,3},
                     orb::Orbit{T},
                     ker::Kernel{T};
                     kwds...) where {T<:AbstractFloat}
    @assert length(dat) > 0
    @assert !Base.has_offset_axes(dat.a[1], dat.a[1])
    nx, ny, nλ, nt = size(dat.a[1])..., last(length(dat.a))

    a = Array{T,2}(undef, (nλ, nt))
    b = Array{T,2}(undef, (nλ, nt))
    fill!(a, 0)
    fill!(b, 0)

    for t in 1:nt
        # Projected position at the time of observation.
        ΔRA, ΔDec = projected_position(orb, dat.epochs[t]; polar=false, kwds...)

        # Convert tangential Right Ascension and Declination (ra,dec) into
        # pixel position in the image.
        pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]

        # Integrate numerator and denominator of SNR estimator.
        interpolate!(a, b, dat.a[t], dat.b[t], ker, pt, t)
    end
    return a, b
end

function interpolate(A::AbstractArray{T,2},
                     ker::BSpline{1,T},
                     pt::Point{T}) where {T<:AbstractFloat}

    nx, ny = size(A)
    rpt = round(Int, pt)
    @assert (1 ≤ rpt.x ≤ nx) && (1 ≤ rpt.y ≤ ny)
    return A[rpt.x, rpt.y]
end

function interpolate(A::AbstractArray{T,2},
                     ker::BSpline{2,T},
                     pt::Point{T}) where {T<:AbstractFloat}
    nx, ny = size(A)
    fx = floor(pt.x)
    fy = floor(pt.y)
    if (1 ≤ fx < nx) & (1 ≤ fy < ny)
        i = Int(fx)
        j = Int(fy)
        u1 = pt.x - fx
        v1 = pt.y - fy
        u0 = one(T) - u1
        v0 = one(T) - v1
        return (v0*(u0*A[i,j] + u1*A[i+1,j]) +
                v1*(u0*A[i,j+1] + u1*A[i+1,j+1]))
    else
        error("Out of bound !")
    end
end

function interpolate(A::AbstractArray{T,2},
                     ker::Kernel{T,4},
                     pt::Point{T}) where {T<:AbstractFloat}

    x_off, x_wgt = compute_offset_and_weights(ker, pt.x)
    y_off, y_wgt = compute_offset_and_weights(ker, pt.y)
    return compute_interp_ker4(A, Int(x_off), x_wgt, Int(y_off), y_wgt)
end

function interpolate(A::AbstractArray{T,3},
                     ker::Kernel{T,4},
                     pt::Point{T}) where {T<:AbstractFloat}

    x_off, x_wgt = compute_offset_and_weights(ker, pt.x)
    y_off, y_wgt = compute_offset_and_weights(ker, pt.y)
    res = Array{T,1}(undef, size(A,3))
    for k in 1:size(A,3)
        res[k] = compute_interp_ker4(A, Int(x_off), x_wgt, Int(y_off), y_wgt, k)
    end
    return res
end

function interpolate(A::AbstractArray{T,3},
                     ker::Kernel{T,4},
                     pt::Point{T},
                     k::Int) where {T<:AbstractFloat}

    x_off, x_wgt = compute_offset_and_weights(ker, pt.x)
    y_off, y_wgt = compute_offset_and_weights(ker, pt.y)
    return compute_interp_ker4(A, Int(x_off), x_wgt, Int(y_off), y_wgt, k)
end

"""
   interpolate_ab_derivs_wrt_orb(dat, orb, ker; kwds...) -> a, b, ∂a, ∂b

interpolates all individual temporal and spectral `a` and `b` PACO products from
data `dat` with interpolation kernel `ker` (of degree 4) for an orbit `orb` as
well as the derivatives of `a` and `b` (`∂a` and `∂b` respectively) with respect
to the orbital paremeters of `orb`. The derivatives are computed with numerical
finite differences.

"""

function interpolate_ab_derivs_wrt_orb(dat::PacomeData{T,3},
                                       orb::Orbit{T},
                                       ker::Kernel{T,4};
                                       kwds...) where {T<:AbstractFloat}
    @assert length(dat) > 0
    @assert !Base.has_offset_axes(dat.a[1], dat.b[1])
    nx, ny, nλ, nt = size(dat.a[1])..., last(length(dat.a))
    ∂a_∂μ = fill!(Array{T,3}(undef, (7, nλ, nt)),0)
    ∂b_∂μ = fill!(Array{T,3}(undef, (7, nλ, nt)),0)

    # Retrieves the derivatives of the inteprolated values of a and b at
    # the 2-D projected positions
    a, b = interpolate_with_derivatives(dat, orb, ker)
    for t in 1:nt
        # Retrieves the derivatives of the position with resp. to the orb. elem.
        # and converts the units to match those of ∂a_∂θ and ∂b_∂θ
        _, ∂θ_∂μ = projected_position_derivs(orb, dat.epochs[t])

        ∂θ_∂μ[:,3] = ∂θ_∂μ[:,3]/rad2deg(1) # inclination in deg
        ∂θ_∂μ[:,5] = ∂θ_∂μ[:,5]/rad2deg(1) # arg. of periapsis in deg
        ∂θ_∂μ[:,6] = ∂θ_∂μ[:,6]/rad2deg(1) # long. ascending node in deg
        for μ in 1:7
            ∂θ_∂μ[:,μ] = ∂θ_∂μ[:,μ]/dat.pixres[t] # positions in pixels
            ∂θ_∂μ[1,μ] = -∂θ_∂μ[1,μ] # -ΔRA in the x direction
            for λ in 1:nλ
                ∂a_∂μ[μ,λ,t] = a[2,λ,t]*∂θ_∂μ[1,μ] + a[3,λ,t]*∂θ_∂μ[2,μ]
                ∂b_∂μ[μ,λ,t] = b[2,λ,t]*∂θ_∂μ[1,μ] + b[3,λ,t]*∂θ_∂μ[2,μ]
            end
        end
    end

    return a[1,:,:], b[1,:,:], ∂a_∂μ, ∂b_∂μ
end


"""
   integrate_with_derivatives!(a, b, A, B, ker, ker_prime, pt) -> a, b

integrates in `a` and `b` the values of the maps `A` and `B` interpolated by
kernel `ker` at position `pt` as well as the values of the derivatives of `a`
and `b` interpolated by kernel `ker_prime` at the same position.

---
   interpolate_with_derivatives!(a, b, A, B, ker, ker_prime, pt, t) -> a, b

interpolates the PACO maps `A` and `B` at subpixellic position `pt` and time
index `t` with interpolation kernel `ker` (and their derivatives  with kernel
`ker_prime`) and outputs the results in `a` and `b`.

---
   interpolate_with_derivatives(dat, orb, ker; kwds) -> a, b

interpolates in `a` and `b` the values of the PACO maps and their derivatives at
the epochs stored by the data `dat` for the orbital parameters `orb`
interpolated by kernel `ker` (and `ker_prime` for the derivatives).

"""
function integrate_with_derivatives!(a::AbstractArray{T,2},
                                     b::AbstractArray{T,2},
                                     A::AbstractArray{T,3},
                                     B::AbstractArray{T,3},
                                     ker::Kernel{T,4},
                                     ker_prime::Kernel{T,4},
                                     pt::Point{T}) where {T<:AbstractFloat}

    @assert !Base.has_offset_axes(a, b, A, B)
    nx, ny, nλ = size(A)
    @assert size(B) == (nx, ny, nλ)
    @assert size(a) == (3, nλ)
    @assert size(b) == (3, nλ)

    x_off, x_wgt = compute_offset_and_weights(ker, pt.x)
    y_off, y_wgt = compute_offset_and_weights(ker, pt.y)
    x_off_prime, x_wgt_prime = compute_offset_and_weights(ker_prime, pt.x)
    y_off_prime, y_wgt_prime = compute_offset_and_weights(ker_prime, pt.y)

    @assert x_off_prime == x_off
    @assert y_off_prime == y_off

    # x_off + 1:4 and y_off + 1:4 must be in bounds
    if (0 ≤ x_off)&(x_off + 4 ≤ nx)&(0 ≤ y_off)&(y_off + 4 ≤ ny)
        @inbounds for k in 1:nλ
            a[1,k] += compute_interp_ker4(A, Int(x_off), x_wgt,       Int(y_off), y_wgt,       k)
            a[2,k] += compute_interp_ker4(A, Int(x_off), x_wgt_prime, Int(y_off), y_wgt,       k)
            a[3,k] += compute_interp_ker4(A, Int(x_off), x_wgt,       Int(y_off), y_wgt_prime, k)
        end
        @inbounds for k in 1:nλ
            b[1,k] += compute_interp_ker4(B, Int(x_off), x_wgt,       Int(y_off), y_wgt,       k)
            b[2,k] += compute_interp_ker4(B, Int(x_off), x_wgt_prime, Int(y_off), y_wgt,       k)
            b[3,k] += compute_interp_ker4(B, Int(x_off), x_wgt,       Int(y_off), y_wgt_prime, k)
        end
    end
    return a, b
end

function interpolate_with_derivatives!(a::AbstractArray{T,3},
                                       b::AbstractArray{T,3},
                                       A::AbstractVector{<:AbstractArray{T,3}},
                                       B::AbstractVector{<:AbstractArray{T,3}},
                                       ker::Kernel{T,4},
                                       ker_prime::Kernel{T,4},
                                       pt::Point{T},
                                       t::Int) where {T<:AbstractFloat}
    interpolate_with_derivatives!(a, b, A[t], B[t], ker, ker_prime, pt, t)
end

function interpolate_with_derivatives!(a::AbstractArray{T,3},
                                       b::AbstractArray{T,3},
                                       A::AbstractArray{T,3},
                                       B::AbstractArray{T,3},
                                       ker::Kernel{T,4},
                                       ker_prime::Kernel{T,4},
                                       pt::Point{T},
                                       t::Int) where {T<:AbstractFloat}
    @assert !Base.has_offset_axes(a, b, A, B)
    nx, ny, nλ, nt = size(A)..., last(size(a))
    @assert size(B) == (nx, ny, nλ)
    @assert size(a) == (3, nλ, nt)
    @assert size(b) == (3, nλ, nt)
    @assert 1 ≤ t ≤ nt

    x_off, x_wgt = compute_offset_and_weights(ker, pt.x)
    y_off, y_wgt = compute_offset_and_weights(ker, pt.y)
    x_off_prime, x_wgt_prime = compute_offset_and_weights(ker_prime, pt.x)
    y_off_prime, y_wgt_prime = compute_offset_and_weights(ker_prime, pt.y)

    @assert x_off_prime == x_off
    @assert y_off_prime == y_off

    # x_off + 1:4 and y_off + 1:4 must be in bounds
    if (0 ≤ x_off)&(x_off + 4 ≤ nx)&(0 ≤ y_off)&(y_off + 4 ≤ ny)
        @inbounds for k in 1:nλ
            a[1,k,t] = compute_interp_ker4(A, Int(x_off), x_wgt,       Int(y_off), y_wgt,       k)
            a[2,k,t] = compute_interp_ker4(A, Int(x_off), x_wgt_prime, Int(y_off), y_wgt,       k)
            a[3,k,t] = compute_interp_ker4(A, Int(x_off), x_wgt,       Int(y_off), y_wgt_prime, k)
        end
        @inbounds for k in 1:nλ
            b[1,k,t] = compute_interp_ker4(B, Int(x_off), x_wgt,       Int(y_off), y_wgt,       k)
            b[2,k,t] = compute_interp_ker4(B, Int(x_off), x_wgt_prime, Int(y_off), y_wgt,       k)
            b[3,k,t] = compute_interp_ker4(B, Int(x_off), x_wgt,       Int(y_off), y_wgt_prime, k)
        end
    else
        @inbounds for k in 1:nλ
            a[1,k,t] = zero(T)
            a[2,k,t] = zero(T)
            a[3,k,t] = zero(T)
        end
        @inbounds for k in 1:nλ
            b[1,k,t] = zero(T)
            b[2,k,t] = zero(T)
            b[3,k,t] = zero(T)
        end
    end
    return a, b
end

function interpolate_with_derivatives(dat::PacomeData{T,3},
                                      orb::Orbit{T},
                                      ker::Kernel{T,4};
                                      kwds...) where {T<:AbstractFloat}

    @assert length(dat) > 0
    @assert !Base.has_offset_axes(dat.a[1], dat.a[1])
    nx, ny, nλ, nt = size(dat.a[1])..., last(length(dat.a))

    a = Array{T,3}(undef, (3, nλ, nt))
    b = Array{T,3}(undef, (3, nλ, nt))
    fill!(a, 0)
    fill!(b, 0)

    for t in 1:nt
        # Projected position at the time of observation.
        ΔRA, ΔDec = projected_position(orb, dat.epochs[t]; polar=false, kwds...)

        # Convert tangential Right Ascension and Declination (ra,dec) into
        # pixel position in the image.
        pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]

        # Integrate numerator and denominator of SNR estimator.
        interpolate_with_derivatives!(a, b, dat.a[t], dat.b[t], ker, ker', pt, t)
    end

    return a, b
end


"""
   compute_interp_ker4(A, off1, wgt1, off2, wgt2, k)

interpolates the 3D map `A` with offsets `off1` and `off2` and weights `wgt1`
and `wgt2` at index `k` along the 3rd dimension.

---
   compute_interp_ker4(A, off1, wgt1, off2, wgt2)

interpolates the 2D map `A` with offsets `off1` and `off2` and weights `wgt1`
and `wgt2`.

"""
@inline @propagate_inbounds function compute_interp_ker4(A::AbstractArray{T,3},
                                                 off1::Int,
                                                 wgt1::NTuple{4,T},
                                                 off2::Int,
                                                 wgt2::NTuple{4,T},
                                                 k::Int
                                                 ) where {T<:AbstractFloat}
    i1, i2, i3, i4 = off1 + 1, off1 + 2, off1 + 3, off1 + 4
    j1, j2, j3, j4 = off2 + 1, off2 + 2, off2 + 3, off2 + 4

    res =  ((A[i1,j1,k]*wgt1[1] +
             A[i2,j1,k]*wgt1[2] +
             A[i3,j1,k]*wgt1[3] +
             A[i4,j1,k]*wgt1[4])*wgt2[1] +
            (A[i1,j2,k]*wgt1[1] +
             A[i2,j2,k]*wgt1[2] +
             A[i3,j2,k]*wgt1[3] +
             A[i4,j2,k]*wgt1[4])*wgt2[2] +
            (A[i1,j3,k]*wgt1[1] +
             A[i2,j3,k]*wgt1[2] +
             A[i3,j3,k]*wgt1[3] +
             A[i4,j3,k]*wgt1[4])*wgt2[3] +
            (A[i1,j4,k]*wgt1[1] +
             A[i2,j4,k]*wgt1[2] +
             A[i3,j4,k]*wgt1[3] +
             A[i4,j4,k]*wgt1[4])*wgt2[4])

    if isnan(res)
        return zero(T)
    else
        return res
    end
end

@inline @propagate_inbounds function compute_interp_ker4(A::AbstractArray{T,2},
                                                 off1::Int,
                                                 wgt1::NTuple{4,T},
                                                 off2::Int,
                                                 wgt2::NTuple{4,T}
                                                 ) where {T<:AbstractFloat}
    i1, i2, i3, i4 = off1 + 1, off1 + 2, off1 + 3, off1 + 4
    j1, j2, j3, j4 = off2 + 1, off2 + 2, off2 + 3, off2 + 4
    res =  ((A[i1,j1]*wgt1[1] +
             A[i2,j1]*wgt1[2] +
             A[i3,j1]*wgt1[3] +
             A[i4,j1]*wgt1[4])*wgt2[1] +
            (A[i1,j2]*wgt1[1] +
             A[i2,j2]*wgt1[2] +
             A[i3,j2]*wgt1[3] +
             A[i4,j2]*wgt1[4])*wgt2[2] +
            (A[i1,j3]*wgt1[1] +
             A[i2,j3]*wgt1[2] +
             A[i3,j3]*wgt1[3] +
             A[i4,j3]*wgt1[4])*wgt2[3] +
            (A[i1,j4]*wgt1[1] +
             A[i2,j4]*wgt1[2] +
             A[i3,j4]*wgt1[3] +
             A[i4,j4]*wgt1[4])*wgt2[4])

     if isnan(res)
         return zero(T)
     else
         return res
     end
end

"""
    error_orb_elem_CramerRao(dat, orb, ker) -> σ_μ, Iμ

estimates the errors associated to the orbital elements of `orb` through
Cramer-Rao bounds with respect to PACO data `dat` interpolated by `ker`.
Interpolation kernel `ker` must to be of support 4.

The returned value `σ_μ` is a 7-elements vector containing the errors
of each orbital elements (resp. : a, e, i, τ, ω, Ω, K).

`Iμ` encodes the temporally combined Fisher information matrix that gathers the
informations of all epochs and from which the errors are computed.

The interpolated values of a, b PACO products and their derivatives with respect
to the orbital elements are computed by numerical finite differences.

"""
function error_orb_elem_CramerRao(dat::PacomeData{T,3},
                                  orb::Orbit{T},
                                  ker::Kernel{T,4};
                                  λ::Int=0) where {T<:AbstractFloat}

    nt = length(dat)
    nλ = dat.dims[end]
    @assert 0 ≤ λ ≤ nλ
    λ == 0 ? (λs = Vector(1:nλ)) : λs = [λ]

    # Fisher matrices encoding all the epochs
    Itot = fill!(Array{T,3}(undef, (7,7,nt)),0)

    # Fills in the Fisher information matrices
    error_orb_elem_CramerRao!(Itot, dat, orb, ker, λs)

    # Gathers the Fisher information matrices together
    Iμ = reshape(sum(Itot, dims=3), (7,7))
    inv_Iμ = fill!(Array{T,2}(undef, size(Iμ)), NaN)
    σ_μ = fill!(Array{T,1}(undef, 7), NaN)
    Iμ = Symmetric(Iμ)
    inv_Iμ = inv(Iμ)
    diago = diag(inv_Iμ)
    any(diago .< 0) ? nothing : σ_μ = sqrt.(diago)

    return σ_μ, inv_Iμ
end

function error_orb_elem_CramerRao!(Itot::AbstractArray{T,3},
                                   dat::PacomeData{T,3},
                                   orb::Orbit{T},
                                   ker::Kernel{T,4},
                                   λs::Vector{Int}) where {T<:AbstractFloat}

    nt = length(dat)

    # Retrieves the interpolated values of a, b, and their derivatives at the
    # corresponding projected positions computed by finite differences
    a, b, ∂a∂μ, ∂b∂μ = interpolate_ab_derivs_wrt_orb(dat, orb, ker)

    @assert !any(isnan, a)
    @assert !any(isnan, b)
    @assert !any(zero(T) in a)

    for t in 1:nt
        for i in 1:7
            for j in 1:i
                Li = Lj = 0
                for λ in λs
                    Li += max(b[λ,t],0)/a[λ,t] * ∂b∂μ[i,λ,t] -
                          0.5 * (max(b[λ,t],0)/a[λ,t])^2 * ∂a∂μ[i,λ,t]
                    Lj += max(b[λ,t],0)/a[λ,t] * ∂b∂μ[j,λ,t] -
                          0.5 * (max(b[λ,t],0)/a[λ,t])^2 * ∂a∂μ[j,λ,t]
                end
                Itot[j,i,t] = Itot[i,j,t] = Li*Lj
            end
        end
    end
end

"""
    error_orb_elem_pertubation(dat, orb, ker, grid; ...) ->

estimates a set of perturbed orbits using a perturbative method on the
orbital elements `orb` and PACO data `dat`. The inteprolation is performed by
`ker`. Interpolation kernel `ker` must to be of support 4.

"""

function error_orb_elem_pertubation(dat::PacomeData{T,3},
                                  orb::Orbit{T},
                                  ker::Kernel{T,4},
                                  grid::Grid{T};
                                  kwds...) where {T<:AbstractFloat}

   bounds = Tuple{Vector{T},Vector{T}}()
   lb = [grid.a[1], grid.e[1], grid.i[1], grid.τ[1], grid.ω[1],
         grid.Ω[1], grid.K[1]]
   ub = [grid.a[end], grid.e[end], grid.i[end], grid.τ[end], grid.ω[end],
         grid.Ω[end], grid.K[end]]
   return error_orb_elem_pertubation(dat, orb, ker, (lb,ub); kwds...)
end

function error_orb_elem_pertubation(dat::PacomeData{T,3},
                                  orb::Orbit{T},
                                  ker::Kernel{T,4},
                                  bounds::Tuple{Vector{T},Vector{T}};
                                  nb_iter::Int=1000,
                                  ROI::Int=500,
                                  cal::Bool=false,
                                  λ::Int=0,
                                  hard::Bool=true,
                                  maxeval::Int=100_000) where {T<:AbstractFloat}

    nthreads = Threads.nthreads()
    intervs = split_on_threads(nb_iter, nthreads)

    # Pixel positions of the object in the maps at all time t
    pts = Array{Int,2}(undef, (length(dat),2))
    for t in 1:length(dat)
        ΔRA, ΔDec = projected_position(orb, dat.epochs[t])
        pixpos = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]
        pixpos = round(Int, pixpos)
        pts[t,1] = pixpos.x
        pts[t,2] = pixpos.y

        idx = findall(x->x<=0, dat.a[t])
        dat.a[t][idx] .= NaN
    end

    orbs = Array{T,2}(undef, (7,nb_iter))

    print("\nComputing errors...")
    Threads.@threads for n in 1:nthreads
        error_orb_elem_pertubation!(dat, orb, pts, ker, bounds, orbs,
                                    intervs[n], ROI, cal, λ, hard, maxeval)
    end
    print(repeat("\n", nthreads))

    return orbs
end

function error_orb_elem_pertubation!(dat::PacomeData{T,3},
                                     orb::Orbit{T},
                                     pts::Array{Int,2},
                                     ker::Kernel{T,4},
                                     bounds::Tuple{Vector{T},Vector{T}},
                                     orbs::Array{T,2},
                                     interv::Tuple{Int64, Int64},
                                     ROI::Int,
                                     cal::Bool,
                                     λ::Int,
                                     hard::Bool,
                                     maxeval::Int) where {T<:AbstractFloat}

   id = Threads.threadid()
   dat_copy = deepcopy(dat)

   r = ROI÷2
   orb_arr = orb2arr(orb)

   k_sta, k_end = first(interv), last(interv)
   nk = k_end - k_sta + 1
   pbar = Progress(nk; dt=1, desc="Thread $id : ", start=0, offset=id)
   for k in k_sta:k_end
       for t in 1:length(dat_copy.b)
           x_l, x_u = pts[t,1]-r, pts[t,1]+r
           y_l, y_u = pts[t,2]-r, pts[t,2]+r
           dat_copy.b[t][x_l:x_u,y_l:y_u,:] = dat.b[t][x_l:x_u,y_l:y_u,:] + 1 *
                                        sqrt.(dat.a[t][x_l:x_u,y_l:y_u,:]) .*
                                        randn((ROI+1,ROI+1,dat.dims[end]))
       end

       try
           orbs[:,k], = optimize_orb_param(dat_copy, orb_arr, ker, bounds, pts, ROI;
                                          cal=cal, λ=λ, hard=hard, maxeval=maxeval)
       catch e
           orbs[:,k] .= NaN
       end
       update!(pbar, k-k_sta+1)
   end

   dat_copy = 0
end

"""
    PACOME_MT_mmap(data, grid, ker, nthreads) -> best_orbit

runs the PACOME algorithm on the `grid` structure using the multi-epoch
observations `data` and the interpolation kernel `ker`. A specified number of
threads `nthreads` is used for the search. The results are stored in a mmap file
`mmap_file_path`.

Optional argument `fcost_lim` corresponds to the minimal cost score above which
the orbits are saved, `cal` is to specify whether the data is calibrated or not.
The spectral channels to consider is given by `λ`.

"""
function PACOME_MT_mmap(data::PacomeData{T,N},
                        grid::Grid{T},
                        ker::Kernel{T},
                        nthreads::Int,
                        mmap_file_path::String;
                        fcost_lim::T=0.,
                        cal::Bool=false,
                        λ::Int=0) where {T<:AbstractFloat,N}

    nλ = data.dims[3]
    nt = length(data)
    cpt = fill!(Array{Int,1}(undef, nthreads), 0)

    ss = [open(split(mmap_file_path,".bin")[1] * "_part$i.bin", "w+") for i in 1:Threads.nthreads()]

    @unpack a, e, i, τ, ω, Ω, K = grid

    # Multi-threading
    Threads.@threads for k in 1:grid.norbits
        a_k, e_k, i_k, τ_k, ω_k, Ω_k, K_k = orb_comb(k, a, e, i, τ, ω, Ω, K)
        orb = Orbit{T}(a = a_k, e = e_k, i = i_k, τ = τ_k,
                       ω = ω_k, Ω = Ω_k, K = K_k)

        fcost = cost_func(data, orb, ker; cal=cal, λ=λ)

        if (fcost > fcost_lim)
            write(ss[Threads.threadid()], [fcost, k])
            cpt[Threads.threadid()] = cpt[Threads.threadid()] + 1
        end
    end

    for i in 1:Threads.nthreads()
        close(ss[i])
    end

    s = open(mmap_file_path, "w+")
    write(s, 2)
    write(s, sum(cpt))
    for n in 1:nthreads
        file = split(mmap_file_path,".bin")[1] * "_part$n.bin"
        ss = open(file)
        orbs = Mmap.mmap(ss, Matrix{T}, (2,cpt[n]))
        write(s, orbs)
        close(ss)
        rm(file, force=true)
    end
    close(s)

    s = open(mmap_file_path)
    m = read(s, Int)
    n = read(s, Int)
    all_orbs = Mmap.mmap(s, Matrix{T}, (m,n))
    idx_max = argmax(all_orbs[1,:])
    a_k, e_k, i_k, τ_k, ω_k, Ω_k, K_k = orb_comb(Int(all_orbs[2,idx_max]), grid)
    close(s)

    return [all_orbs[1,idx_max], a_k, e_k, i_k, τ_k, ω_k, Ω_k, K_k]
end

"""
    cost_func(dat, orb, ker; cal, λ) -> fcost

computes the value of the PACOME cost function (related to the snr) of the orbit
`orb` on data `dat` with kernel `ker`.

Keyword `cal` specifies whether the data is calibrated or not. Default is
`false`.

Keyword `λ` specifies whether the cost function should be combining the spectral
channels (`λ=0`) or computed only on specific channel (`λ=n` where n ∈ [1,nλ])

---
   cost_func(dat, orb, ker, pts, ROI, lower, upper, kwds...) -> fcost

does the same as `cost_func(dat, orb, ker; cal, λ)` but first checks if
all projected positions (where the interpolation will be performed) are inside
the region [`pts`-`ROI`,`pts`+`ROI`] and if the orbit `orb` is contrained
whithin the lower and upper bounds `lower` and `upper`.

"""
function cost_func(dat::PacomeData{T,N},
                  orb::Orbit{T},
                  ker::Kernel{T};
                  cal::Bool=false,
                  λ::Int=0) where {T<:AbstractFloat,N}

   nt = length(dat)
   nλ = dat.dims[end]
   @assert 0 ≤ λ ≤ nλ

   if orb.e > 0.999999
       x = orb2arr(orb)
       x[2] = 0.99999
       orb = arr2orb(x)
   end
   as, bs = interpolate(dat, orb, ker)

   if λ == 0
       fcost = 0
       if cal
           for λi in 1:nλ
               num = 0
               den = 0
               for t in 1:nt
                   num += bs[λi,t]
                   den += as[λi,t]
               end
               (num > 0 && den > 0) ? fcost += num^2/den : nothing
           end
       else
           for λi in 1:nλ
               for t in 1:nt
                   (bs[λi,t] > 0 && as[λi,t] > 0) ? fcost += bs[λi,t]^2 / as[λi,t] : nothing
               end
           end
       end
   else
       fcost = 0
       if cal
           num = 0
           den = 0
           for t in 1:nt
               num += bs[λ,t]
               den += as[λ,t]
           end
           (num > 0 && den > 0) ? fcost += num^2/den : nothing
       else
           for t in 1:nt
               (bs[λ,t] > 0 && as[λ,t] > 0) ? fcost += bs[λ,t]^2 / as[λ,t] : nothing
           end
       end
   end

   return fcost
end

function cost_func(dat::PacomeData{T,N},
                          orb::AbstractVector{T},
                          ker::Kernel{T};
                          kwds...) where {T<:AbstractFloat,N}

   return cost_func(dat, arr2orb(orb), ker; kwds...)
end

function cost_func(dat::PacomeData{T,N},
                          orb::Orbit{T},
                          ker::Kernel{T},
                          pts::Array{Int,2},
                          ROI::Int,
                          lower::Array{T,1},
                          upper::Array{T,1};
                          kwds...) where {T<:AbstractFloat,N}

   x = orb2arr(orb)
   for μ in 1:7
       (x[μ] < lower[μ] || x[μ] > upper[μ]) && return +Inf
   end

   for t in 1:length(dat)
       ΔRA, ΔDec = projected_position(orb, dat.epochs[t])
       pos = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]

       @assert pts[t,1] - ROI < pos.x < pts[t,1] + ROI
       @assert pts[t,2] - ROI < pos.y < pts[t,2] + ROI
   end

   return cost_func(dat, orb, ker; kwds...)
end

function cost_func(dat::PacomeData{T,N},
                          orb::Array{T,1},
                          ker::Kernel{T},
                          pts::Array{Int,2},
                          ROI::Int,
                          lower::Array{T,1},
                          upper::Array{T,1};
                          kwds...) where {T<:AbstractFloat,N}

   return cost_func(dat, arr2orb(orb), ker, pts, ROI, lower, upper; kwds...)
end

function cost_func(snrs::Vector{AbstractFloat})

   C = 0
   for snr in snrs
       isnan!(C) || snr > 0 ? (C += snr^2) : nothing
   end
   return C
end


"""
    cost_func_deriv_wrt_orb(dat, orb, ker; λ) -> fcost

computes the value of the PACOME cost function for the orbit `orb` as well as
its semi-analytic derivatives with respect to the orbital elements on data `dat`
with kernel `ker`.

Keyword `λ` specifies whether the cost function should be combining the spectral
channels (`λ=0`) or computed only on specific channel (`λ=n` where n ∈ [1,nλ])

"""
function cost_func_deriv_wrt_orb(dat::PacomeData{T,3}, orb::Orbit{T},
                                 ker::Kernel{T};
                                 λ::Int=0) where {T<:AbstractFloat}

   nt = length(dat)
   nλ = last(dat.dims)
   @assert 0 ≤ λ ≤ nλ

   a, b, ∂a, ∂b = interpolate_ab_derivs_wrt_orb(dat, orb, ker)
   ∂C = fill!(Vector{T}(undef, 7),0)
   C = 0
   if λ == 0
       for t in 1:nt
           for λ in 1:nλ
               if b[λ,t] > 0 && a[λ,t] > 0
                   C += b[λ,t]^2/a[λ,t]
                   for μ in 1:7
                       ∂C[μ] += 2*b[λ,t]/a[λ,t] * ∂b[μ,λ,t] -
                                (b[λ,t]/a[λ,t])^2 * ∂a[μ,λ,t]
                   end
               end
           end
       end
   else
       for t in 1:nt
           if b[λ,t] > 0 && a[λ,t] > 0
               C += b[λ,t]^2/a[λ,t]
               for μ in 1:7
                   ∂C[μ] += 2*b[λ,t]/a[λ,t] * ∂b[μ,λ,t] -
                            (b[λ,t]/a[λ,t])^2 * ∂a[μ,λ,t]
               end
           end
       end
   end

   return C, ∂C
end

function cost_func_deriv_wrt_orb(dat::PacomeData{T,3}, orb::Vector{T},
                                 ker::Kernel{T};
                                 kwds...) where {T<:AbstractFloat}

   @assert length(orb) == 7
   return cost_func_deriv_wrt_orb(dat, arr2orb(orb), ker; kwds...)
end


"""
    cost_func_deriv_wrt_orb_fd(dat, orb, ker; λ) -> fcost

same as the function `cost_func_deriv_wrt_orb` but the derivatives are
estimated via finite differences (longer).

"""
function cost_func_deriv_wrt_orb_fd(dat::PacomeData{T,3}, orb::Orbit{T},
                                    ker::Kernel{T};
                                    λ::Int=0,
                                    cal::Bool=false) where {T<:AbstractFloat}

   ∂C_∂μ_fd, = jacobian(central_fdm(4,1),
                        x->Pacome.cost_func(dat, Pacome.arr2orb(x), ker;
                                            cal=cal, λ=λ),
                        Pacome.orb2arr(orb))
   return ∂C_∂μ_fd
end

"""
    snr(dat, orb[, ker]; cal)

computes the SNR of an orbit with parameters `orb` in PACO data `dat`.
Optional argument `ker` is to specify the interpolation kernel (a linear spline
by default).

"""
function snr(dat::PacomeData{T,3}, orb::Orbit{T},
             ker::Kernel{T}; kwds...) where {T<:AbstractFloat}
    return sqrt(cost_func(dat, orb, ker; kwds...))
end

"""
    snr_monoepoch(dat, orb, ker)

computes the monoepoch SNRs of an orbit with parameters `orb` in PACO data `dat`
with inteprolation kernel `ker`.

"""
function snr_monoepoch(dat::PacomeData{T,3}, orb::Orbit{T},
                       ker::Kernel{T}; kwds...) where {T<:AbstractFloat}
    a, b = Pacome.interpolate(dat, orb, ker; kwds...)
    snrs = Array{T,ndims(a)}(undef, size(a))
    for λ in 1:size(a,1)
        for t in 1:size(a,2)
            a[λ,t] > 0 ? snrs[λ,t] = b[λ,t] / sqrt(a[λ,t]) : snrs[λ,t] = NaN
        end
    end
    return snrs
end

function snr_monoepoch(dat::PacomeData{T,3}, orb::Vector{T},
                       ker::Kernel{T}; kwds...) where {T<:AbstractFloat}
   @assert length(orb) == 7
   return snr_monoepoch(dat, arr2orb(orb), ker; kwds...)
end

"""
    optimize_orb_param(dat, orb, ker; λ, cal, verb) -> x

computes the optimal orbital parameters `x` by maximising the PACOME cost
function on data `dat` with initial guess `orb` and interpolation kernel `ker`.
The optimization method is `vmlmb` and the gradient of the criterion is computed
numerically via finite differences.

Keyword `λ` specifies whether the orbital parameters should be optimized with
respect to all spectral channels (`λ=0`) or with respect to only on specific
channel (`λ=n where n ∈ [1,nλ]`). Default is `0`.
Keyword `cal` encodes whether the data is calibrated or not. Default is `false`.
Keyword `verb` encodes the verbosity of vmlmb function's execution. Default is
`false`.

"""
function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Array{T,1},
                            ker::Kernel{T,4};
                            kwds...) where {T<:AbstractFloat}

   return optimize_orb_param(dat, arr2orb(orb), ker; kwds...)
end

function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Array{Array{T,1},1},
                            ker::Kernel{T,4};
                            kwds...) where {T<:AbstractFloat}

   orb = [arr2orb(orb[i]) for i in 1:length(orb)]
   return optimize_orb_param(dat, orb, ker; kwds...)
end

function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Array{T,1},
                            ker::Kernel{T,4},
                            grid::Grid{T};
                            kwds...) where {T<:AbstractFloat}

   lb = [grid.a[1], grid.e[1], grid.i[1], grid.τ[1], grid.ω[1],
         grid.Ω[1], grid.K[1]]
   ub = [grid.a[end], grid.e[end], grid.i[end], grid.τ[end], grid.ω[end],
         grid.Ω[end], grid.K[end]]
   return optimize_orb_param(dat, arr2orb(orb), ker, (lb,ub); kwds...)
end

function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Array{Array{T,1},1},
                            ker::Kernel{T,4},
                            grid::Grid{T};
                            kwds...) where {T<:AbstractFloat}

   orb = [arr2orb(orb[i]) for i in 1:length(orb)]
   lb = [grid.a[1], grid.e[1], grid.i[1], grid.τ[1], grid.ω[1],
         grid.Ω[1], grid.K[1]]
   ub = [grid.a[end], grid.e[end], grid.i[end], grid.τ[end], grid.ω[end],
         grid.Ω[end], grid.K[end]]
   return optimize_orb_param(dat, orb, ker, (lb, ub); kwds...)
end

function optimize_orb_param(dat::PacomeData{T,3},
                            orbs::Array{Orbit{T},1},
                            ker::Kernel{T,4};
                            λ::Int=-1,
                            kwds...) where {T<:AbstractFloat}

   @assert length(orbs) == dat.dims[end]
   @assert -1 ≤ λ ≤ dat.dims[end]
   xs = Array{T,2}(undef, (7, dat.dims[end]))
   Cs = Array{T,1}(undef, dat.dims[end])

   if λ == -1
       for λi in 1:dat.dims[end]
           xs[:,λi], Cs[λi] = optimize_orb_param(dat,orbs[λi],ker;λ=λi,kwds...)
       end
   else
       for λi in 1:dat.dims[end]
           xs[:,λi], Cs[λi] = optimize_orb_param(dat,orbs[λi],ker;λ=λ,kwds...)
       end
   end

   return xs, C
end

function optimize_orb_param(dat::PacomeData{T,3},
                            orbs::Array{Orbit{T},1},
                            ker::Kernel{T,4},
                            bounds::Tuple{Vector{T},Vector{T}};
                            λ::Int=-1,
                            kwds...) where {T<:AbstractFloat}

   @assert length(orbs) == dat.dims[end]
   @assert -1 ≤ λ ≤ dat.dims[end]
   xs = Array{T,2}(undef, (7, dat.dims[end]))
   Cs = Array{T,1}(undef, dat.dims[end])

   if λ == -1
       for λi in 1:dat.dims[end]
           xs[:,λi], Cs[λi] = optimize_orb_param(dat, orbs[λi], ker, bounds;
                                                 λ=λi, kwds...)
       end
   else
       for λi in 1:dat.dims[end]
           xs[:,λi], Cs[λi] = optimize_orb_param(dat, orbs[λi], ker, bounds;
                                                 λ=λ,kwds...)
       end
   end

   return xs, C
end

function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Orbit{T},
                            ker::Kernel{T,4};
                            λ::Int=0,
                            cal::Bool=false,
                            verb::Bool=false,
                            hard::Bool=true,
                            maxeval::Int=typemax(Int)) where {T<:AbstractFloat}

    nt = length(dat)
    nλ = dat.dims[end]
    @assert 0 ≤ λ ≤ nλ

    x0 = orb2arr(orb)
    x0[4] = mod(x0[4],1)

    lb = [0., 0., -Inf, 0, -Inf, -Inf, 0.]
    ub = [+Inf, 0.999999, +Inf, +Inf, +Inf, +Inf, +Inf]

    function fg!(x::Vector{T}, ∂C::Vector{T})
        for μ in 1:7
            (x[μ] < lb[μ] || x[μ] > ub[μ]) && return +Inf
        end

        C, ∂C[:] = cost_func_deriv_wrt_orb(dat, arr2orb(x), ker; λ=λ)
        ∂C[:] = -∂C[:]

        return -C
    end


    if hard
        x = vmlmb(fg!, x0; verb=verb, lower=lb, upper=ub, maxeval=maxeval,
                  xtol = (0.0,0.0), ftol = (0.0,0.0), gtol = (0.0,0.0))
    else
        x = vmlmb(fg!, x0; verb=verb, lower=lb, upper=ub, maxeval=maxeval,
                  xtol = (0.0,1e-9), ftol = (0.0,1e-10), gtol = (0.0,1e-8))
    end

    return x, cost_func(dat, x, ker)
end


function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Vector{T},
                            ker::Kernel{T,4},
                            bounds::Tuple{Vector{T},Vector{T}};
                            kwds...) where {T<:AbstractFloat}

   return optimize_orb_param(dat, arr2orb(orb), ker, bounds; kwds...)
end
function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Orbit{T},
                            ker::Kernel{T,4},
                            bounds::Tuple{Vector{T},Vector{T}};
                            λ::Int=0,
                            cal::Bool=false,
                            verb::Bool=false,
                            hard::Bool=true,
                            maxeval::Int=typemax(Int)) where {T<:AbstractFloat}

    nt = length(dat)
    nλ = dat.dims[end]
    @assert 0 ≤ λ ≤ nλ

    x0 = orb2arr(orb)
    x0[4] = mod(x0[4],1)

    # lb = [0., 0., -Inf, 0, -Inf, -Inf, 0.]
    # ub = [+Inf, 0.999999, +Inf, +Inf, +Inf, +Inf, +Inf]
    lb, ub = bounds

    function fg!(x::Vector{T}, ∂C::Vector{T})
        for μ in 1:7
            (x[μ] < lb[μ] || x[μ] > ub[μ]) && return +Inf
        end

        # println("e = $(x[2])")
        C, ∂C[:] = cost_func_deriv_wrt_orb(dat, arr2orb(x), ker; λ=λ)
        ∂C[:] = -∂C[:]

        return -C
    end

    #println("ici ! maxeval=$maxeval")
    if hard
        x = vmlmb(fg!, x0; verb=verb, lower=lb, upper=ub, maxeval=maxeval,
                  xtol = (0.0,0.0), ftol = (0.0,0.0), gtol = (0.0,0.0))
    else
        x = vmlmb(fg!, x0; verb=verb, lower=lb, upper=ub, maxeval=maxeval,
                  xtol = (0.0,1e-9), ftol = (0.0,1e-10), gtol = (0.0,1e-8))
    end

    return x, cost_func(dat, x, ker)
end

function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Orbit{T},
                            ker::Kernel{T,4},
                            pts::Array{Int,2},
                            ROI::Int; kwds...) where {T<:AbstractFloat}
   return optimize_orb_param(dat, orb2arr(orb), ker, pts, ROI; kwds...)
end

function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Orbit{T},
                            ker::Kernel{T,4},
                            bounds::Tuple{Vector{T},Vector{T}},
                            pts::Array{Int,2},
                            ROI::Int; kwds...) where {T<:AbstractFloat}
   return optimize_orb_param(dat, orb2arr(orb), ker, bounds, pts, ROI; kwds...)
end

function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Orbit{T},
                            ker::Kernel{T,4},
                            grid::Grid{T},
                            pts::Array{Int,2},
                            ROI::Int; kwds...) where {T<:AbstractFloat}
    lb = [grid.a[1], grid.e[1], grid.i[1], grid.τ[1], grid.ω[1],
          grid.Ω[1], grid.K[1]]
    ub = [grid.a[end], grid.e[end], grid.i[end], grid.τ[end], grid.ω[end],
          grid.Ω[end], grid.K[end]]
    return optimize_orb_param(dat, orb2arr(orb), ker, (lb,ub), pts, ROI; kwds...)
end

function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Vector{T},
                            ker::Kernel{T,4},
                            pts::Array{Int,2},
                            ROI::Int;
                            λ::Int=0,
                            cal::Bool=false,
                            verb::Bool=false,
                            hard::Bool=true,
                            maxeval::Int=typemax(Int)) where {T<:AbstractFloat}

    nt = length(dat)
    nλ = dat.dims[end]
    @assert 0 ≤ λ ≤ nλ

    lb = [0., 0., -Inf, 0, -Inf, -Inf, 0.]
    ub = [+Inf, 0.999999, +Inf, +Inf, +Inf, +Inf, +Inf]

    fx(x::Vector{T}) where {T<:AbstractFloat} = -cost_func(dat, x, ker,
                                                           pts, ROI,
                                                           lb, ub;
                                                           cal=cal, λ=λ)

    function fg!(x, ∂C)
        for μ in 1:7
            (x[μ] < lb[μ] || x[μ] > ub[μ]) && return +Inf
        end

        _, ∂C[:] = cost_func_deriv_wrt_orb(dat, arr2orb(x), ker; λ=λ)
        ∂C[:] = -∂C[:]

        return fx(x)
    end


    if hard
        x = vmlmb(fg!, orb; verb=verb, lower=lb, upper=ub, maxeval=maxeval,
                  xtol = (0.0,0.0), ftol = (0.0,0.0), gtol = (0.0,0.0))
    else
        x = vmlmb(fg!, orb; verb=verb, lower=lb, upper=ub, maxeval=maxeval,
                  xtol = (0.0,1e-9), ftol = (0.0,1e-10), gtol = (0.0,1e-8))
    end

    return x, cost_func(dat, x, ker)
end

function optimize_orb_param(dat::PacomeData{T,3},
                            orb::Vector{T},
                            ker::Kernel{T,4},
                            bounds::Tuple{Vector{T},Vector{T}},
                            pts::Array{Int,2},
                            ROI::Int;
                            λ::Int=0,
                            cal::Bool=false,
                            verb::Bool=false,
                            hard::Bool=true,
                            maxeval::Int=typemax(Int)) where {T<:AbstractFloat}

    nt = length(dat)
    nλ = dat.dims[end]
    @assert 0 ≤ λ ≤ nλ

    lb, ub = bounds

    fx(x::Vector{T}) where {T<:AbstractFloat} = -cost_func(dat, x, ker,
                                                           pts, ROI,
                                                           lb, ub;
                                                           cal=cal, λ=λ)

    function fg!(x, ∂C)
        for μ in 1:7
            (x[μ] < lb[μ] || x[μ] > ub[μ]) && return +Inf
        end

        _, ∂C[:] = cost_func_deriv_wrt_orb(dat, arr2orb(x), ker; λ=λ)
        ∂C[:] = -∂C[:]

        return fx(x)
    end


    if hard
        x = vmlmb(fg!, orb; verb=verb, lower=lb, upper=ub, maxeval=maxeval,
                  xtol = (0.0,0.0), ftol = (0.0,0.0), gtol = (0.0,0.0))
    else
        x = vmlmb(fg!, orb; verb=verb, lower=lb, upper=ub, maxeval=maxeval,
                  xtol = (0.0,1e-9), ftol = (0.0,1e-10), gtol = (0.0,1e-8))
    end

    return x, cost_func(dat, x, ker)
end

"""
    RMSD_orbs(dat, orb1, orb2, ker) -> RMSD

returns the root-mean-square distance `RMSD` of the projected positions given by
`orb1` relative to the projected positions of `orb2` for epochs of PACO data
`dat` using the interpolation kernel `ker`. The result is expressed in pixels.
The orbits `orb1` and `orb2` can either be `Orbit{T}` or `Vector{T}`.

    RMSD_orbs(dat, pts, orb, ker) -> RMSD

same as above but the projected positions given by the "reference" orbit are
a vector of points `pts`. It avoids computing the reference projection several
times if used in a loop.

"""
function RMSD_orbs(dat::PacomeData{T,N},
                   orb1::Orbit{T},
                   orb2::Orbit{T}) where {T<:AbstractFloat, N}

    RMSD = 0
    nt = length(dat)
    for t in 1:nt
        pt1 = Point(projected_position(orb1, dat.epochs[t]))/dat.pixres[t]
        pt2 = Point(projected_position(orb2, dat.epochs[t]))/dat.pixres[t]
        RMSD += (pt1.x-pt2.x)^2 + (pt1.y-pt2.y)^2
    end

    return sqrt(RMSD/nt)
end

function RMSD_orbs(dat::PacomeData{T,N},
                   pts::Vector{Point{T}},
                   orb::Orbit{T}) where {T<:AbstractFloat, N}

    nt = length(dat)
    @assert nt == length(pts)
    RMSD = 0
    for t in 1:nt
        ΔRA, ΔDec = projected_position(orb, dat.epochs[t])
        pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]
        RMSD += (pts[t].x-pt.x)^2 + (pts[t].y-pt.y)^2
    end
    return sqrt(RMSD/nt)
end

function RMSD_orbs(dat::PacomeData{T,N},
                   orb1::Array{T,1},
                   orb2::Array{T,1}) where {T<:AbstractFloat, N}

    return RMSD_orbs(dat, arr2orb(orb1), arr2orb(orb2))
end

function RMSD_orbs(dat::PacomeData{T,N},
                   pts::Vector{Point{T}},
                   orb::Array{T,1}) where {T<:AbstractFloat, N}

    return RMSD_orbs(dat, pts, arr2orb(orb))
end

function RMSD_orbs_to_center(dat::PacomeData{T,N},
                             orb::Orbit{T}) where {T<:AbstractFloat, N}

    nt = length(dat)
    RMSD = 0
    for t in 1:nt
        ΔRA, ΔDec = projected_position(orb, dat.epochs[t])
        RMSD += (ΔRA/dat.pixres[t])^2 + (ΔDec/dat.pixres[t])^2
    end
    return sqrt(RMSD/nt)
end

function RMSD_orbs_to_center(dat::PacomeData{T,N},
                             orb::Vector{T}) where {T<:AbstractFloat, N}

    @assert length(orb) == 7
    return RMSD_orbs_to_center(dat, arr2orb(orb))
end

function build_pacome_header(dat::PacomeData{T,N},
                             ker::Kernel{T},
                             path_mmap::String,
                             cal::Bool,
                             λ::Int,
                             nthreads::Int,
                             time_elapsed::T,
                             fcost_lim::T,
                             n_expl_orbs::Int) where {T<:AbstractFloat, N}

   dt = last(dat.dims)==2 ? "ADI" : "ASDI"
   if dat.dims[1] == dat.dims[2] == 1448
           instr = "IRDIS"
   elseif dat.dims[1] == dat.dims[2] == 410
           instr = "IFS"
   end

   inter = split((string(ker)),"{")
   first(inter)=="BSpline" ? inter = inter[1]*inter[2][1] : inter = first(inter)

   s = open(path_mmap)
   nrow, norbits = read(s, Int), read(s, Int)
   close(s)

   hdr = FitsHeader("DAT_TYPE"      => (dt,
                                        "Type of the used data (ASDI or ADI)"),
                    "INSTRU"        => (instr,
                                        "Name of the instrument (IRDIS, IFS)"),
                    "N_EPOCHS"      => (length(dat),
                                        "Number of epochs used for computation"),
                    "CAL_DAT"      => (cal,
                                        "Calibrated data or not"),
                    "SPEC_CHA"      => (λ,
                                        "Spectral channel used for computation"*
                                        " (0 = all)"),
                    "INTERP"        => (string(inter),
                                        "Interpolator's name"),
                    "NTHREADS"      => (nthreads,
                                        "Number of threads used for computation"),
                    "EX_TIM_S"      => (time_elapsed,
                                        "Computation time (seconds)"),
                    "EX_TIM"        => (formatTimeSeconds(time_elapsed),
                                        "Computation time (hh:mm:ss)"),
                    "CFUNC_L"       => (fcost_lim,
                                        "Cost func. lim. above which orbits"*
                                        " are saved"),
                    "N_E_ORBS"      => (n_expl_orbs,
                                        "Number of explored orbits"),
                    "N_S_ORBS"      => (norbits,
                                        "Number of saved orbits"),
                    "DATE"          => (string(Dates.now()),
                                        "End of computation's date"),
                    )

   return hdr
end

end # module
