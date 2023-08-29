module Display

using InterpolationKernels
using TwoDimensional
using BenchmarkTools
using Printf
using EasyFITS
using LinearAlgebra
using PyPlot
using Dates
using Glob
using Interpolations

using Pacome
using Utils

function partial_year(period::Type{<:Period}, float::AbstractFloat)
    _year, Δ = divrem(float, 1)
    year_start = DateTime(_year)
    year = period((year_start + Year(1)) - year_start)
    partial = period(round(Dates.value(year) * Δ))
    year_start + partial
end
partial_year(float::AbstractFloat) = partial_year(Nanosecond, float)

function get_sub_cmap(cmap::String, minval::Float64=0., maxval::Float64=1.,
                      N::Int=100)

    cmap = plt.get_cmap(cmap)
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "trunc($N,$minval,$maxval)",
        cmap(LinRange(minval, maxval, N)))
    return new_cmap #arr = reshape(LinRange(0,50,100),(10,10))
end

function MidPointLogNorm(c::String, vmin::T, vcenter::T, vmax::T,
                         linthresh::T) where {T<:AbstractFloat}
    @assert vmin ≥ 0. && linthresh > 0.
    ncmap = ColorMap(c)

    x = [vmin, linthresh, log10(vcenter), log10(vmax)]
    y = [0., 1/(vmax-vmin), 0.5, 1.]
    itp = LinearInterpolation(x, y)
    y_val = itp.(LinRange(vmin, log10(vmax), 1000))
    c_val = Matrix{Float64}(undef, (length(y_val),4))
    for i in 1:length(y_val)
        c_val[i,:] .= ncmap(y_val[i])
    end
    new_cmap = ColorMap("new_"*c, c_val, 256, 1.0)
    return new_cmap
end

"""
   plot_projected_orbit(orb; savePath)

plots the orbit `orb` projected on sky plane detector.
Optional argument `savePath` is a `String` specifying the path where the plot
should be saved. Default is `none`.

"""
function plot_projected_orbit(orb::Vector{T}; kwds...) where {T<:AbstractFloat}
    @assert length(orb) == 7
    return plot_projected_orbit(Pacome.arr2orb(orb); kwds...)
end

function plot_projected_orbit(orb::Pacome.Orbit{T};
                              savePath::String="none",
                              transp::Bool=false) where {T<:AbstractFloat}

    Np = Int(ceil(orb.P*10))
    ΔRA = Array{T}(undef, Np)
    ΔDec = Array{T}(undef, Np)
    time = LinRange{T}(0, orb.P, Np)

    for t in 1:Np
       ΔRA[t], ΔDec[t] = Pacome.projected_position(orb, time[t])
    end

    fig = figure(figsize=(6,6))

    # Orbit projected onto the sky plane
    plot(ΔRA,
        ΔDec,
        linewidth = 2,
        color = "red",
        alpha = 0.8,
        zorder=1)

    # Central star
    scatter([0],
           [0],
           marker="+",
           color="black",
           linewidth=2,
           s=500,
           zorder=2)

    gca().invert_xaxis()
    xlabel("ΔRA [mas]", fontsize=15)
    ylabel("ΔDec [mas]", fontsize=15)
    xticks(fontsize=12)
    yticks(fontsize=12)
    axis("equal")
    tight_layout()

    display(fig)

    if isdir(dirname(savePath))
       fig.savefig(savePath, transparent=transp)
    elseif savePath != "none"
       error("Not a directory.")
    end
end

function plot_projected_orbit(orbs::Array{T,2};
                              savePath::String="none",
                              transp::Bool=false) where {T<:AbstractFloat}

    @assert size(orbs,1) == 7
    Norb = size(orbs,2)

    fig = figure(figsize=(6,6))

    # Central star
    scatter([0],
           [0],
           marker="+",
           color="black",
           linewidth=2,
           s=500,
           zorder=2)

    for n in 1:Norb
        orb = Pacome.arr2orb(orbs[:,n])

        Np = Int(ceil(orb.P*10))
        ΔRA = Array{T}(undef, Np)
        ΔDec = Array{T}(undef, Np)
        time = LinRange{T}(0, orb.P, Np)

        for t in 1:Np
           ΔRA[t], ΔDec[t] = Pacome.projected_position(orb, time[t])
        end

        # Orbit projected onto the sky plane
        plot(ΔRA,
            ΔDec,
            linewidth = 1,
            color = "darkorange",
            alpha = 0.1,
            zorder=1)
    end

    gca().invert_xaxis()
    xlabel("ΔRA [mas]", fontsize=15)
    ylabel("ΔDec [mas]", fontsize=15)
    xticks(fontsize=12)
    yticks(fontsize=12)
    axis("equal")
    tight_layout()

    display(fig)

    if isdir(dirname(savePath))
       fig.savefig(savePath, transparent=transp)
    elseif savePath != "none"
       error("Not a directory.")
    end
end

"""
   plot_projected_positions(epochs, orb; pixres, savePath) -> nothing

plots the orbit `orb` projected on the sky plane detector and the positions
along the orbit at epochs `epochs`.
Optional argument `savePath` is a `String` specifying the path where the plot
should be saved. Default is `none`. The argument `pixres` specifies the pixel
resolution (in milli-arc-second) used to plot the pixellic grid under the orbit.
Default is `12.25` pix/mas. If `pixres=0`, no grid is displayed. Argument
`maskrad` is to specify the radius (in mas) of the largest coronagraphic mask of
`data`.

---
   plot_projected_positions(dat, orb; pixres, savePath) -> nothing

plots the orbit `orb` projected on the sky plane detector and the positions
along the orbit at the epochs given by data `dat`.
Optional argument `savePath` is a `String` specifying the path where the plot
is saved. Default is `none`. The argument `pixres` specifies the pixel
resolution (in milli-arc-second) used to plot the pixellic grid under the orbit.
Default is `12.25` pix/mas. If `gridon=false`, no grid is displayed. Argument
`maskrad` is to specify the radius (in mas) of the largest coronagraphic mask of
`data`.

"""
function plot_projected_positions(epochs::Array{T},
                                  orb::Pacome.Orbit{T};
                                  pixres::T=12.25,
                                  gridon::Bool=false,
                                  maskrad::T=0.,
                                  savePath::String="none") where {T<:AbstractFloat}

    Ne = length(epochs)
    Np = Int(ceil(orb.P*10))

    ΔRA = Array{T}(undef, Np)
    ΔDec = Array{T}(undef, Np)
    ΔRA_obj = Array{T}(undef, Ne)
    ΔDec_obj = Array{T}(undef, Ne)
    time = LinRange{T}(0, orb.P, Np)

    for t in 1:Np
       ΔRA[t], ΔDec[t] = Pacome.projected_position(orb, time[t])
    end

    for t in 1:Ne
       ΔRA_obj[t], ΔDec_obj[t] = Pacome.projected_position(orb, epochs[t])
    end

    fig, ax = subplots(figsize=(6,6))

    # Orbit projected onto the sky plane
    plot(ΔRA,
        ΔDec,
        linewidth = 2.5,
        color = "red",
        alpha = 0.75,
        zorder=2,
        label="Projected orbit")

    # Central star
    scatter([0], [0], marker="*", color="black", linewidth=1, s=500, zorder=3,
            facecolor="darkorange",
            edgecolor="black")

    # Orbital positions of the planet at the different epochs
    scatter(ΔRA_obj,
           ΔDec_obj,
           marker="o",
           facecolor="dodgerblue",
           edgecolor="black",
           s=35,#75,
           zorder=4,
           label="Projected positions "*"\$\\mathbb{\\theta}_t(\\mathbb{\\mu})\$")

     if pixres != 0
       ls, a, z, c, lw = "-", 0.6, 1, "gray", 0.5
       ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
       ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
       if gridon
           ax.xaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                   color=c, linewidth=lw)
           ax.yaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                   color=c, linewidth=lw)
           ax.axhline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
           ax.axvline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
       end
       ax.tick_params(which="minor", bottom=false, left=false)
       secax = ax.secondary_xaxis("top", functions=(x->x/pixres, x->x*pixres))
       secax.set_xlabel("ΔRA [pix]", fontsize=15)
       secay = ax.secondary_yaxis("right", functions=(x->x/pixres, x->x*pixres))
       secay.set_ylabel("ΔDec [pix]", fontsize=15)
       secax.tick_params(labelsize=12)
       secay.tick_params(labelsize=12)
    end

    if maskrad > 0
        mask = matplotlib.patches.Circle((0, 0), maskrad, color="grey",
                                         alpha=0.4, zorder=0, label="Coronagraphic mask")
      ax.add_patch(mask)
    end

    gca().invert_xaxis()
    xlabel("ΔRA [mas]", fontsize=15)
    ylabel("ΔDec [mas]", fontsize=15)
    xticks(fontsize=12)
    yticks(fontsize=12)
    axis("equal")
    tight_layout()

    legend(fontsize=11)
    display(fig)

    if isdir(dirname(savePath))
       fig.savefig(savePath)
    elseif savePath != "none"
       error("Not a directory.")
    end
end

function plot_projected_positions(epochs::Array{T},
                                  orbs::Array{Pacome.Orbit{T},1};
                                  pixres::T=12.25,
                                  gridon::Bool=false,
                                  maskrad::T=0.,
                                  savePath::String="none") where {T<:AbstractFloat}


    fig, ax = subplots(figsize=(6,6))
    Ne = length(epochs)
    col = ["black"]

    for n in 1:length(orbs)
        Np = Int(ceil(orbs[n].P*10))
        ΔRA = Array{T}(undef, Np)
        ΔDec = Array{T}(undef, Np)
        ΔRA_obj = Array{T}(undef, Ne)
        ΔDec_obj = Array{T}(undef, Ne)
        time = LinRange{T}(0,orbs[n].P, Np)

        for t in 1:Np
            try
                ΔRA[t], ΔDec[t] = Pacome.projected_position(orbs[n], time[t])
            catch e
                ΔRA[t], ΔDec[t] = NaN, NaN
            end
        end

        for t in 1:Ne
            try
                ΔRA_obj[t], ΔDec_obj[t] = Pacome.projected_position(orbs[n], epochs[t])
            catch e
                ΔRA_obj[t], ΔDec_obj[t] = NaN, NaN
            end
        end

        # Orbit projected onto the sky plane
        if n==1
            plot(ΔRA,
                 ΔDec,
                 linewidth = 1.5,#2.5,
                 color = col[1], #col[n],
                 alpha = 0.75,
                 zorder=2,
                 label="Projected orbits")
        else
            plot(ΔRA,
                 ΔDec,
                 linewidth = 1.5,#2.5,
                 color = col[1], #col[n],
                 alpha = 0.75,
                 zorder=2)
        end
        # Central star
        if n == 1
            scatter([0], [0], marker="*", color="black", linewidth=1, s=500,
                        zorder=3, facecolor="darkorange", edgecolor="black")
        end

        # Orbital positions of the planet at the different epochs
        if n==1
            scatter(ΔRA_obj,
                    ΔDec_obj,
                    marker="o",
                    facecolor="dodgerblue",
                    edgecolor="black",
                    s= 35,#75,
                    zorder=4,
                    label="Projected positions "*L"\theta_t(\mu)")
        else
            scatter(ΔRA_obj,
                    ΔDec_obj,
                    marker="o",
                    facecolor="dodgerblue",
                    edgecolor="black",
                    s= 35,#75,
                    zorder=4)
        end
    end

    if pixres != 0
      ls, a, z, c, lw = "-", 0.6, 1, "gray", 0.5
      ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
      ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
      if gridon
          ax.xaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                  color=c, linewidth=lw)
          ax.yaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                  color=c, linewidth=lw)
          ax.axhline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
          ax.axvline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
      end
      ax.tick_params(which="minor", bottom=false, left=false)
      secax = ax.secondary_xaxis("top", functions=(x->x/pixres, x->x*pixres))
      secax.set_xlabel("ΔRA [pix]", fontsize=15)
      secay = ax.secondary_yaxis("right", functions=(x->x/pixres, x->x*pixres))
      secay.set_ylabel("ΔDec [pix]", fontsize=15)
      secax.tick_params(labelsize=12)
      secay.tick_params(labelsize=12)
    end

    if maskrad > 0
      mask = matplotlib.patches.Circle((0, 0), maskrad, color="grey",
                                       alpha=0.6, zorder=0,
                                       label="Coronagraphic mask")
      ax.add_patch(mask)
    end

    gca().invert_xaxis()
    xlabel("ΔRA [mas]", fontsize=15)
    ylabel("ΔDec [mas]", fontsize=15)
    xticks(fontsize=12)
    yticks(fontsize=12)
    axis("equal")
    tight_layout()
    legend(fontsize=12)

    display(fig)

    if isdir(dirname(savePath))
       fig.savefig(savePath)
    elseif savePath != "none"
       error("Not a directory.")
    end
end

function plot_projected_positions(epochs::Array{T},
                                  orbs0::Array{Pacome.Orbit{T},1},
                                  orbs::Array{Pacome.Orbit{T},1};
                                  pixres::T=12.25,
                                  gridon::Bool=false,
                                  maskrad::T=0.,
                                  savePath::String="none") where {T<:AbstractFloat}


    fig, ax = subplots(figsize=(6,6))
    Ne = length(epochs)
    Ns = length(orbs0)

    col = ["black"]

    ΔRA_obj = Array{T,2}(undef, (Ne,Ns))
    ΔDec_obj = Array{T,2}(undef, (Ne,Ns))

    for s in 1:Ns
        Np = Int(ceil(orbs0[s].P*10))
        ΔRA = Array{T}(undef, Np)
        ΔDec = Array{T}(undef, Np)
        time = LinRange{T}(0,orbs0[s].P, Np)
        for t in 1:Np
            try
                ΔRA[t], ΔDec[t] = Pacome.projected_position(orbs0[s], time[t])
            catch e
                ΔRA[t], ΔDec[t] = NaN, NaN
            end
        end
        if s==1
            plot(ΔRA,
                 ΔDec,
                 linewidth = 1.75,#2.5,
                 color = "red", #col[n],
                 alpha = 0.75,
                 zorder=1000,
                 label="Optimal orbits")
        else
            plot(ΔRA,
                 ΔDec,
                 linewidth = 1.75,#2.5,
                 color = "red", #col[n],
                 alpha = 0.75,
                 zorder=1000)
        end

        for t in 1:Ne
            try
                ΔRA_obj[t,s], ΔDec_obj[t,s] = Pacome.projected_position(orbs0[s], epochs[t])
            catch e
                ΔRA_obj[t,s], ΔDec_obj[t,s] = NaN, NaN
            end
        end
        # Orbital positions of the planet at the different epochs
        if s==1
            scatter(ΔRA_obj[:,s],
                    ΔDec_obj[:,s],
                    marker="o",
                    facecolor="dodgerblue",
                    edgecolor="black",
                    s= 35,#75,
                    zorder=1001,
                    label="Projected positions "*L"\theta_t(\mu)")
        else
            scatter(ΔRA_obj[:,s],
                    ΔDec_obj[:,s],
                    marker="o",
                    facecolor="dodgerblue",
                    edgecolor="black",
                    s= 35,#75,
                    zorder=1001)
        end
    end

    for n in 1:length(orbs)
        Np = Int(ceil(orbs[n].P*10))
        ΔRA = Array{T}(undef, Np)
        ΔDec = Array{T}(undef, Np)
        time = LinRange{T}(0,orbs[n].P, Np)

        for t in 1:Np
            try
                ΔRA[t], ΔDec[t] = Pacome.projected_position(orbs[n], time[t])
            catch e
                ΔRA[t], ΔDec[t] = NaN, NaN
            end
        end

        # Orbit projected onto the sky plane
        if n==1
            plot(ΔRA,
                 ΔDec,
                 linewidth = 1.75,#2.5,
                 color = "black", #col[n],
                 alpha = 0.15,
                 zorder=2,
                 label="Best other orbits")
        else
            plot(ΔRA,
                 ΔDec,
                 linewidth = 1,#2.5,
                 color = "black", #col[n],
                 alpha = 0.15,
                 zorder=2)
        end
        # Central star
        if n == 1
            scatter([0], [0], marker="*", color="black", linewidth=1, s=500,
                        zorder=3, facecolor="darkorange", edgecolor="black")
        end
    end

    if pixres != 0
      ls, a, z, c, lw = "-", 0.6, 1, "gray", 0.5
      ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
      ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
      if gridon
          ax.xaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                  color=c, linewidth=lw)
          ax.yaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                  color=c, linewidth=lw)
          ax.axhline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
          ax.axvline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
      end
      ax.tick_params(which="minor", bottom=false, left=false)
      secax = ax.secondary_xaxis("top", functions=(x->x/pixres, x->x*pixres))
      secax.set_xlabel("ΔRA [pix]", fontsize=15)
      secay = ax.secondary_yaxis("right", functions=(x->x/pixres, x->x*pixres))
      secay.set_ylabel("ΔDec [pix]", fontsize=15)
      secax.tick_params(labelsize=12)
      secay.tick_params(labelsize=12)
    end

    if maskrad > 0
      mask = matplotlib.patches.Circle((0, 0), maskrad, color="grey",
                                       alpha=0.6, zorder=0,
                                       label="Coronagraphic mask")
      ax.add_patch(mask)
    end

    gca().invert_xaxis()
    xlabel("ΔRA [mas]", fontsize=15)
    ylabel("ΔDec [mas]", fontsize=15)
    xticks(fontsize=12)
    yticks(fontsize=12)
    axis("equal")
    tight_layout()
    legend(fontsize=12)

    display(fig)

    if isdir(dirname(savePath))
       fig.savefig(savePath)
    elseif savePath != "none"
       error("Not a directory.")
    end
end

function plot_projected_positions(dat::Pacome.PacomeData{T,3},
                                  orb::Pacome.Orbit{T};
                                  kwds...) where {T<:AbstractFloat}

    mr = maximum([Utils.SPHERE_COR_DIAM[cor] for cor in dat.icors])/2
    plot_projected_positions(dat.epochs, orb; pixres=mean(dat.pixres),
                                              maskrad=mr, kwds...)
end

function plot_projected_positions(dat::Pacome.PacomeData{T,3},
                                  orbs::Array{Pacome.Orbit{T},1};
                                  kwds...) where {T<:AbstractFloat}

    mr = maximum([Utils.SPHERE_COR_DIAM[cor] for cor in dat.icors])/2
    plot_projected_positions(dat.epochs, orbs; pixres=mean(dat.pixres),
                                               maskrad=mr, kwds...)
end

function plot_projected_positions(dat::Pacome.PacomeData{T,3},
                                  orbs::Array{T,2};
                                  kwds...) where {T<:AbstractFloat}
    @assert size(orbs,1) == 7
    orbs2 = [Pacome.arr2orb(orbs[:,k]) for k in 1:size(orbs,2)]
    mr = maximum([Utils.SPHERE_COR_DIAM[cor] for cor in dat.icors])/2
    plot_projected_positions(dat.epochs, orbs2; pixres=mean(dat.pixres),
                                                maskrad=mr, kwds...)
end

function plot_projected_positions(dat::Pacome.PacomeData{T,3},
                                  orbs0::Array{T,2},
                                  orbs::Array{T,2};
                                  kwds...) where {T<:AbstractFloat}
    @assert size(orbs,1) == 7
    @assert size(orbs0,1) == 7
    orbs1 = [Pacome.arr2orb(orbs0[:,k]) for k in 1:size(orbs0,2)]
    orbs2 = [Pacome.arr2orb(orbs[:,k]) for k in 1:size(orbs,2)]
    mr = maximum([Utils.SPHERE_COR_DIAM[cor] for cor in dat.icors])/2
    plot_projected_positions(dat.epochs, orbs1, orbs2; pixres=mean(dat.pixres),
                                                maskrad=mr, kwds...)
end

"""
   individual_snrs(dat, orb, λ, ROI; scale, fsize, savePath) -> nothing

plots all individual signal-to-noise ratios maps at the positions computed for
orbit `orb` on data `dat` at wavelength `λ` and around a region of interest of
radius `ROI`.

Optional argument `scale` specifies whether the colorbars should be the same for
all epochs (default is `false`), `fsize` is a tuple specifying the size of the
figure (default is `(16.5,8.5)`) and `savePath` is a `String` specifying the
path where the plot is saved (default is `none`).

---
   plot_individual_snrs(dat, orb, ROI; kwds...) -> nothing

same as `plot_individual_snrs(dat, orb, λ, ROI; scale, fsize, savePath)` but for
all wavelength.

"""

function plot_individual_snrs(dat::Pacome.PacomeData{T,3},
                              orb::Pacome.Orbit{T},
                              ROI::Int;
                              λ::Int=1,
                              scale::Bool=false,
                              fsize::Tuple=(16.5,8.5),
                              nrow::Int=-1,
                              ncol::Int=-1,
                              savePath::String="none",
                              showSNR::Bool=false,
                              showmas::Bool=false,
                              fontweight::String="normal",
                              digs::Int=1,
                              ker::Kernel{T}=CubicSpline{T}(-0.6,-0.004),
                              α_bbox::Real=0.,
                              c::String="coolwarm") where {T<:AbstractFloat}

   @assert λ ∈ 1:dat.dims[3]
   @assert ROI*2+1 ≤ dat.dims[2]
   @assert ROI*2+1 ≤ dat.dims[1]

   nx, ny = ROI*2+1, ROI*2+1

   snr_mono = Pacome.snr_monoepoch(dat, orb, ker)

   snr_maps = fill!(Array{T,3}(undef, (length(dat), ROI*2+1, ROI*2+1)), 0)
   pos = Array{T,2}(undef, (length(dat),2))
   centers = Array{T,2}(undef, (length(dat),2))
   pos_source = Array{T,2}(undef, (length(dat),2))

   for t in 1:length(dat)
      ΔRA, ΔDec = Pacome.projected_position(orb, dat.epochs[t])
      pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]
      pos[t,:] .= pt.x, pt.y

      i, j = round.(Int, pos[t,:])
      centers[t,:] .= dat.centers[t].x - i + ROI, dat.centers[t].y - j + ROI
      pos_source[t,:] .= pt.x - i + ROI, pt.y - j + ROI

      a = dat.a[t][i-ROI:i+ROI,j-ROI:j+ROI,λ]
      b = dat.b[t][i-ROI:i+ROI,j-ROI:j+ROI,λ]

      idx = findall(x -> x > 0, a)

      snr_maps[t,idx] = b[idx] ./ sqrt.(a[idx])

      snr_maps[t,:,:] = transpose(snr_maps[t,:,:])
   end

   min_snr = minimum(snr_maps)
   max_snr = maximum(snr_maps)

   if ncol == ncol == -1
       nrow = round(Int, sqrt(length(dat)))
       ncol, rest = divrem(length(dat),nrow)
       ncol = (rest==0 ? ncol : ncol+1)
   end

   wspace=0.02
   hspace=0.02

   fig = figure(figsize=fsize)

   for t in 1:length(dat)
      ax = subplot(nrow, ncol, t)
      if scale
         imshow(snr_maps[t,:,:], origin="lower",
                cmap=c)
      else
         imshow(snr_maps[t,:,:], origin="lower",
             norm=matplotlib.colors.TwoSlopeNorm(vmin=minimum(snr_maps[t,:,:]),
                                                 vcenter=0,
                                                 vmax=maximum(snr_maps[t,:,:])),
             cmap=c)
      end

      if occursin("DB",dat.iflts[t])
          temp = split(dat.iflts[t],"_")[end]
          band = temp[1] * temp[λ+1]
      elseif occursin("BB",dat.iflts[t])
          band = replace(dat.iflts[t], "_" => " ")
      end

      date_Ymd = Dates.format(partial_year(dat.epochs[t]), "YYYY-mm-dd")
      text(nx*0.025, ny*0.95, date_Ymd*"\n$band", fontsize = 14,
           color = "black", fontweight=fontweight,
           bbox=Dict("boxstyle"=>"round,pad=0.3", "fc"=>"white", "lw"=>0, "alpha"=>α_bbox),
           verticalalignment="top", ha="left")

      if showSNR
          text(nx*0.025, ny*0.05,
               "SNR"* #L"(\widehat{\mu})"*
               "=$(round(snr_mono[λ,t],digits=digs))",
               fontsize = 14,
               color = "black", fontweight=fontweight,
               bbox=Dict("boxstyle"=>"round,pad=0.3", "fc"=>"white", "lw"=>0, "alpha"=>α_bbox),
               va="center",
               ha="left")
      end

      if showmas
          # mas_bar_size = nx/6
          mas_bar = nx/5
          mas_x = [nx*0.95-mas_bar, nx*0.95]
          mas_y = [ny*0.025, ny*0.025]

          plot(mas_x, mas_y, linewidth=2, color="black")

          text(mean(mas_x), mean(mas_y)+nx*0.04,
               "$(round(mas_bar*mean(dat.pixres)/1000,digits=digs))"*"''", fontsize = 14,
               color = "black",
               va="center",
               ha="center", fontweight=fontweight)
      end


      wspace=0.02
      hspace=0.02
      cb = colorbar(shrink=1, pad=0.015, aspect=15)
      cb.ax.tick_params(labelsize=13, labelcolor="black")

      scatter(pos_source[t,1], pos_source[t,2],
              marker="+", color="black", s=100)

      limits = ax.axis()

      scatter(centers[t,1], centers[t,2],
              marker="*", color="darkorange", s=200)

      ax.axis(limits)

      ax.tick_params(axis = "both", bottom = false, top = false,
                     right = false, left = false)
      ax.tick_params(labelbottom=false, labelleft=false)
   end

   tight_layout()
   subplots_adjust(wspace=wspace,hspace=hspace)

   display(fig)

   if isdir(dirname(savePath))
      fig.savefig(savePath)
   elseif savePath != "none"
      error("Not a directory.")
   end
end

"""
   plot_PACOME_cost_func(dat, orb, λ, ROI; savePath) -> nothing

plots the PACOME optimal cost function map at the positions
computed for orbit `orb` on data `dat` at wavelength `λ` adjusted to the nearest
pixel around a region of interest of radius `ROI`.

Optional argument `savePath` is a `String` specifying the path where the plot is
saved (default is `none`).

"""

function plot_cost_func(dat::Pacome.PacomeData{T,3},
                               orb::Array{T},
                               ROI::Int; kwds...) where {T<:AbstractFloat}

   return plot_cost_func(dat, Pacome.arr2orb(orb), ROI; kwds...)
end

function plot_cost_func(dat::Pacome.PacomeData{T,3},
                       orb::Pacome.Orbit{T},
                       ROI::Int;
                       cal::Bool=false,
                       λ::Int=1,
                       savePath::String="none",
                       c::String="coolwarm",
                       showmas::Bool=false,
                       showC::Bool=false,
                       fontweight::String="normal",
                       digs::Int=1) where {T<:AbstractFloat}

   @assert ROI*2+1 ≤ dat.dims[2]
   @assert ROI*2+1 ≤ dat.dims[1]
   nλ = dat.dims[end]
   @assert 1 ≤ λ ≤ nλ

   nx, ny = ROI*2+1, ROI*2+1

   a_maps = Array{T,3}(undef, (length(dat), ROI*2+1, ROI*2+1))
   b_maps = Array{T,3}(undef, (length(dat), ROI*2+1, ROI*2+1))
   pos = Array{T,2}(undef, (length(dat),2))

   for t in 1:length(dat)
      ΔRA, ΔDec = Pacome.projected_position(orb, dat.epochs[t])
      pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]
      pos[t,:] .= pt.x, pt.y

      i, j = round.(Int, pos[t,:])
      a_maps[t,:,:] = dat.a[t][i-ROI:i+ROI,j-ROI:j+ROI,λ]
      b_maps[t,:,:] = dat.b[t][i-ROI:i+ROI,j-ROI:j+ROI,λ]
   end

   if cal
       num = reshape(max.(0,sum(b_maps, dims=1)).^2, (ROI*2+1, ROI*2+1))
       den = reshape(sum(a_maps, dims=1), (ROI*2+1, ROI*2+1))
       C_map = reshape(sum(num ./ den, dims=3), (ROI*2+1, ROI*2+1))
   else
       C_map = reshape((sum(max.(0, b_maps).^2 ./ a_maps, dims=1)),
                   (ROI*2+1, ROI*2+1))
   end

   fig = figure(figsize=(6,5))
   ax = fig.add_subplot(111)

   imshow(C_map,
          origin="lower",
          norm=matplotlib.colors.SymLogNorm(vmin=0, vmax=maximum(C_map),
                                            base=10, linscale=1, linthresh=1),
          cmap=c)

   if showC
       text(nx*0.025, ny*0.97, L"\mathcal{C}(\widehat{\mu})="*
            "$(round(maximum(C_map), digits=digs))", fontsize=16,
            color = "crimson",
            bbox=Dict("boxstyle"=>"round,pad=0.3", "fc"=>"white",
            "ec"=>"crimson", "lw"=>1, "alpha"=>0.75),
            verticalalignment="top", ha="left")
   end
   if showmas
       mas_bar = nx/6
       mas_x = [nx*0.95-mas_bar, nx*0.95]
       mas_y = [ny*0.025, ny*0.025]

       plot(mas_x, mas_y, linewidth=2.5, color="black")

       text(mean(mas_x), mean(mas_y)+ny*0.03,
            "$(round(mas_bar*mean(dat.pixres)/1000,digits=digs))"*"''",
            fontsize = 14,
            color = "black",
            va="center",
            ha="center", fontweight=fontweight)
   end

   cb = colorbar(shrink=1)

   cb.ax.set_title(L"$\mathcal{C}_{\ell="*"$λ"*L"}$",fontsize=20)
   cb.ax.tick_params(labelsize=14)

   scatter(ROI, ROI, marker="+", s=100, color="black")
   ax.tick_params(axis = "both", bottom = false, top = false,
                  right = false, left = false)
   ax.tick_params(labelbottom = false)
   ax.tick_params(labelleft = false)
   tight_layout()
   display(fig)

   if isdir(dirname(savePath))
      fig.savefig(savePath)
   elseif savePath != "none"
      error("Not a directory.")
   end
end

function plot_multi_epoch_snr(dat::Pacome.PacomeData{T,3},
                               orb::Array{T},
                               ROI::Int; kwds...) where {T<:AbstractFloat}

   return plot_multi_epoch_snr(dat, Pacome.arr2orb(orb), ROI; kwds...)
end

function plot_multi_epoch_snr(dat::Pacome.PacomeData{T,3},
                               orb::Pacome.Orbit{T},
                               ROI::Int;
                               λ::Int=1,
                               savePath::String="none",
                               c::String="coolwarm",
                               showmas::Bool=false,
                               showSNR::Bool=false,
                               fontweight::String="normal",
                               digs::Int=1) where {T<:AbstractFloat}

   @assert ROI*2+1 ≤ dat.dims[2]
   @assert ROI*2+1 ≤ dat.dims[1]
   nλ = dat.dims[end]
   @assert 1 ≤ λ ≤ nλ

   nx, ny = ROI*2+1, ROI*2+1

   a_maps = Array{T,3}(undef, (length(dat), ROI*2+1, ROI*2+1))
   b_maps = Array{T,3}(undef, (length(dat), ROI*2+1, ROI*2+1))
   pos = Array{T,2}(undef, (length(dat),2))

   for t in 1:length(dat)
      ΔRA, ΔDec = Pacome.projected_position(orb, dat.epochs[t])
      pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]
      pos[t,:] .= pt.x, pt.y

      i, j = round.(Int, pos[t,:])
      a_maps[t,:,:] = dat.a[t][i-ROI:i+ROI,j-ROI:j+ROI,λ]
      b_maps[t,:,:] = dat.b[t][i-ROI:i+ROI,j-ROI:j+ROI,λ]
   end

   SNR_map = reshape(sqrt.((sum(max.(0, b_maps).^2 ./ a_maps, dims=1))),
                   (ROI*2+1, ROI*2+1))

   fig = figure(figsize=(6,5))
   ax = fig.add_subplot(111)

   imshow(SNR_map,
          origin="lower",
          norm=matplotlib.colors.SymLogNorm(vmin=0, vmax=maximum(SNR_map),
                                            base=10, linscale=1, linthresh=1),
          cmap=c)

   if showSNR
       text(nx*0.025, ny*0.97, L"\mathcal{S}/\mathcal{N}(\widehat{\mu})="*
            "$(round(maximum(SNR_map), digits=digs))", fontsize=16,
            color = "crimson",
            bbox=Dict("boxstyle"=>"round,pad=0.3", "fc"=>"white",
            "ec"=>"crimson", "lw"=>1, "alpha"=>0.75),
            verticalalignment="top", ha="left")
   end
   if showmas
       mas_bar = nx/6
       mas_x = [nx*0.95-mas_bar, nx*0.95]
       mas_y = [ny*0.025, ny*0.025]

       plot(mas_x, mas_y, linewidth=2.5, color="black")

       text(mean(mas_x), mean(mas_y)+ny*0.03,
            "$(round(mas_bar*mean(dat.pixres)/1000,digits=digs))"*"''",
            fontsize = 14,
            color = "black",
            va="center",
            ha="center", fontweight=fontweight)
   end

   cb = colorbar(shrink=1)
   cb.ax.set_title(L"$\mathcal{S}/\mathcal{N}_{\ell="*"$λ"*L"}$",fontsize=20)
   cb.ax.tick_params(labelsize=14)

   scatter(ROI, ROI, marker="+", s=100, color="black")
   ax.tick_params(axis = "both", bottom = false, top = false,
                  right = false, left = false)
   ax.tick_params(labelbottom = false)
   ax.tick_params(labelleft = false)
   tight_layout()
   display(fig)

   if isdir(dirname(savePath))
      fig.savefig(savePath)
   elseif savePath != "none"
      error("Not a directory.")
   end
end

"""
   plot_cost_func_oversampled(dat, orb, ker, rad;
                                     λ, s, c, savePath) -> nothing

plots the PACOME optimal cost function map at the positions
computed for orbit `orb` on data `dat` at wavelength `λ` with interpolation
kernel ker`.

The 2-D map is resampled by a factor `s` (default is `s=4`) and the colormap `c`
of the plot is tunable. Optional argument  `savePath` is a `String` specifying
the path where the plot is saved (default is `none`).

"""

function plot_cost_func_oversampled(dat::Pacome.PacomeData{T,3},
                               orb::Array{T}, ker::Kernel{T},
                               rad::Int; kwds...) where {T<:AbstractFloat}

   return plot_cost_func_oversampled(dat, Pacome.arr2orb(orb), ker,
                                            rad; kwds...)
end

function plot_cost_func_oversampled(dat::Pacome.PacomeData{T,3},
                               orb::Pacome.Orbit{T},
                               ker::Kernel{T},
                               rad::Int;
                               λ::Int=0,
                               cal::Bool=false,
                               s::Int=4,
                               C_lim::T=-1.,
                               vmax::T=-1.,
                               scale::String="linear",
                               showmas::Bool=false,
                               showC::Bool=false,
                               fontweight::String="normal",
                               c::String="coolwarm",
                               cont::Bool=false,
                               savePath::String="none",
                               digs::Int=1,
                               transp::Bool=false) where {T<:AbstractFloat}

   @assert rad*2+1 ≤ dat.dims[2]
   @assert rad*2+1 ≤ dat.dims[1]
   nλ = dat.dims[end]
   @assert 0 ≤ λ ≤ nλ
   λ == 0 ? (λs = 1:nλ) : λs = [λ]

   npix = rad*2*s
   nx, ny = npix, npix

   a_maps = Array{T,4}(undef, (length(dat), length(λs), npix, npix))
   b_maps = Array{T,4}(undef, (length(dat), length(λs), npix, npix))
   pos = Array{T,2}(undef, (length(dat),2))

   for t in 1:length(dat)
      ΔRA, ΔDec = Pacome.projected_position(orb, dat.epochs[t])
      pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]

      for (xi,x) in enumerate(LinRange(pt.x-rad, pt.x+rad, npix))
          for (yi,y) in enumerate(LinRange(pt.y-rad, pt.y+rad, npix))
              for k in λs
                  a_maps[t,k,xi,yi] = Pacome.interpolate(dat.a[t], ker,
                                                         Point(x,y), k)
                  b_maps[t,k,xi,yi] = Pacome.interpolate(dat.b[t], ker,
                                                         Point(x,y), k)
              end
          end
      end
   end

   C_map = fill!(Array{T,2}(undef, (npix, npix)),0)
   for t in 1:length(dat)
       for i in 1:npix
           for j in 1:npix
               for k in λs
                   if b_maps[t,k,i,j] > 0 && a_maps[t,k,i,j] > 0
                       C_map[i,j] += b_maps[t,k,i,j]^2 / a_maps[t,k,i,j]
                   end
               end
           end
       end
   end

   C_lim == -1. ? vcenter = (maximum(C_map)-minimum(C_map))/2 : vcenter = C_lim
   vmax == -1. ? vmax = maximum(C_map) : nothing

   fig, ax  = subplots(figsize=(6,5))

   if scale == "log"
       ncmap = MidPointLogNorm(c, 0., vcenter, vmax, 1.)
       im = imshow(C_map,
                   origin="lower",
                   norm=matplotlib.colors.SymLogNorm(vmin=0,
                                                     vmax=vmax,
                                                     base=10, linscale=1,
                                                     linthresh=1),
                                                     cmap=ncmap)
    elseif scale == "linear"
        imshow(C_map,
               origin="lower",
               norm=matplotlib.colors.TwoSlopeNorm(vmin=0,
                                                   vcenter=vcenter,
                                                   vmax=vmax),
               cmap=c)
    end

   cb = colorbar(shrink=1)
   cb.ax.tick_params(labelsize=14)

   if showC
       C = Pacome.cost_func(dat, orb, ker; cal=cal, λ=λ)
       text(nx*0.025, ny*0.97, L"\mathcal{C}(\widehat{\mu})="*
            "$(round(C, digits=digs))", fontsize=16,
            color = "crimson",
            bbox=Dict("boxstyle"=>"round,pad=0.3", "fc"=>"white",
            "ec"=>"crimson", "lw"=>1, "alpha"=>0.75),
            verticalalignment="top", ha="left")
   end
   if showmas
       # mas_bar_size = nx/6
       mas_bar = nx/6
       mas_x = [nx*0.95-mas_bar, nx*0.95]
       mas_y = [ny*0.025, ny*0.025]

       plot(mas_x, mas_y, linewidth=2.5, color="black")

       text(mean(mas_x), mean(mas_y)+ny*0.03,
            "$(round((mas_bar-1)/s*mean(dat.pixres)/1000,digits=digs))"*"''",
            fontsize = 14,
            color = "black",
            va="center",
            ha="center", fontweight=fontweight)
   end

   if cont
       contours = ax.contour(C_map, origin="lower",
                             norm=matplotlib.colors.PowerNorm(gamma=.5),
                             cmap="Greys",
                             levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.90]*
                                                               maximum(C_map))
       cb.add_lines(contours)
   else
       ax.scatter(npix/2-0.5, npix/2-0.5, marker="+", s=100, color="black")
   end

   #
   ax.tick_params(axis = "both", bottom = false, top = false, right = false,
                  left = false, labelbottom = false, labelleft = false)
   ax.tick_params()
   ax.tick_params()
   tight_layout()
   display(fig)

   if isdir(dirname(savePath))
      fig.savefig(savePath, transparent=transp)
   elseif savePath != "none"
      error("Not a directory.")
   end
end

"""
   demoOrbits(orb, date_obs) -> nothing

plots an interactive tool to visualize the orbit `orb` projected on the detector
at date `data_obs`.

---
   demoOrbits(orb, pt, date_obs) -> nothing

plots an interactive tool to visualize the orbit `orb` projected on the detector
at date `data_obs`. It also plots an additional point `pt` on the figure.

---
   demoOrbits(orb) -> nothing

plots an interactive tool to visualize the orbit `orb` projected on the detector
at the execution's exact date.

---
   demoOrbits(orb, pt) -> nothing

plots an interactive tool to visualize the orbit `orb` projected on the detector
at the execution's exact date. It also plots an additional point `pt` on the
figure.

"""
function demoOrbits(orb::Pacome.Orbit{T}, date_obs::T) where {T<:AbstractFloat}

   nb_pts = 250
   epochs = LinRange(0,orb.P,nb_pts)

   X = Array{T,1}(undef, nb_pts)
   Y = Array{T,1}(undef, nb_pts)
   for t in 1:nb_pts
      X[t], Y[t] = Pacome.projected_position(orb, epochs[t])
   end
   X_at_date, Y_at_date = Pacome.projected_position(orb, date_obs)

   fig, ax = subplots(figsize=(8,8))

   subplots_adjust(top=0.98,
                   bottom=0.385,
                   left=0.125,
                   right=0.98,
                   hspace=0.0,
                   wspace=0.0)

   frame, = plot(X, Y, color="dodgerblue", lw=2)
   frame_pt, = plot(X_at_date, Y_at_date, marker="o", color="dodgerblue", markersize=10)
   scatter([0], [0], marker="+", color="red", s=60)
   xlabel("ΔRA", fontsize=14)
   ylabel("ΔDec", fontsize=14)
   xticks(fontsize=13)
   yticks(fontsize=13)

   gca().invert_xaxis()
   axis("equal")

   ax.margins(x=0)
   axcolor = "white"

   ax_a  = plt.axes([0.25, 0.26, 0.65, 0.03], facecolor=axcolor)
   ax_e  = plt.axes([0.25, 0.22, 0.65, 0.03], facecolor=axcolor)
   ax_i  = plt.axes([0.25, 0.18, 0.65, 0.03], facecolor=axcolor)
   ax_τ = plt.axes([0.25, 0.14, 0.65, 0.03], facecolor=axcolor)
   ax_ω  = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
   ax_Ω  = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
   ax_K  = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor=axcolor)

   s_a = matplotlib.widgets.Slider(ax_a, "a", 10, 2500, valinit=orb.a)
   s_e = matplotlib.widgets.Slider(ax_e, "e", 0, 0.9999999, valinit=orb.e)
   s_i = matplotlib.widgets.Slider(ax_i, "i", 0, 180, valinit=orb.i)
   s_τ = matplotlib.widgets.Slider(ax_τ, "τ", 0, 1, valinit=orb.τ)
   s_ω = matplotlib.widgets.Slider(ax_ω, "ω", 0, 360, valinit=orb.ω)
   s_Ω = matplotlib.widgets.Slider(ax_Ω, "Ω", 0, 360, valinit=orb.Ω)
   s_K = matplotlib.widgets.Slider(ax_K, "K", orb.K*0.5, orb.K*1.5, valinit=orb.K)

   function update(val::Real)
       new_a  = s_a.val
       new_e  = s_e.val
       new_i  = s_i.val
       new_τ = s_τ.val
       new_ω  = s_ω.val
       new_Ω  = s_Ω.val
       new_K  = s_K.val

       new_orb = Pacome.Orbit{T}(a  = new_a,   e = new_e, i = new_i,
                                 τ = new_τ,  ω = new_ω, Ω = new_Ω,
                                 K  = new_K)
       new_epochs  = LinRange(0,new_orb.P,nb_pts)

       for t in 1:nb_pts
          X[t], Y[t] = Pacome.projected_position(new_orb, new_epochs[t])
       end
       X_at_date, Y_at_date = Pacome.projected_position(new_orb, date_obs)

       frame.set_xdata(X)
       frame.set_ydata(Y)
       frame_pt.set_xdata(X_at_date)
       frame_pt.set_ydata(Y_at_date)
       fig.canvas.draw_idle()
   end

   s_a.on_changed(update)
   s_e.on_changed(update)
   s_i.on_changed(update)
   s_τ.on_changed(update)
   s_ω.on_changed(update)
   s_Ω.on_changed(update)
   s_K.on_changed(update)

end

function demoOrbits(orb::Pacome.Orbit{T}) where {T<:AbstractFloat}
    return demoOrbits(orb, Dates.today())
end

function demoOrbits(orb::Pacome.Orbit{T},
                    date_obs::Date) where {T<:AbstractFloat}
    return demoOrbits(orb, Utils.yearfraction(date_obs))
end

function demoOrbits(orb::Pacome.Orbit{T},
                    date_obs::String) where {T<:AbstractFloat}
    return demoOrbits(orb, yearfraction(date_obs))
end

function demoOrbits(orb::Pacome.Orbit{T},
                    pt::Point{T},
                    date_obs::T) where {T<:AbstractFloat}

   nb_pts = 250
   epochs = LinRange(0,orb.P,nb_pts)

   X = Array{T,1}(undef, nb_pts)
   Y = Array{T,1}(undef, nb_pts)
   for t in 1:nb_pts
      X[t], Y[t] = Pacome.projected_position(orb, epochs[t])
   end
   X_at_date, Y_at_date = Pacome.projected_position(orb, date_obs)

   fig, ax = subplots(figsize=(8,8))

   subplots_adjust(top=0.98,
                   bottom=0.385,
                   left=0.125,
                   right=0.98,
                   hspace=0.0,
                   wspace=0.0)

   frame, = plot(X, Y, color="dodgerblue", lw=2)
   frame_pt, = plot(X_at_date, Y_at_date, marker="o", color="dodgerblue",
                    markersize=10)
   scatter([0], [0], marker="+", color="red", s=60)
   scatter([pt.x], [pt.y], marker="s", color="black", s=40, alpha=0.75)
   xlabel("ΔRA", fontsize=14)
   ylabel("ΔDec", fontsize=14)
   xticks(fontsize=13)
   yticks(fontsize=13)

   gca().invert_xaxis()
   axis("equal")

   ax.margins(x=0)
   axcolor = "white"

   ax_a  = plt.axes([0.25, 0.26, 0.65, 0.03], facecolor=axcolor)
   ax_e  = plt.axes([0.25, 0.22, 0.65, 0.03], facecolor=axcolor)
   ax_i  = plt.axes([0.25, 0.18, 0.65, 0.03], facecolor=axcolor)
   ax_τ = plt.axes([0.25, 0.14, 0.65, 0.03], facecolor=axcolor)
   ax_ω  = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
   ax_Ω  = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
   ax_K  = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor=axcolor)

   s_a = matplotlib.widgets.Slider(ax_a, "a", 10, 2500, valinit=orb.a)
   s_e = matplotlib.widgets.Slider(ax_e, "e", 0, 0.9999999, valinit=orb.e)
   s_i = matplotlib.widgets.Slider(ax_i, "i", 0, 180, valinit=orb.i)
   s_τ = matplotlib.widgets.Slider(ax_τ, "τ", 0, 1, valinit=orb.τ)
   s_ω = matplotlib.widgets.Slider(ax_ω, "ω", 0, 360, valinit=orb.ω)
   s_Ω = matplotlib.widgets.Slider(ax_Ω, "Ω", 0, 360, valinit=orb.Ω)
   s_K = matplotlib.widgets.Slider(ax_K, "K", orb.K*0.5, orb.K*1.5, valinit=orb.K)

   function update(val::Real)
       new_a  = s_a.val
       new_e  = s_e.val
       new_i  = s_i.val
       new_τ = s_τ.val
       new_ω  = s_ω.val
       new_Ω  = s_Ω.val
       new_K  = s_K.val

       new_orb = Pacome.Orbit{T}(a  = new_a,   e = new_e, i = new_i,
                                 τ = new_τ,  ω = new_ω, Ω = new_Ω,
                                 K  = new_K)
       new_epochs  = LinRange(0,new_orb.P,nb_pts)

       for t in 1:nb_pts
          X[t], Y[t] = Pacome.projected_position(new_orb, new_epochs[t])
       end
       X_at_date, Y_at_date = Pacome.projected_position(new_orb, date_obs)


       frame.set_xdata(X)
       frame.set_ydata(Y)
       frame_pt.set_xdata(X_at_date)
       frame_pt.set_ydata(Y_at_date)
       fig.canvas.draw_idle()
   end

   s_a.on_changed(update)
   s_e.on_changed(update)
   s_i.on_changed(update)
   s_τ.on_changed(update)
   s_ω.on_changed(update)
   s_Ω.on_changed(update)
   s_P.on_changed(update)

end

function demoOrbits(orb::Pacome.Orbit{T},
                    pt::Point{T}) where {T<:AbstractFloat}
    return demoOrbits(orb, pt, Dates.today())
end

function demoOrbits(orb::Pacome.Orbit{T}, pt::Point{T},
                    date_obs::Date) where {T<:AbstractFloat}
    return demoOrbits(orb, pt, yearfraction(date_obs))
end

function demoOrbits(orb::Pacome.Orbit{T}, pt::Point{T},
                    date_obs::String) where {T<:AbstractFloat}
    return demoOrbits(orb, pt, yearfraction(date_obs))
end

"""
   plot_projOrbitAndErrors(dat, orb, all_orb; legend_loc, otherLabel, savePath)

plots the optimal orbit `orb` projected on the detector for data `dat` along
with all orbits contained in `all_orb` (which should be of size 7xN where N is
the number of orbits).

Optional argument `legend_loc` is a `String` specifying where the legend of the
plot should be located (default is `"upper right"`), `err_esti` is the method
used for the error estimation (can either be `"empirical"` or `"analytical"`)
and `savePath` is a `String` specifying the path where the plot is saved
(default is `none`).

"""
function plot_projOrbitAndErrors(dat::Pacome.PacomeData{T,3},
                                 orb::Pacome.Orbit{T},
                                 all_orb::AbstractArray{T,2};
                                 pixres::T=12.25,
                                 gridon::Bool=false,
                                 mask::Bool=false,
                                 transp::Bool=false,
                                 legend_loc="upper right",
                                 otherLabel::String="empirical",
                                 frameon::Bool=true,
                                 framealpha::T=0.8,
                                 labelcolor::String="None",
                                 savePath::String="none") where {T<:AbstractFloat}

    Ne = length(dat)
    Np = Int(ceil(orb.P*10))

    ΔRA = Array{T}(undef, Np)
    ΔDec = Array{T}(undef, Np)
    ΔRA_obj = Array{T}(undef, Ne)
    ΔDec_obj = Array{T}(undef, Ne)
    time = LinRange{T}(0,orb.P, Np)

    for t in 1:Np
       ΔRA[t], ΔDec[t] = Pacome.projected_position(orb, time[t])
    end

    for t in 1:Ne
       ΔRA_obj[t], ΔDec_obj[t] = Pacome.projected_position(orb, dat.epochs[t])
    end

    fig, ax = subplots(figsize=(6,6))

    plot(ΔRA,
        ΔDec,
        linewidth = 2.25,
        color = "red",
        alpha = 1.,
        zorder=3,
        label="Optimal orbit")

    # Orbit projected onto the sky plane
    plot(ΔRA,
        ΔDec,
        linewidth = 2.25,
        color = "darkorange",
        alpha = 1.,
        zorder=2,
        label="Best other orbits")# ("*L"$d_\mu \leq 1$"*" pix)")



    # Central star
    scatter([0], [0], marker="*", color="black", linewidth=1, s=500, zorder=4,
            facecolor="darkorange",
            edgecolor="black")

    # All other orbits projected onto the sky plane
    for k in 1:size(all_orb)[2]
       orb_k = Pacome.arr2orb(all_orb[:,k])
       Np = Int(ceil(orb_k.P*10))
       time = LinRange{T}(0,orb_k.P, Np)
       ΔRA = Array{T}(undef, Np)
       ΔDec = Array{T}(undef, Np)

       for t in 1:Np
          # println("k = $(k), t = $(time[t])")
          ΔRA[t], ΔDec[t] = Pacome.projected_position(orb_k, time[t])
       end

       if k == 1 && otherLabel=="empirical"
          plot(ΔRA,
             ΔDec,
             linewidth = 0.75,
             color = "darkorange",
             alpha = 0.6,
             zorder=3)#,
             #label="Perturbed orbits")
       elseif k == 1 && otherLabel=="analytical"
          plot(ΔRA,
             ΔDec,
             linewidth = 0.75,
             color = "darkorange",
             alpha = 0.6,
             zorder=3,
             label="Orbits within error bars")

       else
          plot(ΔRA,
             ΔDec,
             linewidth = 0.6,
             color = "darkorange",
             alpha = 0.2,
             zorder=2)
       end
    end

    # Orbital positions of the planet at the different epochs
    scatter(ΔRA_obj,
           ΔDec_obj,
           marker="o",
           facecolor="dodgerblue",
           edgecolor="black",
           s=35,
           zorder=5,
           label="Projected positions")

     if pixres != 0
       ls, a, z, c, lw = "-", 0.6, 1, "gray", 0.5
       ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
       ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
       if gridon
           ax.xaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                   color=c, linewidth=lw)
           ax.yaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                   color=c, linewidth=lw)
           ax.axhline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
           ax.axvline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
       end
       ax.tick_params(which="minor", bottom=false, left=false)
       secax = ax.secondary_xaxis("top", functions=(x->x/pixres, x->x*pixres))
       secax.set_xlabel("ΔRA [pix]", fontsize=15)
       secay = ax.secondary_yaxis("right", functions=(x->x/pixres, x->x*pixres))
       secay.set_ylabel("ΔDec [pix]", fontsize=15)
       secax.tick_params(labelsize=12)
       secay.tick_params(labelsize=12)
    end

    if mask
        mr = maximum([Utils.SPHERE_COR_DIAM[cor] for cor in dat.icors])/2
        maskobj = matplotlib.patches.Circle((0, 0), mr, color="grey",
                                         alpha=0.4, zorder=0)
      ax.add_patch(maskobj)
    end

    gca().invert_xaxis()
    xlabel("ΔRA [mas]", fontsize=15)
    ylabel("ΔDec [mas]", fontsize=15)
    xticks(fontsize=12)
    yticks(fontsize=12)
    axis("equal")
    legend(loc=legend_loc, labelcolor=labelcolor,
           frameon=frameon, framealpha=framealpha, fontsize=11)
    tight_layout()

    display(fig)

    if isdir(dirname(savePath))
       fig.savefig(savePath, transparent=transp)
    elseif savePath != "none"
       error("Not a directory.")
    end
end

function plot_projOrbitAndErrors(dat::Pacome.PacomeData{T,3},
                                 orb::Pacome.Orbit{T},
                                 C::T,
                                 all_orb::AbstractArray{T,2},
                                 all_C::Vector{T};
                                 pixres::T=12.25,
                                 gridon::Bool=false,
                                 mask::Bool=false,
                                 transp::Bool=false,
                                 legend_loc="upper right",
                                 otherLabel::String="empirical",
                                 frameon::Bool=true,
                                 framealpha::T=0.8,
                                 cb_ticks_perso::Bool=false,
                                 labelcolor::String="None",
                                 savePath::String="none") where {T<:AbstractFloat}

    Ne = length(dat)
    Np = Int(ceil(orb.P*10))

    ΔRA = Array{T}(undef, Np)
    ΔDec = Array{T}(undef, Np)
    ΔRA_obj = Array{T}(undef, Ne)
    ΔDec_obj = Array{T}(undef, Ne)
    time = LinRange{T}(0,orb.P, Np)

    for t in 1:Np
       ΔRA[t], ΔDec[t] = Pacome.projected_position(orb, time[t])
    end

    for t in 1:Ne
       ΔRA_obj[t], ΔDec_obj[t] = Pacome.projected_position(orb, dat.epochs[t])
    end

    # fig, ax = subplots(figsize=(6,6))
    fig, ax = subplots(figsize=(6.5,6))

    vmin, vmax = minimum(all_C), max(C,maximum(all_C))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    cmap = get_sub_cmap("YlOrRd",0.25,0.7,100)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plot(ΔRA,
        ΔDec,
        linewidth = 2.25,
        color = sm.to_rgba(C),
        alpha = 1.,
        zorder=3,
        label="Optimal orbit")

    # Orbit projected onto the sky plane
    plot(ΔRA,
        ΔDec,
        linewidth = 2.25,
        color = sm.to_rgba((vmax+vmin)/2),
        alpha = 1.,
        zorder=2,
        label="Best other orbits")# ("*L"$d_\mu \leq 1$"*" pix)")

    # Central star
    scatter([0], [0], marker="*", color="black", linewidth=1, s=500, zorder=4,
            facecolor="darkorange",
            edgecolor="black")

    # All other orbits projected onto the sky plane
    for k in 1:size(all_orb)[2]
       orb_k = Pacome.arr2orb(all_orb[:,k])
       Np = Int(ceil(orb_k.P*10))
       time = LinRange{T}(0,orb_k.P, Np)
       ΔRA = Array{T}(undef, Np)
       ΔDec = Array{T}(undef, Np)

       for t in 1:Np
          # println("k = $(k), t = $(time[t])")
          ΔRA[t], ΔDec[t] = Pacome.projected_position(orb_k, time[t])
       end

       if k == 1 && otherLabel=="empirical"
          plot(ΔRA,
             ΔDec,
             linewidth = 1.5, #0.75,
             color = "darkorange",
             alpha = 0.2,
             zorder=3)#,
             #label="Perturbed orbits")
       elseif k == 1 && otherLabel=="analytical"
          plot(ΔRA,
             ΔDec,
             linewidth = 1.5,#0.75,
             color = "darkorange",
             alpha = 0.2,
             zorder=3,
             label="Orbits within error bars")

       else
          plot(ΔRA,
             ΔDec,
             linewidth = 1.5,#0.6,
             color = sm.to_rgba(all_C[k]),#"darkorange",
             alpha = 0.2,
             zorder=2)
       end
    end

    cbar = colorbar(sm, pad=0.15)
    if cb_ticks_perso
        max_pow = floor(Int,log10(vmax))
        # cbar_ticks = Vector(0.05:0.15:0.95) .* (vmax-vmin) .+ vmin
        cbar_ticks = round.(cbar.get_ticks(), digits=1)
        cbar_ticks_formatted = round.(cbar_ticks ./ 10^max_pow, digits=2)
        cbar_labs = split.(string.(cbar_ticks_formatted), ".")
        cbar_ndig = maximum(length.([last(elem) for elem in cbar_labs]))
        cbar_labs = [first(elem)*"."*rpad(last(elem), cbar_ndig, "0")
                     for elem in cbar_labs]
        # cbar_ticks = LinRange(vmin, vmax, 5+2)[2:end-1]
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_labs)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_title("\$ \\mathcal{C} (\\times 10^{$max_pow}) \$",fontsize=16)
    else
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_title("\$ \\mathcal{C} \$",fontsize=18)
    end

    # Orbital positions of the planet at the different epochs
    scatter(ΔRA_obj,
           ΔDec_obj,
           marker="o",
           facecolor="dodgerblue",
           edgecolor="black",
           s=35,
           zorder=5,
           label="Projected positions")

     if pixres != 0
       ls, a, z, c, lw = "-", 0.6, 1, "gray", 0.5
       ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
       ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
       if gridon
           ax.xaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                   color=c, linewidth=lw)
           ax.yaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                   color=c, linewidth=lw)
           ax.axhline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
           ax.axvline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
       end
       ax.tick_params(which="minor", bottom=false, left=false)
       secax = ax.secondary_xaxis("top", functions=(x->x/pixres, x->x*pixres))
       secax.set_xlabel("ΔRA [pix]", fontsize=15)
       secay = ax.secondary_yaxis("right", functions=(x->x/pixres, x->x*pixres))
       secay.set_ylabel("ΔDec [pix]", fontsize=15, rotation=-90)
       secax.tick_params(labelsize=12)
       secay.tick_params(labelsize=12)
    end

    if mask
        mr = maximum([Utils.SPHERE_COR_DIAM[cor] for cor in dat.icors])/2
        maskobj = matplotlib.patches.Circle((0, 0), mr, color="grey",
                                         alpha=0.4, zorder=0)
      ax.add_patch(maskobj)
    end

    gca().invert_xaxis()
    xlabel("ΔRA [mas]", fontsize=15)
    ylabel("ΔDec [mas]", fontsize=15)
    xticks(fontsize=12)
    yticks(fontsize=12)
    axis("equal")
    legend(loc=legend_loc, labelcolor=labelcolor,
           frameon=frameon, framealpha=framealpha, fontsize=11)
    tight_layout()

    display(fig)

    if isdir(dirname(savePath))
       fig.savefig(savePath, transparent=transp)
    elseif savePath != "none"
       error("Not a directory.")
    end
end

"""
   plot_all_projected_orbits(dat, orb, all_orb; pixres, legend_loc, savePath)

plots the optimal orbit `orb` projected on the detector for data `dat` along
with all orbits contained in `all_orb` (which should be of size 7xN where N is
the number of orbits).

Optional argument `legend_loc` is a `String` specifying where the legend of the
plot should be located (default is `"upper right"`) and `savePath` is a `String`
specifying the path where the plot is saved (default is `none`). The argument
`pixres` specifies the pixel resolution (in milli-arc-second) used to plot the
pixellic grid under the orbit. Default is `12.25` pix/mas. If `pixres=0`, no
grid is displayed.

"""
function plot_all_projected_orbits(dat::Pacome.PacomeData{T,3},
                                   orb::Pacome.Orbit{T},
                                   all_orb::AbstractArray{T,2};
                                   pixres::T=12.25,
                                   gridon::Bool=true,
                                   mask::Bool=true,
                                   legend_loc="upper right",
                                   savePath::String="none") where {T<:AbstractFloat}

    Ne = length(dat)
    Np = Int(ceil(orb.P*10))

    ΔRA = Array{T}(undef, Np)
    ΔDec = Array{T}(undef, Np)
    ΔRA_obj = Array{T}(undef, Ne)
    ΔDec_obj = Array{T}(undef, Ne)
    time = LinRange{T}(0,orb.P, Np)

    for t in 1:Np
       ΔRA[t], ΔDec[t] = Pacome.projected_position(orb, time[t])
    end

    for t in 1:Ne
       ΔRA_obj[t], ΔDec_obj[t] = Pacome.projected_position(orb, dat.epochs[t])
    end

    fig, ax = subplots(figsize=(6,6))

    # Central star
    scatter([0], [0], marker="*", color="black", linewidth=1, s=500, zorder=100,
            facecolor="darkorange",
            edgecolor="black")

    # Plot first orb
    Np = Int(ceil(orb.P*10))
    time = LinRange{T}(0,orb.P, Np)
    ΔRA = Array{T}(undef, Np)
    ΔDec = Array{T}(undef, Np)
    for t in 1:Np
       ΔRA[t], ΔDec[t] = Pacome.projected_position(orb, time[t])
   end
    plot(ΔRA,
         ΔDec,
         linewidth = 4,
         color = "red",
         alpha = 0.75,
         zorder=3,
         label="Optimal projected orbit")

    # All other orbits projected onto the sky plane
    for k in 1:size(all_orb)[2]
       orb_k = Pacome.arr2orb(all_orb[:,k])
       Np = Int(ceil(orb_k.P*10))
       time = LinRange{T}(0,orb_k.P, Np)
       ΔRA = Array{T}(undef, Np)
       ΔDec = Array{T}(undef, Np)

       for t in 1:Np
          # println("k = $(k), t = $(time[t])")
          ΔRA[t], ΔDec[t] = Pacome.projected_position(orb_k, time[t])
       end

       if k == 1
          plot(ΔRA,
             ΔDec,
             linewidth = 0.75,
             color = "darkorange",
             alpha = 0.4,
             zorder=3, label="Other best orbits")
       else
          plot(ΔRA,
             ΔDec,
             linewidth = 0.75,
             color = "darkorange",
             alpha = 0.4,
             zorder=2)
       end
    end

    # Orbital positions of the planet at the different epochs
    scatter(ΔRA_obj,
           ΔDec_obj,
           marker="o",
           facecolor="dodgerblue",
           edgecolor="black",
           s=70,
           zorder=5,
           label="Projected positions "*L"$\theta_t(\mu)$")
           # label="Observations")

     if pixres != 0
        ls, a, z, c, lw = "-", 0.6, 1, "gray", 0.5
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
        if gridon
            ax.xaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                    color=c, linewidth=lw)
            ax.yaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                    color=c, linewidth=lw)
            ax.axhline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
            ax.axvline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
        end
        ax.tick_params(which="minor", bottom=false, left=false)
        secax = ax.secondary_xaxis("top", functions=(x->x/pixres, x->x*pixres))
        secax.set_xlabel("ΔRA [pix]", fontsize=15)
        secay = ax.secondary_yaxis("right", functions=(x->x/pixres, x->x*pixres))
        secay.set_ylabel("ΔDec [pix]", fontsize=15)
        secax.tick_params(labelsize=12)
        secay.tick_params(labelsize=12)
     end

     if mask
         mr = maximum([Utils.SPHERE_COR_DIAM[cor] for cor in dat.icors])/2
         maskobj = matplotlib.patches.Circle((0, 0), mr, color="grey",
                                          alpha=0.4, zorder=10, label="Coronagraphic mask")
       ax.add_patch(maskobj)
     end

    gca().invert_xaxis()
    xlabel("ΔRA [mas]", fontsize=15)
    ylabel("ΔDec [mas]", fontsize=15)
    xticks(fontsize=12)
    yticks(fontsize=12)
    axis("equal")
    legend(loc=legend_loc, fontsize=10.5)
    tight_layout()

    display(fig)

    if isdir(dirname(savePath))
       fig.savefig(savePath)
    elseif savePath != "none"
       error("Not a directory.")
    end
end

function plot_all_projected_orbits(dat::Pacome.PacomeData{T,3},
                                   all_orb::AbstractArray{T,2};
                                   pixres::T=12.25,
                                   gridon::Bool=true,
                                   mask::Bool=true,
                                   legend_loc="upper right",
                                   savePath::String="none") where {T<:AbstractFloat}

    Ne = length(dat)

    fig, ax = subplots(figsize=(6,6))

    # Central star
    scatter([0],
           [0],
           marker="*",
           color="black",
           linewidth=2,
           s=500,
           zorder=4)

    # All orbits projected onto the sky plane
    for k in 1:size(all_orb)[2]
       orb_k = Pacome.arr2orb(all_orb[:,k])
       Np = Int(ceil(orb_k.P*10))
       time = LinRange{T}(0,orb_k.P, Np)
       ΔRA = Array{T}(undef, Np)
       ΔDec = Array{T}(undef, Np)

       for t in 1:Np
          ΔRA[t], ΔDec[t] = Pacome.projected_position(orb_k, time[t])
       end

       plot(ΔRA,
            ΔDec,
            linewidth = 1,
            color = "darkorange",
            alpha = 0.6,
            zorder=2)
    end

    if pixres != 0
      ls, a, z, c, lw = "-", 0.6, 1, "gray", 0.5
      ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
      ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
      if gridon
          ax.xaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                  color=c, linewidth=lw)
          ax.yaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
                  color=c, linewidth=lw)
          ax.axhline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
          ax.axvline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
      end
      ax.tick_params(which="minor", bottom=false, left=false)
      secax = ax.secondary_xaxis("top", functions=(x->x/pixres, x->x*pixres))
      secax.set_xlabel("ΔRA [pix]", fontsize=15)
      secay = ax.secondary_yaxis("right", functions=(x->x/pixres, x->x*pixres))
      secay.set_ylabel("ΔDec [pix]", fontsize=15)
      secax.tick_params(labelsize=12)
      secay.tick_params(labelsize=12)
    end

    if mask
        mr = maximum([Utils.SPHERE_COR_DIAM[cor] for cor in dat.icors])/2
        maskobj = matplotlib.patches.Circle((0, 0), mr, color="grey",
                                         alpha=0.4, zorder=0)
      ax.add_patch(maskobj)
    end

    gca().invert_xaxis()
    xlabel("ΔRA [mas]", fontsize=15)
    ylabel("ΔDec [mas]", fontsize=15)
    xticks(fontsize=12)
    yticks(fontsize=12)
    axis("equal")
    tight_layout()

    display(fig)

    if isdir(dirname(savePath))
       fig.savefig(savePath)
    elseif savePath != "none"
       error("Not a directory.")
    end
end

end # module
