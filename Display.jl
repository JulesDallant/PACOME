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

# rcParams = PyPlot.PyDict(matplotlib."rcParams")
# rcParams["ytick.color"] = "w"
# rcParams["xtick.color"] = "w"
# rcParams["axes.labelcolor"] = "w"
# rcParams["axes.edgecolor"] = "w"

function partial_year(period::Type{<:Period}, float::AbstractFloat)
    _year, Δ = divrem(float, 1)
    year_start = DateTime(_year)
    year = period((year_start + Year(1)) - year_start)
    partial = period(round(Dates.value(year) * Δ))
    year_start + partial
end
partial_year(float::AbstractFloat) = partial_year(Nanosecond, float)

"""
    get_sub_cmap(cmap, minval=0.0, maxval=1.0, n=100)

truncate_colormap
"""
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
   itions(epochs, orb; pixres, savePath) -> nothing

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
    # col = ["orangered", "royalblue", "forestgreen", "mediumorchid"]
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
    # col = ["orangered", "royalblue", "forestgreen", "mediumorchid"]
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

# function plot_projected_positions(epochs::Array{T},
#                                   orbsInj::Array{Pacome.Orbit{T},1},
#                                   orbsFound::Array{Pacome.Orbit{T},1};
#                                   pixres::T=12.25,
#                                   gridon::Bool=false,
#                                   maskrad::T=0.,
#                                   savePath::String="none") where {T<:AbstractFloat}
#
#
#     fig, ax = subplots(figsize=(7,7))
#     Ne = length(epochs)
#     col = ["orangered", "royalblue", "forestgreen", "mediumorchid"]
#
#     for n in 1:length(orbsInj)
#         Np = Int(ceil(orbsInj[n].P*10))
#         Np2 = Int(ceil(orbsInj[n].P*10))
#         ΔRA = Array{T}(undef, Np)
#         ΔDec = Array{T}(undef, Np)
#         ΔRA2 = Array{T}(undef, Np2)
#         ΔDec2 = Array{T}(undef, Np2)
#         ΔRA_obj = Array{T}(undef, Ne)
#         ΔDec_obj = Array{T}(undef, Ne)
#         ΔRA_obj2 = Array{T}(undef, Ne)
#         ΔDec_obj2 = Array{T}(undef, Ne)
#         time = LinRange{T}(0,orbsInj[n].P, Np)
#         time2 = LinRange{T}(0,orbsFound[n].P, Np2)
#
#         for t in 1:Np
#             ΔRA[t], ΔDec[t] = Pacome.projected_position(orbsInj[n], time[t])
#             ΔRA2[t], ΔDec2[t] = Pacome.projected_position(orbsFound[n], time2[t])
#         end
#
#         for t in 1:Ne
#            ΔRA_obj[t], ΔDec_obj[t] = Pacome.projected_position(orbsInj[n], epochs[t])
#            ΔRA_obj2[t], ΔDec_obj2[t] = Pacome.projected_position(orbsFound[n], epochs[t])
#         end
#
#         # Orbit projected onto the sky plane
#         if n==1
#             plot(ΔRA,
#                  ΔDec,
#                  linewidth = 2.5,
#                  color = col[n],
#                  alpha = 0.5,
#                  zorder=2,
#                  label="Injected orbits")
#              plot(ΔRA2,
#                   ΔDec2,
#                   linewidth = 2.5,
#                   linestyle = "--",
#                   color = col[n],
#                   alpha = 0.75,
#                   zorder=2,
#                   label="Found orbits")
#         else
#             plot(ΔRA,
#                  ΔDec,
#                  linewidth = 2.5,
#                  color = col[n],
#                  alpha = 0.5,
#                  zorder=2)
#              plot(ΔRA2,
#                   ΔDec2,
#                   linewidth = 2.5,
#                   linestyle = "--",
#                   color = col[n],
#                   alpha = 0.75,
#                   zorder=2)
#         end
#
#         # Central star
#         if n == 1
#             scatter([0], [0], marker="*", color="black", linewidth=1, s=500,
#                         zorder=3, facecolor="darkorange", edgecolor="black")
#         end
#
#         # Orbital positions of the planet at the different epochs
#         if n==1
#             scatter(ΔRA_obj,
#                     ΔDec_obj,
#                     marker="o",
#                     facecolor=col[n],
#                     s= 20,#75,
#                     zorder=4,
#                     alpha = 0.75,
#                     label="Injected positions")
#             scatter(ΔRA_obj2,
#                     ΔDec_obj2,
#                     marker="D",
#                     facecolor=col[n],
#                     s= 20,#75,
#                     zorder=4,
#                     alpha = 0.75,
#                     label="Found positions")
#         else
#             scatter(ΔRA_obj,
#                     ΔDec_obj,
#                     marker="o",
#                     facecolor=col[n],
#                     s= 20,#75,
#                     alpha = 0.75,
#                     zorder=4)
#             scatter(ΔRA_obj2,
#                     ΔDec_obj2,
#                     marker="D",
#                     facecolor=col[n],
#                     alpha = 0.75,
#                     s= 20,#75,
#                     zorder=4)
#         end
#     end
#
#     if pixres != 0
#       ls, a, z, c, lw = "-", 0.6, 1, "gray", 0.5
#       ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
#       ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(pixres))
#       if gridon
#           ax.xaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
#                   color=c, linewidth=lw)
#           ax.yaxis.grid(true, which="minor", zorder=z, alpha=a, linestyle=ls,
#                   color=c, linewidth=lw)
#           ax.axhline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
#           ax.axvline(0, zorder=z, alpha=a, linestyle=ls, color=c, linewidth=lw)
#       end
#       ax.tick_params(which="minor", bottom=false, left=false)
#       secax = ax.secondary_xaxis("top", functions=(x->x/pixres, x->x*pixres))
#       secax.set_xlabel("ΔRA [pix]", fontsize=15)
#       secay = ax.secondary_yaxis("right", functions=(x->x/pixres, x->x*pixres))
#       secay.set_ylabel("ΔDec [pix]", fontsize=15)
#       secax.tick_params(labelsize=12)
#       secay.tick_params(labelsize=12)
#     end
#
#     if maskrad > 0
#       mask = matplotlib.patches.Circle((0, 0), maskrad, color="grey",
#                                        alpha=0.6, zorder=0)#,
#                                        #label="Coronagraphic mask")
#       ax.add_patch(mask)
#     end
#
#     gca().invert_xaxis()
#     xlabel("ΔRA [mas]", fontsize=15)
#     ylabel("ΔDec [mas]", fontsize=15)
#     xticks(fontsize=12)
#     yticks(fontsize=12)
#     axis("equal")
#     tight_layout()
#     legend(fontsize=11)
#
#     display(fig)
#
#     if isdir(dirname(savePath))
#        fig.savefig(savePath)
#     elseif savePath != "none"
#        error("Not a directory.")
#     end
# end

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

    println("here")
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

function plot_individual_snrs_sameLeg(dat::Pacome.PacomeData{T,3},
                              orb::Pacome.Orbit{T},
                              λ::Int,
                              ROI::Int;
                              scale::Bool=false,
                              fsize::Tuple=(16.5,8.5),
                              savePath::String="none",
                              c::String="bwr") where {T<:AbstractFloat}

   @assert λ ∈ 1:dat.dims[3]
   @assert ROI*2+1 ≤ dat.dims[2]
   @assert ROI*2+1 ≤ dat.dims[1]

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

   # n_row = round(Int, sqrt(length(dat)))
   # n_col, rest = divrem(length(dat),n_row)
   # n_col = (rest==0 ? n_col : n_col+1)
   n_row = 4
   n_col = 6
   wspace=0.02
   hspace=0.02

   fig = figure(figsize=fsize)

   for t in 1:length(dat)
      ax = subplot(n_row, n_col, t)
      if scale
         imshow(snr_maps[t,:,:], origin="lower",
                norm=matplotlib.colors.TwoSlopeNorm(vmin=min_snr,
                                                    vcenter=0,
                                                    vmax=max_snr),
                cmap=c)
      else
         imshow(snr_maps[t,:,:], origin="lower",
             norm=matplotlib.colors.TwoSlopeNorm(vmin=minimum(snr_maps[t,:,:]),
                                                 vcenter=0,
                                                 vmax=maximum(snr_maps[t,:,:])),
             cmap=c)
      end

      date_Ymd = Dates.format(partial_year(dat.epochs[t]), "YYYY-mm-dd")
      text(ROI*0.1, ROI*0.15, date_Ymd, fontsize = 12, fontweight="bold",
           color = "black", bbox=Dict("facecolor"=>"white", "alpha"=>0.7),
           verticalalignment="center")
      # text(ROI*0.1, 2*ROI-ROI*0.2, "\$\\lambda_"*string(λ)*"\$", fontsize = 16,
      #      fontweight="bold", color = "black",
      #      bbox=Dict("facecolor"=>"white", "alpha"=>0.7),
      #      verticalalignment="center")
      if occursin("DB",dat.iflts[t])
          temp = split(dat.iflts[t],"_")[end]
          band = temp[1] * temp[λ+1]
      elseif occursin("BB",dat.iflts[t])
          band = replace(dat.iflts[t], "_" => " ")
      end

      text(ROI*0.1, 2*ROI-ROI*0.15, band, fontsize = 12,
           fontweight="bold", color = "black",
           bbox=Dict("facecolor"=>"white", "alpha"=>0.7),
           verticalalignment="center")

      # text((2*ROI+1)*0.75, ROI*0.15,
      #      "SNR=$(round(maximum(snr_maps[t,:,:]),digits=1))", fontsize = 12,
      #      fontweight="bold", color = "black",
      #      #bbox=Dict("facecolor"=>"white", "alpha"=>0.3),
      #      va="center",
      #      ha="center")

      cb = colorbar(shrink=1, pad=0.015, aspect=15)
      cb.ax.tick_params(labelsize=11, labelcolor="black")

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

   # C_map[C_map .<= 0] .= 1e-3

   fig = figure(figsize=(6,5))
   ax = fig.add_subplot(111)

   imshow(C_map,
          origin="lower",
          norm=matplotlib.colors.SymLogNorm(vmin=0, vmax=maximum(C_map),
                                            base=10, linscale=1, linthresh=1),
          cmap=c)
   # imshow(C_map,
   #        origin="lower",
   #        vmin=0, vmax=maximum(C_map),
   #        cmap="bwr")
   if showC
       text(nx*0.025, ny*0.97, L"\mathcal{C}(\widehat{\mu})="*
            "$(round(maximum(C_map), digits=digs))", fontsize=16,
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
            "$(round(mas_bar*mean(dat.pixres)/1000,digits=digs))"*"''",
            fontsize = 14,
            color = "black",
            va="center",
            ha="center", fontweight=fontweight)
   end

   cb = colorbar(shrink=1)
   # cb = colorbar(shrink=1, label=L"$\mathcal{C}$")
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
   # imshow(C_map,
   #        origin="lower",
   #        vmin=0, vmax=maximum(C_map),
   #        cmap="bwr")
   if showSNR
       text(nx*0.025, ny*0.97, L"\mathcal{S}/\mathcal{N}(\widehat{\mu})="*
            "$(round(maximum(SNR_map), digits=digs))", fontsize=16,
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
            "$(round(mas_bar*mean(dat.pixres)/1000,digits=digs))"*"''",
            fontsize = 14,
            color = "black",
            va="center",
            ha="center", fontweight=fontweight)
   end

   cb = colorbar(shrink=1)
   # cb = colorbar(shrink=1, label=L"$\mathcal{C}$")
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

# """
#    plot_PACOME_snr(dat, orb, λ, ROI; savePath) -> nothing
#
# plots the PACOME snr map at the positions
# computed for orbit `orb` on data `dat` at wavelength `λ` adjusted to the nearest
# pixel around a region of interest of radius `ROI`.
#
# Optional argument `savePath` is a `String` specifying the path where the plot is
# saved (default is `none`).
#
# """
#
# function plot_PACOME_snr(dat::Pacome.PacomeData{T,3},
#                                orb::Array{T},
#                                ROI::Int; kwds...) where {T<:AbstractFloat}
#
#    return plot_PACOME_snr(dat, Pacome.arr2orb(orb), ROI; kwds...)
# end
#
# function plot_PACOME_snr(dat::Pacome.PacomeData{T,3},
#                                orb::Pacome.Orbit{T},
#                                ROI::Int;
#                                cal::Bool=false,
#                                λ::Int=1,
#                                savePath::String="none",
#                                c::String="coolwarm",
#                                showmas::Bool=false,
#                                showSNR::Bool=false,
#                                fontweight::String="normal",
#                                digs::Int=1) where {T<:AbstractFloat}
#
#    @assert ROI*2+1 ≤ dat.dims[2]
#    @assert ROI*2+1 ≤ dat.dims[1]
#    nλ = dat.dims[end]
#    @assert 1 ≤ λ ≤ nλ
#    nx, ny = ROI*2+1, ROI*2+1
#
#    a_maps = Array{T,3}(undef, (length(dat), ROI*2+1, ROI*2+1))
#    b_maps = Array{T,3}(undef, (length(dat), ROI*2+1, ROI*2+1))
#    pos = Array{T,2}(undef, (length(dat),2))
#
#    for t in 1:length(dat)
#       ΔRA, ΔDec = Pacome.projected_position(orb, dat.epochs[t])
#       pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]
#       pos[t,:] .= pt.x, pt.y
#
#       i, j = round.(Int, pos[t,:])
#       a_maps[t,:,:] = dat.a[t][i-ROI:i+ROI,j-ROI:j+ROI,λ]
#       b_maps[t,:,:] = dat.b[t][i-ROI:i+ROI,j-ROI:j+ROI,λ]
#    end
#
#    if cal
#        error("Not implemented yet !")
#    else
#        SNR_map = reshape( sum(b_maps ./ a_maps, dims=1) ./
#                            sqrt.(sum(1. ./ a_maps, dims=1)), (ROI*2+1, ROI*2+1))
#    end
#
#    # C_map[C_map .<= 0] .= 1e-3
#
#    fig = figure(figsize=(6,5))
#    ax = fig.add_subplot(111)
#
#    imshow(SNR_map,
#           origin="lower",
#           norm=matplotlib.colors.TwoSlopeNorm(vmin=minimum(SNR_map),
#                                            vcenter=0.,
#                                            vmax=maximum(SNR_map)),
#           cmap=c)
#    # imshow(C_map,
#    #        origin="lower",
#    #        vmin=0, vmax=maximum(C_map),
#    #        cmap="bwr")
#    if showSNR
#        text(nx*0.025, ny*0.97, L"$\mathrm{SNR}(\widehat{\mu})=$"*
#             "$(round(maximum(SNR_map), digits=digs))", fontsize=16,
#             color = "crimson",
#             bbox=Dict("boxstyle"=>"round,pad=0.3", "fc"=>"white",
#             "ec"=>"crimson", "lw"=>1, "alpha"=>0.75),
#             verticalalignment="top", ha="left")
#    end
#    if showmas
#        # mas_bar_size = nx/6
#        mas_bar = nx/6
#        mas_x = [nx*0.95-mas_bar, nx*0.95]
#        mas_y = [ny*0.025, ny*0.025]
#
#        plot(mas_x, mas_y, linewidth=2.5, color="black")
#
#        text(mean(mas_x), mean(mas_y)+ny*0.03,
#             "$(round(mas_bar*mean(dat.pixres)/1000,digits=digs))"*"''",
#             fontsize = 14,
#             color = "black",
#             va="center",
#             ha="center", fontweight=fontweight)
#    end
#
#    cb = colorbar(shrink=1)
#    # cb = colorbar(shrink=1, label=L"$\mathcal{C}$")
#    # cb.ax.set_title(L"$\mathrm{SNR}_{\ell="*"$λ"*L"}$",fontsize=17)
#    cb.ax.tick_params(labelsize=14)
#
#    scatter(ROI, ROI, marker="+", s=100, color="black")
#    ax.tick_params(axis = "both", bottom = false, top = false,
#                   right = false, left = false)
#    ax.tick_params(labelbottom = false)
#    ax.tick_params(labelleft = false)
#    tight_layout()
#    display(fig)
#
#    if isdir(dirname(savePath))
#       fig.savefig(savePath)
#    elseif savePath != "none"
#       error("Not a directory.")
#    end
# end

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
   # println("maximum(C_map) = $(maximum(C_map)))")

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
   # cb.ax.set_title(L"$\mathcal{C}_{\ell="*"$λ"*L"}$",fontsize=20)
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
       # levels=cb.get_ticks())
       # ax.clabel(contours, inline=true, fontsize=10)
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


function plot_snr_oversampled(dat::Pacome.PacomeData{T,3},
                               orb::Array{T}, ker::Kernel{T},
                               rad::Int; kwds...) where {T<:AbstractFloat}

   return plot_snr_oversampled(dat, Pacome.arr2orb(orb), ker,
                                            rad; kwds...)
end

function plot_snr_oversampled(dat::Pacome.PacomeData{T,3},
                               orb::Pacome.Orbit{T},
                               ker::Kernel{T},
                               rad::Int;
                               λ::Int=0,
                               cal::Bool=false,
                               s::Int=4,
                               SNR_lim::T=-1.,
                               vmin::T=-1.,
                               vmax::T=-1.,
                               scale::String="linear",
                               showmas::Bool=false,
                               showSNR::Bool=false,
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

   SNR_map = fill!(Array{T,2}(undef, (npix, npix)),0)
   for t in 1:length(dat)
       for i in 1:npix
           for j in 1:npix
               for k in λs
                   if b_maps[t,k,i,j] > 0 && a_maps[t,k,i,j] > 0
                       SNR_map[i,j] += b_maps[t,k,i,j]^2 / a_maps[t,k,i,j]
                   end
               end
           end
       end
   end

   SNR_map = sqrt.(SNR_map)

   vmax == -1. ? vmax = maximum(SNR_map) : nothing
   vmin == -1. ? vmin = 0. : nothing
   SNR_lim == -1. ? vcenter = (vmax-minimum(SNR_map))/2 : vcenter = SNR_lim
   # println("maximum(C_map) = $(maximum(C_map)))")

   fig, ax  = subplots(figsize=(6,5))

   if scale == "log"
       ncmap = MidPointLogNorm(c, 0., vcenter, vmax, 1.)
       im = imshow(SNR_map,
                   origin="lower",
                   norm=matplotlib.colors.SymLogNorm(vmin=vmin,
                                                     vmax=vmax,
                                                     base=10, linscale=1,
                                                     linthresh=1),
                                                     cmap=ncmap)
    elseif scale == "linear"
        imshow(SNR_map,
               origin="lower",
               norm=matplotlib.colors.TwoSlopeNorm(vmin=vmin,
                                                   vcenter=vcenter,
                                                   vmax=vmax),
               cmap=c)
    end

   cb = colorbar(shrink=1)
   # cb.ax.set_title(L"$\mathcal{C}_{\ell="*"$λ"*L"}$",fontsize=20)
   cb.ax.tick_params(labelsize=14)

   if showSNR
       SNR = Pacome.snr(dat, orb, ker; cal=cal, λ=λ)
       text(nx*0.025, ny*0.97, L"\mathcal{S}/\mathcal{N}(\widehat{\mu})="*
            "$(round(SNR, digits=digs))", fontsize=16,
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
       contours = ax.contour(SNR_map, origin="lower",
                             norm=matplotlib.colors.PowerNorm(gamma=.5),
                             cmap="Greys",
                             levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.90]*
                                                               maximum(C_map))
       cb.add_lines(contours)
       # levels=cb.get_ticks())
       # ax.clabel(contours, inline=true, fontsize=10)
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


function plot_mono_epoch_snr_oversampled(dat::Pacome.PacomeData{T,3},
                               orb::Array{T}, ker::Kernel{T}, t::Int,
                               rad::Int; kwds...) where {T<:AbstractFloat}

   return plot_mono_epoch_snr_oversampled(dat, Pacome.arr2orb(orb), ker,
                                            rad; kwds...)
end

function plot_mono_epoch_snr_oversampled(dat::Pacome.PacomeData{T,3},
                               orb::Pacome.Orbit{T},
                               ker::Kernel{T},
                               t::Int,
                               rad::Int;
                               λ::Int=1,
                               cal::Bool=false,
                               s::Int=4,
                               SNR_lim::T=-1.,
                               vmin::T=-1.,
                               vmax::T=-1.,
                               scale::String="linear",
                               showmas::Bool=false,
                               showSNR::Bool=false,
                               fontweight::String="normal",
                               c::String="coolwarm",
                               cont::Bool=false,
                               savePath::String="none",
                               digs::Int=1,
                               transp::Bool=false) where {T<:AbstractFloat}

   @assert rad*2+1 ≤ dat.dims[2]
   @assert rad*2+1 ≤ dat.dims[1]
   nλ = dat.dims[end]
   @assert 1 ≤ λ ≤ nλ
   @assert 1 ≤ t ≤ length(dat)

   npix = rad*2*s
   nx, ny = npix, npix

   a_maps = Array{T,2}(undef, (npix, npix))
   b_maps = Array{T,2}(undef, (npix, npix))
   pos = Array{T,2}(undef, (length(dat),2))

   ΔRA, ΔDec = Pacome.projected_position(orb, dat.epochs[t])
   pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]

   for (xi,x) in enumerate(LinRange(pt.x-rad, pt.x+rad, npix))
       for (yi,y) in enumerate(LinRange(pt.y-rad, pt.y+rad, npix))
           a_maps[xi,yi] = Pacome.interpolate(dat.a[t], ker,
                                                 Point(x,y), λ)
           b_maps[xi,yi] = Pacome.interpolate(dat.b[t], ker,
                                                 Point(x,y), λ)
       end
   end

   SNR_map = fill!(Array{T,2}(undef, (npix, npix)),0)
   for i in 1:npix
       for j in 1:npix
           if a_maps[i,j] > 0
               SNR_map[i,j] = b_maps[i,j] / sqrt(a_maps[i,j])
           else
               SNR_map[i,j] = NaN
           end
       end
   end

   vmax == -1. ? vmax = maximum(SNR_map) : nothing
   vmin == -1. ? vmin = minimum(SNR_map) : nothing
   SNR_lim == -1. ? vcenter = (vmax-minimum(SNR_map))/2 : vcenter = SNR_lim
   # println("maximum(C_map) = $(maximum(C_map)))")

   fig, ax  = subplots(figsize=(6,5))

   if scale == "log"
       ncmap = MidPointLogNorm(c, 0., vcenter, vmax, 1.)
       im = imshow(SNR_map,
                   origin="lower",
                   norm=matplotlib.colors.SymLogNorm(vmin=vmin,
                                                     vmax=vmax,
                                                     base=10, linscale=1,
                                                     linthresh=1),
                                                     cmap=ncmap)
    elseif scale == "linear"
        imshow(SNR_map,
               origin="lower",
               norm=matplotlib.colors.TwoSlopeNorm(vmin=vmin,
                                                   vcenter=vcenter,
                                                   vmax=vmax),
               cmap=c)
    end

   cb = colorbar(shrink=1)
   # cb.ax.set_title(L"$\mathcal{C}_{\ell="*"$λ"*L"}$",fontsize=20)
   cb.ax.tick_params(labelsize=14)

   if showSNR
       a, b = Pacome.interpolate(dat, orb, ker)
       SNR = (b ./ sqrt.(a))[λ,t]
       text(nx*0.025, ny*0.97, L"\mathcal{S}/\mathcal{N}_{t,\ell}(\widehat{\mu})="*
            "$(round(SNR, digits=digs))", fontsize=16,
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
       contours = ax.contour(SNR_map, origin="lower",
                             norm=matplotlib.colors.PowerNorm(gamma=.5),
                             cmap="Greys",
                             levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.90]*
                                                               maximum(C_map))
       cb.add_lines(contours)
       # levels=cb.get_ticks())
       # ax.clabel(contours, inline=true, fontsize=10)
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

# """
#    plot_PACOME_snr_oversampled(dat, orb, ker, rad;
#                                      cal, λ, s, c, savePath) -> nothing
#
# plots the PACOME snr map at the positions
# computed for orbit `orb` on data `dat` at wavelength `λ` with interpolation
# kernel ker`.
#
# The 2-D map is resampled by a factor `s` (default is `s=4`) and the colormap `c`
# of the plot is tunable. Optional argument  `savePath` is a `String` specifying
# the path where the plot is saved (default is `none`).
#
# """
#
# function plot_PACOME_snr_oversampled(dat::Pacome.PacomeData{T,3},
#                                orb::Array{T}, ker::Kernel{T},
#                                rad::Int; kwds...) where {T<:AbstractFloat}
#
#    return plot_PACOME_snr_oversampled(dat, Pacome.arr2orb(orb), ker,
#                                             rad; kwds...)
# end
#
# function plot_PACOME_snr_oversampled(dat::Pacome.PacomeData{T,3},
#                                orb::Pacome.Orbit{T},
#                                ker::Kernel{T},
#                                rad::Int;
#                                cal::Bool=false,
#                                λ::Int=1,
#                                s::Int=4,
#                                showmas::Bool=false,
#                                showSNR::Bool=false,
#                                fontweight::String="normal",
#                                c::String="coolwarm",
#                                cont::Bool=false,
#                                savePath::String="none",
#                                digs::Int=1,
#                                transp::Bool=false) where {T<:AbstractFloat}
#
#    @assert rad*2+1 ≤ dat.dims[2]
#    @assert rad*2+1 ≤ dat.dims[1]
#    nλ = dat.dims[end]
#    @assert 1 ≤ λ ≤ nλ
#
#    npix = rad*2*s
#    nx, ny = npix, npix
#
#    a_maps = Array{T,3}(undef, (length(dat), npix, npix))
#    b_maps = Array{T,3}(undef, (length(dat), npix, npix))
#    pos = Array{T,2}(undef, (length(dat),2))
#
#    for t in 1:length(dat)
#       ΔRA, ΔDec = Pacome.projected_position(orb, dat.epochs[t])
#       pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]
#
#       for (xi,x) in enumerate(LinRange(pt.x-rad, pt.x+rad, npix))
#           for (yi,y) in enumerate(LinRange(pt.y-rad, pt.y+rad, npix))
#               a_maps[t,xi,yi] = Pacome.interpolate(dat.a[t], ker, Point(x,y), λ)
#               b_maps[t,xi,yi] = Pacome.interpolate(dat.b[t], ker, Point(x,y), λ)
#           end
#       end
#    end
#
#    if cal
#        error("not implemented yet !")
#    else
#        SNR_map = reshape(sum(b_maps ./ a_maps, dims=1) ./
#                        sqrt.(sum(1. ./ a_maps, dims=1)), (npix, npix))
#    end
#
#    fig, ax  = subplots(figsize=(6,5))
#
#    im = imshow(SNR_map,
#           origin="lower",
#           norm=matplotlib.colors.TwoSlopeNorm(vmin=minimum(SNR_map),
#                                               vcenter=0.,
#                                               vmax=maximum(SNR_map)),
#           cmap=c)
#    # im = imshow(C_map,
#    #        origin="lower",
#    #        vmin=0, vmax=maximum(C_map),
#    #        cmap=c)
#    # imshow(C_map,
#    #        origin="lower",
#    #        norm=matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=37.5,
#    #                                            vmax=maximum(C_map)),
#    #        cmap=c)
#
#    # im = ax.imshow(C_map,
#    #       origin="lower",
#    #       norm=matplotlib.colors.PowerNorm(gamma=.25),
#    #       cmap=c)
#    cb = colorbar(shrink=1)
#    # cb.ax.set_title(L"$\mathrm{SNR}_{\ell="*"$λ"*L"}$",fontsize=20)
#    cb.ax.tick_params(labelsize=14)
#
#    if showSNR
#        SNR = Pacome.PACOME_snr(dat, orb, ker; cal=cal, λ=λ)
#        text(nx*0.025, ny*0.97, L"\mathrm{SNR}(\widehat{\mu})="*
#             "$(round(SNR, digits=digs))", fontsize=16,
#             color = "crimson",
#             bbox=Dict("boxstyle"=>"round,pad=0.3", "fc"=>"white",
#             "ec"=>"crimson", "lw"=>1, "alpha"=>0.75),
#             verticalalignment="top", ha="left")
#    end
#    if showmas
#        # mas_bar_size = nx/6
#        mas_bar = nx/6
#        mas_x = [nx*0.95-mas_bar, nx*0.95]
#        mas_y = [ny*0.025, ny*0.025]
#
#        plot(mas_x, mas_y, linewidth=2.5, color="black")
#
#        text(mean(mas_x), mean(mas_y)+ny*0.03,
#             "$(round((mas_bar-1)/s*mean(dat.pixres)/1000,digits=digs))"*"''",
#             fontsize = 14,
#             color = "black",
#             va="center",
#             ha="center", fontweight=fontweight)
#    end
#
#    if cont
#        contours = ax.contour(SNR_map, origin="lower",
#                              norm=matplotlib.colors.PowerNorm(gamma=.5),
#                              cmap="Greys",
#                              levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.90]*
#                                                                maximum(SNR_map))
#        cb.add_lines(contours)
#        # levels=cb.get_ticks())
#        # ax.clabel(contours, inline=true, fontsize=10)
#    else
#        ax.scatter(npix/2-0.5, npix/2-0.5, marker="+", s=100, color="black")
#    end
#
#    #
#    ax.tick_params(axis = "both", bottom = false, top = false, right = false,
#                   left = false, labelbottom = false, labelleft = false)
#    ax.tick_params()
#    ax.tick_params()
#    tight_layout()
#    display(fig)
#
#    if isdir(dirname(savePath))
#       fig.savefig(savePath, transparent=transp)
#    elseif savePath != "none"
#       error("Not a directory.")
#    end
# end
#
# """
#    plot_PACOME_snr_oversampled(dat, orb, ker, rad;
#                                      cal, λ, s, c, savePath) -> nothing
#
# plots the PACOME snr map at the positions
# computed for orbit `orb` on data `dat` at wavelength `λ` with interpolation
# kernel ker`.
#
# The 2-D map is resampled by a factor `s` (default is `s=4`) and the colormap `c`
# of the plot is tunable. Optional argument  `savePath` is a `String` specifying
# the path where the plot is saved (default is `none`).
#
# """
#
# function plot_PACOME_snr_oversampled(dat::Pacome.PacomeData{T,3},
#                                orb::Array{T}, ker::Kernel{T},
#                                rad::Int; kwds...) where {T<:AbstractFloat}
#
#    return plot_PACOME_snr_oversampled(dat, Pacome.arr2orb(orb), ker,
#                                             rad; kwds...)
# end
#
# function plot_PACOME_snr_oversampled(dat::Pacome.PacomeData{T,3},
#                                orb::Pacome.Orbit{T},
#                                ker::Kernel{T},
#                                rad::Int;
#                                cal::Bool=false,
#                                λ::Int=1,
#                                s::Int=4,
#                                showmas::Bool=false,
#                                showSNR::Bool=false,
#                                fontweight::String="normal",
#                                c::String="coolwarm",
#                                cont::Bool=false,
#                                savePath::String="none",
#                                digs::Int=1,
#                                transp::Bool=false) where {T<:AbstractFloat}
#
#    @assert rad*2+1 ≤ dat.dims[2]
#    @assert rad*2+1 ≤ dat.dims[1]
#    nλ = dat.dims[end]
#    @assert 1 ≤ λ ≤ nλ
#
#    npix = rad*2*s
#    nx, ny = npix, npix
#
#    a_maps = Array{T,3}(undef, (length(dat), npix, npix))
#    b_maps = Array{T,3}(undef, (length(dat), npix, npix))
#    pos = Array{T,2}(undef, (length(dat),2))
#
#    for t in 1:length(dat)
#       ΔRA, ΔDec = Pacome.projected_position(orb, dat.epochs[t])
#       pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]
#
#       for (xi,x) in enumerate(LinRange(pt.x-rad, pt.x+rad, npix))
#           for (yi,y) in enumerate(LinRange(pt.y-rad, pt.y+rad, npix))
#               a_maps[t,xi,yi] = Pacome.interpolate(dat.a[t], ker, Point(x,y), λ)
#               b_maps[t,xi,yi] = Pacome.interpolate(dat.b[t], ker, Point(x,y), λ)
#           end
#       end
#    end
#
#    if cal
#        error("not implemented yet !")
#    else
#        SNR_map = reshape(sum(b_maps ./ a_maps, dims=1) ./
#                        sqrt.(sum(1. ./ a_maps, dims=1)), (npix, npix))
#    end
#
#    fig, ax  = subplots(figsize=(6,5))
#
#    im = imshow(SNR_map,
#           origin="lower",
#           norm=matplotlib.colors.TwoSlopeNorm(vmin=minimum(SNR_map),
#                                               vcenter=0.,
#                                               vmax=maximum(SNR_map)),
#           cmap=c)
#    # im = imshow(C_map,
#    #        origin="lower",
#    #        vmin=0, vmax=maximum(C_map),
#    #        cmap=c)
#    # imshow(C_map,
#    #        origin="lower",
#    #        norm=matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=37.5,
#    #                                            vmax=maximum(C_map)),
#    #        cmap=c)
#
#    # im = ax.imshow(C_map,
#    #       origin="lower",
#    #       norm=matplotlib.colors.PowerNorm(gamma=.25),
#    #       cmap=c)
#    cb = colorbar(shrink=1)
#    # cb.ax.set_title(L"$\mathrm{SNR}_{\ell="*"$λ"*L"}$",fontsize=20)
#    cb.ax.tick_params(labelsize=14)
#
#    if showSNR
#        SNR = Pacome.PACOME_snr(dat, orb, ker; cal=cal, λ=λ)
#        text(nx*0.025, ny*0.97, L"\mathrm{SNR}(\widehat{\mu})="*
#             "$(round(SNR, digits=digs))", fontsize=16,
#             color = "crimson",
#             bbox=Dict("boxstyle"=>"round,pad=0.3", "fc"=>"white",
#             "ec"=>"crimson", "lw"=>1, "alpha"=>0.75),
#             verticalalignment="top", ha="left")
#    end
#    if showmas
#        # mas_bar_size = nx/6
#        mas_bar = nx/6
#        mas_x = [nx*0.95-mas_bar, nx*0.95]
#        mas_y = [ny*0.025, ny*0.025]
#
#        plot(mas_x, mas_y, linewidth=2.5, color="black")
#
#        text(mean(mas_x), mean(mas_y)+ny*0.03,
#             "$(round((mas_bar-1)/s*mean(dat.pixres)/1000,digits=digs))"*"''",
#             fontsize = 14,
#             color = "black",
#             va="center",
#             ha="center", fontweight=fontweight)
#    end
#
#    if cont
#        contours = ax.contour(SNR_map, origin="lower",
#                              norm=matplotlib.colors.PowerNorm(gamma=.5),
#                              cmap="Greys",
#                              levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.90]*
#                                                                maximum(SNR_map))
#        cb.add_lines(contours)
#        # levels=cb.get_ticks())
#        # ax.clabel(contours, inline=true, fontsize=10)
#    else
#        ax.scatter(npix/2-0.5, npix/2-0.5, marker="+", s=100, color="black")
#    end
#
#    #
#    ax.tick_params(axis = "both", bottom = false, top = false, right = false,
#                   left = false, labelbottom = false, labelleft = false)
#    ax.tick_params()
#    ax.tick_params()
#    tight_layout()
#    display(fig)
#
#    if isdir(dirname(savePath))
#       fig.savefig(savePath, transparent=transp)
#    elseif savePath != "none"
#       error("Not a directory.")
#    end
# end

# """
#    plot_PACOME_snrEric_oversampled(dat, orb, ker, rad;
#                                      cal, λ, s, c, savePath) -> nothing
#
# plots the PACOME snr map at the positions
# computed for orbit `orb` on data `dat` at wavelength `λ` with interpolation
# kernel ker`.
#
# The 2-D map is resampled by a factor `s` (default is `s=4`) and the colormap `c`
# of the plot is tunable. Optional argument  `savePath` is a `String` specifying
# the path where the plot is saved (default is `none`).
#
# """
#
# function plot_PACOME_snrEric_oversampled(dat::Pacome.PacomeData{T,3},
#                                orb::Array{T}, ker::Kernel{T},
#                                rad::Int; kwds...) where {T<:AbstractFloat}
#
#    return plot_PACOME_snrEric_oversampled(dat, Pacome.arr2orb(orb), ker,
#                                             rad; kwds...)
# end
#
# function plot_PACOME_snrEric_oversampled(dat::Pacome.PacomeData{T,3},
#                                orb::Pacome.Orbit{T},
#                                ker::Kernel{T},
#                                rad::Int;
#                                cal::Bool=false,
#                                λ::Int=1,
#                                s::Int=4,
#                                showmas::Bool=false,
#                                showSNR::Bool=false,
#                                fontweight::String="normal",
#                                c::String="coolwarm",
#                                cont::Bool=false,
#                                savePath::String="none",
#                                digs::Int=1,
#                                transp::Bool=false) where {T<:AbstractFloat}
#
#    @assert rad*2+1 ≤ dat.dims[2]
#    @assert rad*2+1 ≤ dat.dims[1]
#    nλ = dat.dims[end]
#    @assert 1 ≤ λ ≤ nλ
#
#    npix = rad*2*s
#    nx, ny = npix, npix
#
#    a_maps = Array{T,3}(undef, (length(dat), npix, npix))
#    b_maps = Array{T,3}(undef, (length(dat), npix, npix))
#    pos = Array{T,2}(undef, (length(dat),2))
#
#    for t in 1:length(dat)
#       ΔRA, ΔDec = Pacome.projected_position(orb, dat.epochs[t])
#       pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]
#
#       for (xi,x) in enumerate(LinRange(pt.x-rad, pt.x+rad, npix))
#           for (yi,y) in enumerate(LinRange(pt.y-rad, pt.y+rad, npix))
#               a_maps[t,xi,yi] = Pacome.interpolate(dat.a[t], ker, Point(x,y), λ)
#               b_maps[t,xi,yi] = Pacome.interpolate(dat.b[t], ker, Point(x,y), λ)
#           end
#       end
#    end
#
#    if cal
#        error("not implemented yet !")
#    else
#        SNR_map = reshape(sqrt.(sum(b_maps.^2 ./ a_maps, dims=1)), (npix, npix))
#    end
#
#    fig, ax  = subplots(figsize=(6,5))
#
#    im = imshow(SNR_map,
#           origin="lower",
#           norm=matplotlib.colors.TwoSlopeNorm(vmin=minimum(SNR_map),
#                                               vcenter=mean(SNR_map),
#                                               vmax=maximum(SNR_map)),
#           cmap=c)
#    # im = imshow(C_map,
#    #        origin="lower",
#    #        vmin=0, vmax=maximum(C_map),
#    #        cmap=c)
#    # imshow(C_map,
#    #        origin="lower",
#    #        norm=matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=37.5,
#    #                                            vmax=maximum(C_map)),
#    #        cmap=c)
#
#    # im = ax.imshow(C_map,
#    #       origin="lower",
#    #       norm=matplotlib.colors.PowerNorm(gamma=.25),
#    #       cmap=c)
#    cb = colorbar(shrink=1)
#    # cb.ax.set_title(L"$\mathrm{SNR}_{\ell="*"$λ"*L"}$",fontsize=20)
#    cb.ax.tick_params(labelsize=14)
#
#    if showSNR
#        SNR = Pacome.PACOME_snr(dat, orb, ker; cal=cal, λ=λ)
#        text(nx*0.025, ny*0.97, L"\mathrm{SNR}(\widehat{\mu})="*
#             "$(round(SNR, digits=digs))", fontsize=16,
#             color = "crimson",
#             bbox=Dict("boxstyle"=>"round,pad=0.3", "fc"=>"white",
#             "ec"=>"crimson", "lw"=>1, "alpha"=>0.75),
#             verticalalignment="top", ha="left")
#    end
#    if showmas
#        # mas_bar_size = nx/6
#        mas_bar = nx/6
#        mas_x = [nx*0.95-mas_bar, nx*0.95]
#        mas_y = [ny*0.025, ny*0.025]
#
#        plot(mas_x, mas_y, linewidth=2.5, color="black")
#
#        text(mean(mas_x), mean(mas_y)+ny*0.03,
#             "$(round((mas_bar-1)/s*mean(dat.pixres)/1000,digits=digs))"*"''",
#             fontsize = 14,
#             color = "black",
#             va="center",
#             ha="center", fontweight=fontweight)
#    end
#
#    if cont
#        contours = ax.contour(SNR_map, origin="lower",
#                              norm=matplotlib.colors.PowerNorm(gamma=.5),
#                              cmap="Greys",
#                              levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.90]*
#                                                                maximum(SNR_map))
#        cb.add_lines(contours)
#        # levels=cb.get_ticks())
#        # ax.clabel(contours, inline=true, fontsize=10)
#    else
#        ax.scatter(npix/2-0.5, npix/2-0.5, marker="+", s=100, color="black")
#    end
#
#    #
#    ax.tick_params(axis = "both", bottom = false, top = false, right = false,
#                   left = false, labelbottom = false, labelleft = false)
#    ax.tick_params()
#    ax.tick_params()
#    tight_layout()
#    display(fig)
#
#    if isdir(dirname(savePath))
#       fig.savefig(savePath, transparent=transp)
#    elseif savePath != "none"
#       error("Not a directory.")
#    end
# end
#
# """
#    plot_PACOME_snrPositEric_oversampled(dat, orb, ker, rad;
#                                      cal, λ, s, c, savePath) -> nothing
#
# plots the PACOME snr map at the positions
# computed for orbit `orb` on data `dat` at wavelength `λ` with interpolation
# kernel ker`.
#
# The 2-D map is resampled by a factor `s` (default is `s=4`) and the colormap `c`
# of the plot is tunable. Optional argument  `savePath` is a `String` specifying
# the path where the plot is saved (default is `none`).
#
# """
#
# function plot_PACOME_snrPositEric_oversampled(dat::Pacome.PacomeData{T,3},
#                                orb::Array{T}, ker::Kernel{T},
#                                rad::Int; kwds...) where {T<:AbstractFloat}
#
#    return plot_PACOME_snrPositEric_oversampled(dat, Pacome.arr2orb(orb), ker,
#                                             rad; kwds...)
# end
#
# function plot_PACOME_snrPositEric_oversampled(dat::Pacome.PacomeData{T,3},
#                                orb::Pacome.Orbit{T},
#                                ker::Kernel{T},
#                                rad::Int;
#                                cal::Bool=false,
#                                λ::Int=1,
#                                s::Int=4,
#                                showmas::Bool=false,
#                                showSNR::Bool=false,
#                                fontweight::String="normal",
#                                c::String="coolwarm",
#                                cont::Bool=false,
#                                savePath::String="none",
#                                digs::Int=1,
#                                transp::Bool=false) where {T<:AbstractFloat}
#
#    @assert rad*2+1 ≤ dat.dims[2]
#    @assert rad*2+1 ≤ dat.dims[1]
#    nλ = dat.dims[end]
#    @assert 1 ≤ λ ≤ nλ
#
#    npix = rad*2*s
#    nx, ny = npix, npix
#
#    a_maps = Array{T,3}(undef, (length(dat), npix, npix))
#    b_maps = Array{T,3}(undef, (length(dat), npix, npix))
#    pos = Array{T,2}(undef, (length(dat),2))
#
#    for t in 1:length(dat)
#       ΔRA, ΔDec = Pacome.projected_position(orb, dat.epochs[t])
#       pt = dat.centers[t] + Point(-ΔRA, ΔDec)/dat.pixres[t]
#
#       for (xi,x) in enumerate(LinRange(pt.x-rad, pt.x+rad, npix))
#           for (yi,y) in enumerate(LinRange(pt.y-rad, pt.y+rad, npix))
#               a_maps[t,xi,yi] = Pacome.interpolate(dat.a[t], ker, Point(x,y), λ)
#               b_maps[t,xi,yi] = Pacome.interpolate(dat.b[t], ker, Point(x,y), λ)
#           end
#       end
#    end
#
#    if cal
#        error("not implemented yet !")
#    else
#        SNR_map = reshape(sqrt.(sum(max.(b_maps,0).^2 ./ a_maps, dims=1)), (npix, npix))
#    end
#
#    fig, ax  = subplots(figsize=(6,5))
#
#    im = imshow(SNR_map,
#           origin="lower",
#           norm=matplotlib.colors.TwoSlopeNorm(vmin=minimum(SNR_map),
#                                               vcenter=mean(SNR_map),
#                                               vmax=maximum(SNR_map)),
#           cmap=c)
#    # im = imshow(C_map,
#    #        origin="lower",
#    #        vmin=0, vmax=maximum(C_map),
#    #        cmap=c)
#    # imshow(C_map,
#    #        origin="lower",
#    #        norm=matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=37.5,
#    #                                            vmax=maximum(C_map)),
#    #        cmap=c)
#
#    # im = ax.imshow(C_map,
#    #       origin="lower",
#    #       norm=matplotlib.colors.PowerNorm(gamma=.25),
#    #       cmap=c)
#    cb = colorbar(shrink=1)
#    # cb.ax.set_title(L"$\mathrm{SNR}_{\ell="*"$λ"*L"}$",fontsize=20)
#    cb.ax.tick_params(labelsize=14)
#
#    if showSNR
#        SNR = Pacome.PACOME_snr(dat, orb, ker; cal=cal, λ=λ)
#        text(nx*0.025, ny*0.97, L"\mathrm{SNR}(\widehat{\mu})="*
#             "$(round(SNR, digits=digs))", fontsize=16,
#             color = "crimson",
#             bbox=Dict("boxstyle"=>"round,pad=0.3", "fc"=>"white",
#             "ec"=>"crimson", "lw"=>1, "alpha"=>0.75),
#             verticalalignment="top", ha="left")
#    end
#    if showmas
#        # mas_bar_size = nx/6
#        mas_bar = nx/6
#        mas_x = [nx*0.95-mas_bar, nx*0.95]
#        mas_y = [ny*0.025, ny*0.025]
#
#        plot(mas_x, mas_y, linewidth=2.5, color="black")
#
#        text(mean(mas_x), mean(mas_y)+ny*0.03,
#             "$(round((mas_bar-1)/s*mean(dat.pixres)/1000,digits=digs))"*"''",
#             fontsize = 14,
#             color = "black",
#             va="center",
#             ha="center", fontweight=fontweight)
#    end
#
#    if cont
#        contours = ax.contour(SNR_map, origin="lower",
#                              norm=matplotlib.colors.PowerNorm(gamma=.5),
#                              cmap="Greys",
#                              levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.90]*
#                                                                maximum(SNR_map))
#        cb.add_lines(contours)
#        # levels=cb.get_ticks())
#        # ax.clabel(contours, inline=true, fontsize=10)
#    else
#        ax.scatter(npix/2-0.5, npix/2-0.5, marker="+", s=100, color="black")
#    end
#
#    #
#    ax.tick_params(axis = "both", bottom = false, top = false, right = false,
#                   left = false, labelbottom = false, labelleft = false)
#    ax.tick_params()
#    ax.tick_params()
#    tight_layout()
#    display(fig)
#
#    if isdir(dirname(savePath))
#       fig.savefig(savePath, transparent=transp)
#    elseif savePath != "none"
#       error("Not a directory.")
#    end
# end

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
      # elseif k == 1
      #      plot(ΔRA,
      #      ΔDec,
      #      linewidth = 0.75,
      #      color = "darkorange",
      #      alpha = 0.6,
      #      zorder=3,
      #      label=otherLabel)
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
    # if scale_cb == "lin"
    #     norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # elseif scale_cb == "pow"
    #     norm = matplotlib.colors.PowerNorm(gamma=0.5)
    # elseif scale_cb == "log"
    #     norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    # else
    #     error("scale_cb $scale_cb not recognized...")
    # end
    # norm = matplotlib.colors.LogNorm(vmin=minimum(all_C), vmax=max(C,maximum(all_C)))
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
      # elseif k == 1
      #      plot(ΔRA,
      #      ΔDec,
      #      linewidth = 0.75,
      #      color = "darkorange",
      #      alpha = 0.6,
      #      zorder=3,
      #      label=otherLabel)
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


"""
   plot_search_grid_coverage(data, field, ROI; savePath) -> nothing

plots the search grid coverage matrix contained in `field` for data `dat` around
a region of interest `ROI`.

Optional argument `savePath` is a `String` specifying the path where the plot is
saved (default is `none`)

"""
function plot_search_grid_coverage(data::Pacome.PacomeData{T},
                             field::AbstractArray{T,2},
                             ROI::Int;
                             savePath::String="none") where {T<:AbstractFloat}

   field = transpose(field)

   pixres = mean(data.pixres)
   nx, ny = data.dims[1:2]

   fig = figure(figsize=(7,7))
   imshow(field, origin="lower",
          norm=matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=1,
                                              vmax=maximum(field)),
          cmap ="bwr", extent=(nx÷2-nx,nx÷2,ny÷2-ny,ny÷2) .* pixres)
   xlim((-ROI,ROI) .* pixres)
   ylim((-ROI,ROI) .* pixres)
   xlabel("ΔRA [mas]", fontsize=15)
   ylabel("ΔDec [mas]", fontsize=15)
   xticks(fontsize=13)
   yticks(fontsize=13)

   cb = colorbar(fraction=0.046, orientation="horizontal", pad=0.1)
   cb.ax.tick_params(labelsize=12)
   cb.update_ticks()

   scatter(0,0,color="black",marker="+",s=200, linewidth=3)

   tight_layout()
   subplots_adjust(top=0.979,
                     bottom=0.057,
                     left=0.124,
                     right=0.979,
                     hspace=0.2,
                     wspace=0.2)
   display(fig)

   if isdir(dirname(savePath))
      fig.savefig(savePath)
   elseif savePath != "none"
      error("Not a directory.")
   end
end

"""
   slider_imagettes(im)

creates an interactive visualization tool allowing to plot multi-dimensional
images `im` (of 5 dimensions).

"""
function slider_imagettes(im::AbstractArray{T,5}) where {T<:AbstractFloat}
    fig, ax = subplots(figsize=(6,6))
    subplots_adjust(top=0.99,
                    bottom=0.19,
                    left=0.065,
                    right=1.0,
                    hspace=0.0,
                    wspace=0.0)

    frame = imshow(im[1,1,:,:,1])

    ax.margins(x=0)
    axcolor = "white"

    ax_k  = plt.axes([0.25, 0.11, 0.65, 0.03], facecolor=axcolor)
    ax_t  = plt.axes([0.25, 0.07, 0.65, 0.03], facecolor=axcolor)
    ax_λ  = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)

    s_k = matplotlib.widgets.Slider(ax_k, "   N° iter : k", 1, size(im)[1], valinit=1, valstep=1)
    s_t = matplotlib.widgets.Slider(ax_t, "     Epoch : t", 1, size(im)[2], valinit=1, valstep=1)
    s_λ = matplotlib.widgets.Slider(ax_λ, "Wavelength : λ", 1, size(im)[5], valinit=1, valstep=1)

    function update(val::Int)
        new_k  = s_k.val
        new_t  = s_t.val
        new_λ  = s_λ.val

        frame.set_data(im[new_k, new_t, :, :, new_λ])
        fig.canvas.draw_idle()
    end

    s_k.on_changed(update)
    s_t.on_changed(update)
    s_λ.on_changed(update)

end

"""
   plot_orbitalParam_covariance_matrix(Cov; savePath)

plots the covariance matrix `Cov`. If `Cov` has 3 dimensions (7 x 7 x nλ), the
covariance matrices of each spectral channels are displayed.

Optional argument `savePath` is a `String` specifying the path where the plot is
saved (default is `none`)

"""
function plot_orbitalParam_covariance_matrix(Cov::Array{T,3};
                               savePath::String="none") where {T<:AbstractFloat}

    C = copy(Cov)
    nμ = size(C,1)

    for i in 1:nμ
        for j in i+1:nμ
            C[i,j,:] .= NaN
        end
    end

    μ = ["\$a\$ [pix]", "\$e\$ [-]", "\$i\$ [°]", "\$τ\$ [-]",
         "\$\\omega\$ [°]", "\$\\Omega\$ [°]", "\$K\$ [pix^3/yr^2]"]

    f = figure(figsize=(1.2*nμ*2,1.2*nμ/1.25))

    for nλ in 1:size(C,3)

        valMin, valMax = extrema(C[:,:,nλ])

        ax = f.add_subplot(1, size(C,3), nλ)
        imshow(C[:,:,nλ],
                  # norm=matplotlib.colors.LogNorm(vmin=valMin,
                  #                                vmax=valMax),
                  # norm=matplotlib.colors.TwoSlopeNorm(vmin=valMin,
                  #                                     vcenter=0,
                  #                                     vmax=valMax),
                  norm=matplotlib.colors.SymLogNorm(linthresh=20,
                                                    vmin=valMin,
                                                    #vcenter=0,
                                                    vmax=valMax),
                  # norm=matplotlib.colors.TwoSlopeNorm(vmin=valMin-rangeColor*1/5,
                  #                                     vcenter=valMin-rangeColor*1/5/2+valMax/2,
                  #                                     vmax=valMax),
                  cmap = "bwr")

        # cmap = "Greens"
        # cmap = "copper_r"
        # cmap = "summer_r"
        for i in 1:nμ
            for j in 1:i
                text(j-1, i-1,round(C[i,j,nλ],digits=2),
                     va="center", ha="center", rotation=45,
                     fontsize=12) #, fontweight="bold")
            end
        end

        title("@ "*"\$\\lambda_"*"$nλ"*"\$", fontsize=25, fontweight="bold")

        ax.set_xticks(0.5:nμ-0.5, minor=false)
        ax.set_yticks(0.5:nμ-0.5, minor=false)
        ax.set_xticks(0:nμ-1, minor=true)
        ax.set_yticks(0:nμ-1, minor=true)
        ax.set_xticklabels(μ, fontsize=14, minor=true)
        ax.set_yticklabels(μ, fontsize=14, minor=true)
        ax.grid(color="white", linestyle="-", linewidth=2, which="major")

        cb = colorbar(shrink=1)
        cb.ax.tick_params(labelsize=14)

        ax.tick_params(axis = "both", which="both",
                       bottom = false, top = false,
                       right = false, left = false)
        ax.tick_params(labelbottom=false, labelleft=false, which="major")
        ax.tick_params(labelbottom=true, labelleft=true, which="minor")
        ax.spines["right"].set_visible(false)
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        ax.spines["top"].set_visible(false)
    end

    tight_layout()
    display(f)

    if isdir(dirname(savePath))
       fig.savefig(savePath)
    elseif savePath != "none"
       error("Not a directory.")
    end
end


function plot_orbitalParam_covariance_matrix(Cov::Array{T,2};
                               savePath::String="none") where {T<:AbstractFloat}

    C = copy(Cov)
    nμ = size(C,1)
    valMin, valMax = extrema(C)

    for i in 1:nμ
        for j in i+1:nμ
            C[i,j,:] .= NaN
        end
    end

    μ = ["\$a\$ [pix]", "\$e\$ [-]", "\$i\$ [°]", "\$τ\$ [-]",
         "\$\\omega\$ [°]", "\$\\Omega\$ [°]", "\$K\$ [mas\$^3\$/yr\$^2\$]"]

    f = figure(figsize=(1.2*nμ,1.2*nμ/1.25))

    ax = f.add_subplot(1, size(C,3), 1)
    imshow(C,
              norm=matplotlib.colors.SymLogNorm(linthresh=20,
                                                vmin=valMin,
                                                vmax=valMax),
              cmap = "bwr")

    for i in 1:nμ
        for j in 1:i
            text(j-1, i-1,round(C[i,j],digits=2),
                 va="center", ha="center", rotation=45,
                 fontsize=12)
        end
    end

    ax.set_xticks(0.5:nμ-0.5, minor=false)
    ax.set_yticks(0.5:nμ-0.5, minor=false)
    ax.set_xticks(0:nμ-1, minor=true)
    ax.set_yticks(0:nμ-1, minor=true)
    ax.set_xticklabels(μ, fontsize=14, minor=true)
    ax.set_yticklabels(μ, fontsize=14, minor=true)
    ax.grid(color="white", linestyle="-", linewidth=2, which="major")

    cb = colorbar(shrink=1)
    cb.ax.tick_params(labelsize=14)

    ax.tick_params(axis = "both", which="both",
                   bottom = false, top = false,
                   right = false, left = false)
    ax.tick_params(labelbottom=false, labelleft=false, which="major")
    ax.tick_params(labelbottom=true, labelleft=true, which="minor")
    ax.spines["right"].set_visible(false)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.spines["top"].set_visible(false)


    tight_layout()
    display(f)

    if isdir(dirname(savePath))
       fig.savefig(savePath)
    elseif savePath != "none"
       error("Not a directory.")
    end
end

"""
   plot_costfunc_orbitElem(orbs, fcost, sgrid; thresh, cal, path)

plots the distribution of PACOME's cost function `fcost` with respect to all
orbital elements contained in `orbs` explored on the search grid `sgrid`.


Optional arguments `cal` refers to whether the cost function was used in
calibrated mode or uncalibrated (default is `false`). Only the orbital
elements for which the cost function value is above `thresh`*`max(fcost)` are
plotted (default is `thresh=0.8`). `path` is a `String` specifying the path
where the plot is saved (default is `none`)

"""
function plot_costfunc_orbitElem(orbs::Array{T,2},
                                 fcost::Array{T,1},
                                 sgrid::Pacome.Grid{T};
                                 thresh::T=0.8,
                                 cal::Bool=false,
                                 path::String="none") where {T<:AbstractFloat}

    # ORDER PARAMETERS
    # a  -> 1; e  -> 2; i  -> 3; τ -> 4; ω  -> 5; Ω  -> 6;  K -> 7
    #     a  e  i  τ ω  Ω  K
    tl = [6, 6, 6, 7, 7, 7, 6]

    idx = [1, 2, 3, 5, 6, 4, 7]
    fcost_min, fcost_max = maximum(fcost)*thresh, maximum(fcost)*1.01
    slct = fcost .≥ fcost_min
    fcost = fcost[slct]
    orbs = orbs[:,slct]

    fig = figure(figsize=(6,8.75))
    for μ in 1:7
        subplot(7,1,μ)
        scatter(fcost, orbs[idx[μ],:], s=10, alpha=0.6, c="royalblue")
        scatter(fcost[argmax(fcost)], orbs[idx[μ],argmax(fcost)],
                s=15, c="orangered")
        xlim(fcost_min, fcost_max)

        if idx[μ] == 1
            yticks(range(minimum(sgrid.a),maximum(sgrid.a),length=tl[idx[μ]]))
            ylabel("\$a\$ [mas]", fontsize=13)
        elseif idx[μ] == 2
            yticks(range(minimum(sgrid.e),maximum(sgrid.e),length=tl[idx[μ]]))
            ylabel("\$e\$ [-]", fontsize=13)
        elseif idx[μ] == 3
            yticks(range(minimum(sgrid.i),maximum(sgrid.i),length=tl[idx[μ]]))
            ylabel("\$i\$ [°]", fontsize=13)
            ylim(-8, 188)
        elseif idx[μ] == 4
            yticks(range(minimum(sgrid.τ),maximum(sgrid.τ),length=tl[idx[μ]]))
            ylabel("\$τ\$ [-]", fontsize=13)
        elseif idx[μ] == 5
            yticks(range(minimum(sgrid.ω),maximum(sgrid.ω),length=tl[idx[μ]]))
            ylabel("\$\\omega\$ [°]", fontsize=13)
        elseif idx[μ] == 6
            yticks(range(minimum(sgrid.Ω),maximum(sgrid.Ω),length=tl[idx[μ]]))
            ylabel("\$\\Omega\$ [°]", fontsize=13)
        elseif idx[μ] == 7
            yticks(range(minimum(sgrid.P),maximum(sgrid.P),length=tl[idx[μ]]))
            ylabel("\$K\$ [mas\$^3\$/yr\$^2\$]", fontsize=13)
        end

        if μ == 7
            if cal
                xlabel("Cost function \$\\mathcal{C}_{cal}\$", fontsize=14)
            else
                xlabel("Cost function \$\\mathcal{C}_{uncal}\$", fontsize=14)
            end

            xticks(range(fcost_min, fcost_max, length=8)[2:end-1])
        else
            tick_params(bottom=false, labelbottom=false)
        end

    end

    subplots_adjust(top=0.99, bottom=0.07, left=0.16, right=0.99,
                    hspace=0.15, wspace=0.2)

    display(fig)

    if isdir(dirname(path))
       fig.savefig(path)
    elseif path != "none"
       error("Not a directory.")
    end
end


"""
   plot_fp_histo(fp, pixres; pix_lim, path)

plots the histogram distribution of the fitness function values `fp` (taken from
Vachier et al. 2012, A&A). `fp` is expressed in milli arc-seconds so the pixel
resolution `pixres` is need to convert mas to pixels.

Optional argument `pix_lim` represents the upper limit of the x-axis of the 2nd
histogram (default is `5`) and `path` is a `String` specifying the path where
the plot is saved (default is `none`).

"""
function plot_fp_histo(fp::Array{T,1},
                       pixres::T;
                       pix_lim::Union{Int,T}=5,
                       path::String="none") where {T<:AbstractFloat}

    bins = 0:pixres:maximum(fp)
    n = length(fp)

    #figure(figsize=(12,5))
    fig, (ax0, ax1) = subplots(1, 2, gridspec_kw=Dict("width_ratios" => [3,1]))

    #suptitle("\$f_p = \\sum_{i=1}^n \\sqrt{ \\dfrac{1}{2n} \\left( (x_i - x_i^{opt})^2 + (y_i - y_i^{opt})^2 \\right) } \$", fontsize=16)

    ax0.hist(fp, bins, density=false, color="dodgerblue", edgecolor="white", linewidth=1.1)
    ax0.set_xlabel("\$f_p\$ [pix]", fontsize=14)
    ax0.set_ylabel("Counts [%]", fontsize=14)
    lab = range(0, maximum(fp)/pixres, step=10)
    tic = lab*pixres
    ax0.set_xticks(tic)
    ax0.set_xticklabels(round.(Int,lab), fontsize=12)
    ylab = ax0.get_yticks()
    y_lim = round(maximum(ylab)/n*100)
    ylab = Int.(range(0, y_lim+5, step=5))
    ax0.set_yticklabels(string.(ylab).*"%", fontsize=12)
    ax0.set_yticks(ylab/100*n)

    ax1.hist(fp, bins, density=false, color="dodgerblue", edgecolor="white", linewidth=1.1)
    ax1.set_xlabel("\$f_p\$ [pix]", fontsize=14)
    ax1.set_xlim(-pix_lim*pixres*0.08,pix_lim*pixres*1.08)
    lab = range(0, pix_lim, step=1)
    tic = lab*pixres
    ax1.set_xticks(tic)
    ax1.set_xticklabels(lab)
    ax1.tick_params(left=false, labelleft=false)

    tight_layout()

    display(fig)

    if isdir(dirname(path))
       fig.savefig(path)
    elseif path != "none"
       error("Not a directory.")
    end
end

"""
    plot_fcost_vs_fp(fp_origin, fcost, fp, pixres)

plots the distribution of the PACOME cost function `fcost` with respect to the
Root Mean Square Distance `fp` (pixels) computed with
`fp_origin` as origin. The RMSD `fp` in converted in pixel using
the pixel resolution of the detector `pixres`. `path` is a `String` specifying
the path where the plot is saved (default is `none`).

---
    plot_fcost_vs_fp(fp_origin, dat, fcost, fp)

same as above but the pixel resolution is the mean of all pixel resolution
taken from PACOME data `dat`.

"""

function plot_C_vs_RMSD(fp_origin::String,
                          C::Vector{T},
                          RMSD::Vector{T};
                          fsize::Tuple = (6,4),
                          xlims::Tuple = (),
                          ylims::Tuple = (),
                          path::String="none",
                          transp::Bool=false) where {T<:AbstractFloat}

    min_C, max_C = minimum(C), maximum(C)
    min_RMSD, max_RMSD = minimum(RMSD), maximum(RMSD)

    fig, ax = subplots(figsize=fsize)

    scatter(RMSD, C, s=0.5, alpha=0.9, color="dodgerblue", zorder=1)
    # plot([0, 0],[min_C, max_C],
    #      linestyle="--", color="darkorange", linewidth=2.5, zorder=3)

    isempty(ylims) ? ylim(min_C, last(ax.get_ylim())) : ylim(first(ylims),
                                                             last(ylims))
    yscale("log")
    isempty(xlims) ? nothing : xlim(first(xlims), last(xlims))

    # text((0+(max_RMSD-min_RMSD)*0.018), max_C-(max_C-min_C)*0.008,
    #       "Origin :\n$fp_origin", fontweight="bold",
    #       fontsize=13.5, color="darkorange", va="top", ha="left",
    #       bbox=Dict("facecolor"=>"white", "edgecolor"=>"white", "alpha"=>0.35))

    ylabel("Multi-epoch score", fontsize=14)
    # ylabel("Fonction objectif "*L"\mathcal{C}_{\ell=1}", fontsize=14.5)

    #ylabel(L"Fonction objectif $\mathcal{C}_{\lambda}$", fontsize=14.5)
    yticks(fontsize=13)
    xlabel("RMSD to stellar center [pix]", fontsize=14)
    # xlabel(L"Distance quadratique moyenne $d(\mu)$ [pix]", fontsize=14.5)
    xticks(fontsize=13)
    tight_layout()

    if isdir(dirname(path))
       fig.savefig(path, transparent=transp)
    elseif path != "none"
       error("Not a directory.")
    end
end

"""
    plot_discret_corner_plot(orbs; par, path)

plots the corner plot of the 7 orbital elements from the orbits stored in the
`orbs` array (whose size should be 7xN). If parallax `par` is different than 0,
the semi-major is expressed in AU, else in milli-arcsecond. `path` is a `String`
specifying where the plot is saved (default is `none`). If a reference orbit
`orb` is specified in keywords, its orbital elements are plotted on top of the
corner plot.

"""
function plot_discret_corner_plot(orb::Pacome.Orbit{T},
                                  orbs::Array{T,2};
                                  kwds...) where {T<:AbstractFloat}

    plot_discret_corner_plot(orbs; orb=Pacome.orb2arr(orb), kwds...)
end

function plot_discret_corner_plot(orb::Vector{T},
                                  orbs::Array{T,2};
                                  kwds...) where {T<:AbstractFloat}

    plot_discret_corner_plot(orbs; orb=orb, kwds...)
end

function plot_discret_corner_plot(orbs::Array{T,2};
                                  orb::Vector{T}=T[],
                                  par::T=0.,
                                  path::String="none") where {T<:AbstractFloat}

    @assert size(orbs,1) == 7
    !isempty(orb) ? (@assert length(orb) == 7) : nothing
    orbs_blob = deepcopy(orbs)

    c_orb = "white" #"mediumspringgreen" #"lightgreen" "lime"


    μ = ["\$e\$ [-]", "\$i\$ [°]", "\$τ\$ [-]",
         "\$\\omega\$ [°]", "\$\\Omega\$ [°]", "\$K\$ [mas\$^3\$/yr\$^2\$]"]
    mnl = [4, 4, 4, 4, 4, 4]

    if par == 0
        pushfirst!(μ, "\$a\$ [mas]")
        pushfirst!(mnl, 3)
    else
        orbs_blob[1,:] = orbs_blob[1,:]/par
        pushfirst!(μ, "\$a\$ [au]")
        pushfirst!(mnl, 4)
        !isempty(orb) ? orb[1] /= par : nothing
    end

    all_bins = []
    all_vals = []
    for μ in 1:7
         vals = sort(collect(Set(orbs_blob[μ,:])))
         bins = [(vals[i+1]+vals[i])/2 for i in 1:length(vals)-1]
         pushfirst!(bins, (3*vals[1]-vals[2])/2)
         push!(bins, (3*vals[end]-vals[end-1])/2)

         push!(all_bins, bins)
         push!(all_vals, vals)
    end

    nrows, ncols = 7, 7

    fig = figure(figsize=(8.75,8.75))

    for i in 1:nrows
        for j in 1:ncols

            if j > i
                continue
            elseif i == j
                ax = subplot(nrows,ncols,(i-1)*nrows+j)
                n, bins, patches = hist(orbs_blob[i,:], bins=all_bins[i], color="black", histtype="step")
                # patches[findlast(x-> x < orb[i], bins)].set_fc("darkorange")

                if !isempty(orb)
                    idx = findfirst(x->x>=orb[i], bins)
                    plot([orb[i], orb[i]], [0, n[idx-1]],
                          color="black", linestyle="--", linewidth=1)
                end

                xlim(first(bins), last(bins))
                if j != ncols
                    ax.tick_params(labelbottom=false,labelleft=false,left=false)
                else
                    ax.tick_params(labelleft=false,left=false)
                    xlabel("$(μ[j])", fontsize=14)
                end
            else
                ax = subplot(nrows,ncols,(i-1)*nrows+j)
                h, = hist2D(orbs_blob[j,:], orbs_blob[i,:],
                            bins=[all_bins[j], all_bins[i]],
                            cmap="plasma")

                ## for black and white plot (better contrast ?)
                # ax.clear()
                # hmax = maximum(h)
                # hist2D(orbs_blob[j,:], orbs_blob[i,:],
                #             bins=[all_bins[j], all_bins[i]],
                #             cmap="gray_r",
                #             norm = matplotlib.colors.Normalize(vmin=0,
                #                                                vmax=hmax*1.25))

                if !isempty(orb)
                    ylims = ax.get_ylim()
                    plot([orb[j], orb[j]], [first(ylims), last(ylims)],
                         color="white", linestyle="--", linewidth=1)

                    xlims = ax.get_xlim()
                    plot([first(xlims), last(xlims)], [orb[i], orb[i]],
                         color="white", linestyle="--", linewidth=1)
                end

                if j == 1
                    ylabel("$(μ[i])", fontsize=14)
                else
                    ax.tick_params(labelleft=false)
                end

                if i == nrows
                    xlabel("$(μ[j])", fontsize=14)
                else
                    ax.tick_params(labelbottom=false)
                end
            end

            xticks(fontsize=9)
            yticks(fontsize=9)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(mnl[j]))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(mnl[i]))
        end
    end

    tight_layout()

    subplots_adjust(top=0.986,
                    bottom=0.073,
                    left=0.088,
                    right=0.982,
                    hspace=0.1,
                    wspace=0.1)

    if isdir(dirname(path))
       fig.savefig(path, transparent=transp)
    elseif path != "none"
       error("Not a directory.")
    end
end

function plot_orb_elem_corner_plots(orbs::AbstractArray{T,2};
                                    par::T=0.,
                                    nbins::Int=30,
                                    path::String="none") where {T<:AbstractFloat}

    @assert size(orbs,1) == 7
    orbs_blob = deepcopy(orbs)

    Pacome.wrap_orb!(orbs_blob)

    μ = ["\$e\$ [-]", "\$i\$ [°]", "\$τ\$ [-]",
         "\$\\omega\$ [°]", "\$\\Omega\$ [°]", "\$K\$ [mas\$^3\$/yr\$^2\$]"]
    mnl = [4, 4, 4, 4, 4, 4]

    if par == 0
        pushfirst!(μ, "\$a\$ [mas]")
        pushfirst!(mnl, 3)
    else
        orbs_blob[1,:] = orbs_blob[1,:]/par
        pushfirst!(μ, "\$a\$ [au]")
        pushfirst!(mnl, 4)
    end

    nrows, ncols = 7, 7

    fig = figure(figsize=(9,8.75))

    for i in 1:nrows
        for j in 1:ncols

            if j > i
                continue
            elseif i == j
                ax = subplot(nrows,ncols,(i-1)*nrows+j)
                n, bins, patches = hist(orbs_blob[i,:], color="black", histtype="step", bins=nbins)
                xlim(minimum(bins), maximum(bins))

                if j != ncols
                    ax.tick_params(labelbottom=false,labelleft=false,left=false)
                else
                    ax.tick_params(labelleft=false,left=false)
                    xlabel("$(μ[j])", fontsize=14)
                end
            else
                ax = subplot(nrows,ncols,(i-1)*nrows+j)
                # scatter(orbs_blob[j,:], orbs_blob[i,:], marker=",",
                #         color="black", s=1,
                #         alpha=0.005)
                hist2D(orbs_blob[j,:], orbs_blob[i,:], bins=nbins)
                grid(false)

                if j == 1
                    ylabel("$(μ[i])", fontsize=14)
                else
                    ax.tick_params(labelleft=false)
                end

                if i == nrows
                    xlabel("$(μ[j])", fontsize=14)
                else
                    ax.tick_params(labelbottom=false)
                end
            end

            xticks(fontsize=9, rotation=45)
            yticks(fontsize=9, rotation=45)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(mnl[j]))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(mnl[i]))
        end
    end

    tight_layout()

    subplots_adjust(top=0.986,
                    bottom=0.093,
                    left=0.088,
                    right=0.982,
                    hspace=0.1,
                    wspace=0.1)

    if isdir(dirname(path))
       fig.savefig(path)
    elseif path != "none"
       error("Not a directory.")
    end
end

function plot_orb_elem_distribs(orbs::AbstractArray{T,2};
                                par::T=0.,
                                nbins::Int=30,
                                path::String="none") where {T<:AbstractFloat}

    @assert size(orbs,1) == 7
    orbs_blob = deepcopy(orbs)

    Pacome.wrap_orb!(orbs_blob)

    μ = ["\$e\$ [-]", "\$i\$ [°]", "\$τ\$ [-]",
         "\$\\omega\$ [°]", "\$\\Omega\$ [°]", "\$K\$ [mas\$^3\$/yr\$^2\$]"]

    if par == 0
        pushfirst!(μ, "\$a\$ [mas]")
    else
        orbs_blob[1,:] = orbs_blob[1,:]/par
        pushfirst!(μ, "\$a\$ [au]")
    end

    nrows, ncols = 2, 4

    fig = figure(figsize=(12,5))

    for i in 1:7
        ax = subplot(nrows,ncols,i)
        n, bins, patches = hist(orbs_blob[i,:], density="true", bins=nbins,
                                color="black", histtype="step")

        xlabel("$(μ[i])", fontsize=14)
        xticks(fontsize=12)
        yticks(fontsize=12)
        xlim(minimum(bins), maximum(bins))
        ax.tick_params(labelleft=false,left=false)
    end

    tight_layout()

    subplots_adjust(top=0.98,
                    bottom=0.127,
                    left=0.012,
                    right=0.992,
                    hspace=0.381,
                    wspace=0.095)

    if isdir(dirname(path))
       fig.savefig(path, transparent=transp)
    elseif path != "none"
       error("Not a directory.")
    end
end


end # module
