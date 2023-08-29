module Utils

using Dates
using Printf
using EasyFITS
using TwoDimensional
using Measurements
using PyPlot
# using InterpolationKernels

IRDIS = Dict("BB_Y"=> [1.043, 1.043],
         "BB_J"=> [1.245, 1.245],
         "BB_H"=> [1.625, 1.625],
         "BB_Ks"=> [2.182, 2.182],
         "DB_Y23"=> [1.022, 1.076],
         "DB_J23"=> [1.190, 1.273],
         "DB_H23"=> [1.593, 1.667],
         "DB_ND-H23"=> [1.593, 1.667],
         "DB_H34"=> [1.667, 1.733],
         "DB_K12"=> [2.110, 2.251],
         "NB_HeI"=> [1.085, 1.085],
         "NB_CntJ"=> [1.213, 1.213],
         "NB_PaB"=> [1.283, 1.283],
         "NB_CntH"=> [1.573, 1.573],
         "NB_FeII"=> [1.642, 1.642],
         "NB_CntK1"=> [2.091, 2.091],
         "NB_H2"=> [2.124, 2.124],
         "NB_BrG"=> [2.170, 2.170],
         "NB_CntK2"=> [2.266, 2.266],
         "NB_CO"=> [2.290, 2.290])

IFS = Dict("H"=>[0.957478, 0.971862, 0.986779, 1.0022, 1.0181, 1.03444,
               1.0512, 1.06834, 1.08585, 1.10369, 1.12183, 1.14025,
               1.15891, 1.17779, 1.19685, 1.21607, 1.23542, 1.25488,
               1.2744, 1.29397, 1.31355, 1.33312, 1.35265, 1.3721,
               1.39146, 1.41068, 1.42975, 1.44863, 1.46729, 1.48571,
               1.50386, 1.5217, 1.53922, 1.55637, 1.57314, 1.58949,
               1.60539, 1.62082, 1.63575],
        "YJ"=>[0.957557, 0.96595, 0.974572, 0.983407, 0.992443,
                1.00167, 1.01107, 1.02064, 1.03036, 1.04021, 1.05019,
                1.06029, 1.07048, 1.08076, 1.09111, 1.10153, 1.11199,
                1.1225, 1.13302, 1.14356, 1.15409, 1.16461, 1.1751,
                1.18555, 1.19595, 1.20628, 1.21654, 1.2267, 1.23676,
                1.2467, 1.25652, 1.26619, 1.2757, 1.28505, 1.29422,
                1.30319, 1.31197, 1.32052, 1.32884])

SPHERE_filters = Dict("IRDIS"=> IRDIS, "IFS"=> IFS)

"""
    SPHERE_COR_DIAM

encodes the diameter of several infra-red coronagraphic masks (in mas).

"""
SPHERE_COR_DIAM = Dict("N_ALC_YJH_S" => 185.,
                       "N_ALC_YJH_L" => 240.,
                       "N_ALC_YJ_S"  => 145.,
                       "N_ALS_YJ_L"  => 185.,
                       "N_ALC_Ks"    => 240.)

"""
    guess_paco_hduname(path) -> hduname

yields FITS Header Data Unit (HDU) name guessed from the name of the PACO file
`path`.

"""
function guess_paco_hduname(path::AbstractString)
    name = basename(path)
    if findfirst("paco1_denom_aligned", name) !== nothing
        return "PACO_DENOM_ALIGNED"
    elseif findfirst("paco1_num_aligned", name) !== nothing
        return "PACO_NUM_ALIGNED"
    elseif findfirst("paco1_tr_denom_aligned", name) !== nothing
        return "PACO_ROBUST_DENOM_ALIGNED"
    elseif findfirst("paco1_tr_num_aligned", name) !== nothing
        return "PACO_ROBUST_NUM_ALIGNED"
    elseif findfirst("paco2_tr_sr_ww_num_aligned_wwms", name) !== nothing
        return "PACO2_ROBUST_WW_NUM_ALIGNED"
    elseif findfirst("paco2_tr_sr_ww_denom_aligned_wwms", name) !== nothing
        return "PACO2_ROBUST_WW_DENOM_ALIGNED"
    elseif findfirst("paco2_tr_ww_num_aligned_wwms", name) !== nothing
        return "PACO2_ROBUST_WW_NUM_ALIGNED"
    elseif findfirst("paco2_tr_ww_denom_aligned_wwms", name) !== nothing
        return "PACO2_ROBUST_WW_DENOM_ALIGNED"
    elseif findfirst("paco2_tr_sr_wwm_num_aligned_wwms", name) !== nothing
        return "PACO2_ROBUST_WWM_NUM_ALIGNED"
    elseif findfirst("paco2_tr_sr_wwm_denom_aligned_wwms", name) !== nothing
        return "PACO2_ROBUST_WWM_DENOM_ALIGNED"
    else
        error("unknown PACO file \"$name\"")
    end
end

"""
    rewrite_paco_file(src, dst)

updates header of old PACO file `src` and save it in `dst`.

Keyword `hduname` is to specify the name of the FITS Header Data Unit (HDU); if
unspecified, [`guess_paco_hduname`](@ref) is called to guess the HDU name from
`src`.

Other keywords are passed to the `write` method.

"""
function rewrite_paco_file(src::AbstractString,
                           dst::AbstractString;
                           hduname::String = guess_paco_hduname(src),
                           kwds...)
    write(dst, fix_paco_image!(read(FitsImage, src), hduname); kwds...)
end

"""
    fix_paco_image!(A, hduname) -> A

updates header stored in FITS Image `A` which is assumed to be one of the
possible outputs of PACO.  Argument `hduname` is the name of the FITS Header
Data Unit (HDU).

"""
function fix_paco_image!(A::FitsImage, hduname::String)
    @assert ndims(A) ≥ 2
    scale = A["PIXTOARC"] # pixel to milliarcseconds factor
    units = "mas"
    n1, n2, = size(A)
    A["HDUNAME"] = (hduname, "contents")
    A["CTYPE1"] = ("X", "name of 1st axis (delta-RA)")
    A["CUNIT1"] = (units, "coordinate units along 1st axis")
    A["CRPIX1"] = ((1 + n1)/2, "index of reference pixel along 1st axis")
    A["CRVAL1"] = (0.0, "coordinate of reference pixel along 1st axis")
    A["CDELT1"] = (-scale, "coordinate increment along 1st axis")
    A["CTYPE2"] = ("Y", "name of 2nd axis (delta-Dec)")
    A["CUNIT2"] = (units, "coordinate units along 2nd axis")
    A["CRPIX2"] = ((1 + n2)/2, "index of reference pixel along 2nd axis")
    A["CRVAL2"] = (0.0, "coordinate of reference pixel along 2nd axis")
    A["CDELT2"] = (+scale, "coordinate increment along 2nd axis")
    return A
end

"""
    central_pixel(A) -> pt::Point{Float64}

yields the position of the central pixel in FITS Image `A`.  The central pixel
has physical coordinates `(0,0)`.

"""
# central_pixel(A::FitsImage) =
#     Point((A["CRPIX1"]::Float64) - (A["CRVAL1"]::Float64)/(A["CDELT1"]::Float64),
#           (A["CRPIX2"]::Float64) - (A["CRVAL2"]::Float64)/(A["CDELT2"]::Float64))
central_pixel(A::Union{FitsImage,FitsHeader}) = begin

   if A["NAXIS1"] == A["NAXIS2"] == 1448
       return Point(725.,725.)
   elseif A["NAXIS1"] == A["NAXIS2"] == 410
       return Point(206.,206.)
   elseif A["NAXIS1"] == A["NAXIS2"] == 1024
       return Point(513.,513.)
   elseif A["NAXIS1"] == A["NAXIS2"] == 290
       return Point(146.,146.)
   else
       error("Image size not recognised...")
   end
end

"""
    mjd_to_epoch(mjd)

yields modified Julian day `mjd` in epoch expressed in years units.

""" mjd_to_epoch

const DAYS_PER_YEAR = 365.24219879

mjd_to_epoch(mjd::Real) =
    # convert MJD to Unix Time (in years) and add origin of Unix Time (1970)
    (mjd - 40587.5)/DAYS_PER_YEAR + 1970

"""
    epoch_to_mjd(epoch)

converts the epoch (in years units) into modified Julian day `mjd`.

"""
epoch_to_mjd(epoch::Real) = (epoch - 1970)*DAYS_PER_YEAR + 40587.5


"""
    pacome_head(version, date, giturl) -> nothing

prints a fancy header message in prompt announcing the start of PACOME computation.

"""

function pacome_head()
    print("\n")
    printstyled("                             _\n", bold=true, color=:red)
    printstyled("                            (_)", bold=true, color=:red) ; printstyled( "--__\n")
    printstyled("                                   `--_    |\n")
    printstyled(" _ __   __ _  ___  ___", bold=true)       ; printstyled("  _   _  ___", bold=true, color=:blue) ; printstyled("     \\   |");   printstyled("  This is the PACO Multi-Epoch algorithm.\n", bold=true)
    printstyled("| `_ \\ / _` |/ __|/ _ \\", bold=true)    ; printstyled("| `-' |/ __)", bold=true, color=:blue) ; printstyled("     )  |\n")
    printstyled("| |_) | (_| | (__( (_) ", bold=true)      ; printstyled("| |-| |  _]", bold=true, color=:blue) ; printstyled("    _/   |"); printstyled("  Version 1.0 (2020-12-05)\n", bold=true)
    printstyled("|  __/ \\__'_|\\___|\\___/", bold=true)   ; printstyled("|_| |_|\\___)", bold=true, color=:blue) ; printstyled(" _/     |"); printstyled("  Official Git: ", bold=true); printstyled("https://git-cral.univ-lyon1.fr/julesdallant/pacome\n", bold=true, color=:blue)
    printstyled("| | ", bold=true) ; printstyled("                        ____---'       |\n")
    printstyled("|_| ", bold=true) ; printstyled("                ____---'               |\n")
    print("\n\n")
    return nothing
end

function pacome_head2(; version::String, date::String, giturl::String)
    print("\n\n")
    print(" _ __   __ _  ___  ___  _   _  ___   |\n")
    print("| `_ \\ / _` |/ __|/ _ \\| `-' |/ __)  |  This is the PACO Multi-Epoch algorithm.\n")
    print("| |_) | (_| | (__( (_) | |-| |  _]   |\n")
    @printf("|  __/ \\__'_|\\___|\\___/|_| |_|\\___)  |  Version %s (%s)\n", version, date)
    @printf("| |                                  |  Official Git: %s\n", giturl)
    print("|_|                                  |\n")
    print("\n\n")
    return nothing
end

"""
    formatTimeSeconds(t) -> t_str

transforms the time `t` given in seconds to the following format
`t_str`=`YY:DD:HH:MM:SS`.

"""
function formatTimeSeconds(t::Real)

    t_m, t_s = divrem(t, 60)
    t_h, t_m = divrem(t_m, 60)
    t_d, t_h = divrem(t_h, 24)

    t_s = lpad(round(Int,t_s), 2, "0")
    t_m = lpad(round(Int,t_m), 2, "0")
    t_h = lpad(round(Int,t_h), 2, "0")
    t_d = lpad(round(Int,t_d), 2, "0")

    if parse(Int,t_d) > 0
        t_str = "$(t_d)d:$(t_h)h:$(t_m)m:$(t_s)s"
    else
        t_str = "$(t_h)h:$(t_m)m:$(t_s)s"
    end

    return t_str
end

"""
    estimated_time_for_computation(nbr_orbits) -> time

returns the estimated time for computing the combined PACOME SNRs of
the `nbr_orbits` orbits and returns the string result `time`.

"""
function estimated_time_for_computation(nbr_orbits::Real)
    t_elapsed_s = Int(round(nbr_orbits*126/(8.32E+06)))
    return formatTimeSeconds(t_elapsed_s)
end


function yearfraction(d::Date)
    return year(d) + dayofyear(d)/daysinyear(d)
end

function yearfraction(d::String)
    return yearfraction(Date(d))
end

function yearfraction_to_iso8061(d::AbstractFloat; format::String="short")
    y = floor(Int,d)
    days = floor(Int,(d-y)*daysinyear(d))
    if format == "short"
        D = Dates.Date(y) + Dates.Day(days)
    elseif format == "long"
        h = floor(Int,((d-y)*daysinyear(d) - days) * 24)
        m = floor(Int, (((d-y)*daysinyear(d) - days) * 24 - h)*60)
        s = floor(Int, (((d-y)*daysinyear(d) - days) * 24 - h)*60 - m)
        D = Dates.Date(y) + Dates.Day(days) + Time(h,m,s)
    end

    return string(D)
end

function save_outliers(outliers::Vector{String}, savePath::String)

    if isdir(dirname(savePath))
        open(joinpath(savePath,"outliers.txt"), "w") do io
            write(io, "outliers = $outliers")
        end;
        return nothing
    else
        return nothing
    end
end

function save_outliers(outliers::Vector{Any}, savePath::String)

    if isdir(dirname(savePath))
        open(joinpath(savePath,"outliers.txt"), "w") do io
            write(io, "outliers = None")
        end;
        return nothing
    else
        return nothing
    end
end

function save_orbits_LaTeX(path::String, μ::Array{T,1}, σ::Array{T,1};
                           kwds...) where {T<:AbstractFloat}
   save_orbits_LaTeX(path, reshape(μ, (7,1)), reshape(σ, (7,1)); kwds...)
end

function save_orbits_LaTeX(path::String, μ::Array{T,2}, σ::Array{T,2};
                           dig::Int=2, nam::Vector{String}=Vector{String}(),
                           fontsize::String="small") where {T<:AbstractFloat}

   @assert size(μ,1) == size(σ,1) == 7
   norb = size(μ,2)
   isempty(nam) ? nam = [string(i) for i in 1:norb] : nothing
   @assert norb == size(σ,2) == length(nam)

   fontsize in ["tiny", "scriptsize", "small", "normal", "large",
                "Large"] ? nothing : error("Fontsize not recognized...")

   param = [L"a"*" [mas]", L"e"*" [-]", L"i"*" [deg]", L"\tau"*" [-]",
            L"\omega"*" [deg]", L"\Omega"*" [deg]", L"K"*" ["*
            L"\text{mas}^3/\text{yr}^2"*"]", L"P"*" [yr]"]

   P_val, P_err  = Vector{T}(undef, norb), Vector{T}(undef, norb)
   for i in 1:norb
       P = sqrt(measurement(μ[1,i],σ[1,i])^3/measurement(μ[7,i],σ[7,i]))
       P_val[i] = P.val
       P_err[i] = P.err
   end

   μ = round.([μ; P_val'], digits=dig)
   σ = round.([σ; P_err'], digits=dig)

   open(path, "w") do io
       write(io, "\\begin{table}[bt]\n")
       write(io, "    \\"*fontsize*"\n")
       write(io, "    \\centering\n")
       write(io, "    \\begin{tabular}{c|"*"c"^norb*"}\n")

       write(io, "    \\hline")
       write(io, "    \\textbf{Param.}")
       for i in 1:norb
           write(io, " & \\textbf{Planet \\textit{"*nam[i]*"}}")
       end
       write(io, " \\\\ \\hline\n")

       for i in 1:8
           write(io, "    " * param[i])
           for j in 1:norb
               write(io, " & \$ "*string(μ[i,j])*" \\pm "*string(σ[i,j])*" \$")
           end
           write(io, " \\\\ \n")
       end
       write(io, "    \\hline")

       write(io, "    \\end{tabular}\n")
       write(io, "    \\caption{}\n")
       write(io, "    \\label{tab:}\n")
       write(io, "\\end{table}\n")
   end
end

function save_orbits_LaTeX(path::String, μ::Array{T,2}, σ::Array{T,2},
                           C::Vector{T}; dig::Int=2,
                           nam::Vector{String}=Vector{String}(),
                           fontsize::String="small",
                           cal::Bool=false) where {T<:AbstractFloat}

   @assert size(μ,1) == size(σ,1) == 7
   norb = size(μ,2)
   isempty(nam) ? nam = [string(i) for i in 1:norb] : nothing
   @assert norb == size(σ,2) == length(nam) == length(C)

   fontsize in ["tiny", "scriptsize", "small", "normalsize", "large",
                "Large"] ? nothing : error("Fontsize not recognized...")

   if cal
       Cnam = L"\mathcal{C}^{\text{cal}}"
   else
       Cnam = L"\mathcal{C}^{\text{uncal}}"
   end

   param = [L"a"*" [mas]", L"e"*" [-]", L"i"*" [deg]", L"\tau"*" [-]",
            L"\omega"*" [deg]", L"\Omega"*" [deg]", L"K"*" ["*
            L"\text{mas}^3/\text{yr}^2"*"]", L"P"*" [yr]"]

   P_val, P_err  = Vector{T}(undef, norb), Vector{T}(undef, norb)
   for i in 1:norb
       P = sqrt(measurement(μ[1,i],σ[1,i])^3/measurement(μ[7,i],σ[7,i]))
       P_val[i] = P.val
       P_err[i] = P.err
   end

   μ = round.([μ; P_val'], digits=dig)
   σ = round.([σ; P_err'], digits=dig)
   C = round.(C, digits=dig)

   open(path, "w") do io
       write(io, "\\begin{table}[bt]\n")
       write(io, "    \\"*fontsize*"\n")
       write(io, "    \\centering\n")
       write(io, "    \\begin{tabular}{c|"*"c"^norb*"}\n")

       write(io, "    \\hline")
       write(io, "    \\textbf{Param.}")
       for i in 1:norb
           write(io, " & \\textbf{Planet \\textit{"*nam[i]*"}}")
       end
       write(io, " \\\\ \\hline\n")

       for i in 1:8
           write(io, "    " * param[i])
           for j in 1:norb
               write(io, " & \$ "*string(μ[i,j])*" \\pm "*string(σ[i,j])*" \$")
           end
           write(io, " \\\\ \n")
       end
       write(io, "    \\hline\n")

       write(io, "    "*Cnam)
       for j in 1:norb
           write(io, " & "*string(C[j]))
       end
       write(io, "\\\\ \n    \\hline\n")
       write(io, "    \\end{tabular}\n")
       write(io, "    \\caption{}\n")
       write(io, "    \\label{tab:}\n")
       write(io, "\\end{table}\n")
   end
end

function print_orbits_LaTeX(μ0::Array{T,2}, σ0::Array{T,2},
                           C::Vector{T}, Clim::Vector{T};
                           dig::Int=2, dof::String="") where {T<:AbstractFloat}

   μ = deepcopy(μ0)
   σ = deepcopy(σ0)
   @assert size(μ,1) == size(σ,1) == 7
   norb = size(μ,2)
   @assert norb == size(σ,2) == length(C) == length(Clim)
   # param = [L"a"*" [mas]", L"e"*" [-]", L"i"*" [deg]", L"\tau"*" [-]",
   #          L"\omega"*" [deg]", L"\Omega"*" [deg]", L"K"*" ["*
   #          L"\text{mas}^3/\text{yr}^2"*"]", L"P"*" [yr]"]
   pow = floor(Int,log10(maximum(μ[end,:])))

   elem = ["\$a\$", "\$e\$", "\$i\$", "\$\\tau\$",
            "\$\\omega\$", "\$\\Omega\$", "\$K \\, (\\times 10^$pow)\$", "\$ P \$"]
   units = ["mas", "-", "deg", "-", "deg", "deg", "mas\$^3\$/yr\$^2\$", "yr"]

   P_val, P_err  = Vector{T}(undef, norb), Vector{T}(undef, norb)
   for i in 1:norb
       P = sqrt(measurement(μ[1,i],σ[1,i])^3/measurement(μ[7,i],σ[7,i]))
       P_val[i] = P.val
       P_err[i] = P.err
   end

   μ[end,:], σ[end,:] = μ[end,:]/10^pow, σ[end,:]/10^pow

   μ = round.([μ; P_val'], digits=dig)
   σ = round.([σ; P_err'], digits=dig)
   C = round.(C, digits=dig)
   SNR = round.(sqrt.(C), digits=dig)
   Clim = round.(Clim, digits=dig)

   for i in 1:8
       print("    $(elem[i]) & $(units[i]) ")
       for j in 1:norb
           print(" & \$ "*string(μ[i,j])*" \\pm "*string(σ[i,j])*" \$")
       end
       print(" \\\\ \n")
   end
   print("    \\midrule\n")


   print("    \\multicolumn{2}{c||}{\\textbf{Multi-epoch} \$\\SNR\$}")
   for j in 1:norb
       print(" & "*string(SNR[j]))
   end
   print("\\\\\n    \\multicolumn{2}{c||}{\\textbf{Criterion} \$ \\CostFunc \$} ")
   for j in 1:norb
       print(" & "*string(C[j]))
   end
   print(" \\\\\n    \\multicolumn{2}{c||}{\\textbf{Threshold} \$ \\widehat{\\mathcal{Q}}_{$dof}(1-10^{-6}) \$} ")
   for j in 1:norb
       print(" & "*string(Clim[j]))
   end
   print(" \\\\\n")
end


"""
    split_on_threads(nop, nthreads) -> intervs

equally shares the total number of operations `nop` into a number of
intervals equal to the number of threads `nthreads`. The vector `intervs`
contains 2D-tuples indexing the lower and upper bounds of operation number.

"""
function split_on_threads(nop::Int, nthreads::Int)
    op_per_proc = nop ÷ nthreads
    intervs = Vector{Tuple{Int, Int}}(undef, nthreads)
    for k in 1:nthreads
        if k == nthreads
            intervs[k] = (k-1)*op_per_proc+1, nop
        else
            intervs[k] = (k-1)*op_per_proc+1, k*op_per_proc
        end
    end
    return intervs
end

end # module
