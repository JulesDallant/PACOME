cd("/Users/julesdallant/Documents/PhD/PACOME/old_PACOME_ab_based/PACOME")
pop!(LOAD_PATH)
push!(LOAD_PATH, "/Users/julesdallant/Documents/PhD/PACOME/old_PACOME_ab_based/PACOME/src")

using Revise, Pacome
using Revise, Orbits
using Revise, Display
using Revise, Utils
using Revise, Simus

using InterpolationKernels
using BenchmarkTools
using Mmap

## SETUP

# Loading multi-epoch dataset reduced with PACO
data = Pacome.PacomeData{Float64}("./data/*paco_outputs", []; mode="adi")
nt, nλ = length(data), data.dims[3]

# Interpolation kernel
ker = CatmullRomSpline{Float64}()

# Minimum score above which orbits are saved
fcost_lim = Simus.model_conflevel_SIMU_DATA(1e-7)

# Orbital elements to explore
Nnodes = 12
a = LinRange(800., 1100., Nnodes)
e = LinRange(0., 0.95, Nnodes)
i = LinRange(0., 180., Nnodes)
τ = LinRange(0., 1., Nnodes)
ω = LinRange(0., 360., Nnodes)
Ω = LinRange(0., 360., Nnodes)
K = LinRange(107_666.7, 122_359.5, Nnodes)
G = Orbits.Grid{Float64}(; a, e, i, τ, ω, Ω, K)

lb = [G.a[1], 0., -Inf, -Inf, -Inf, -Inf, G.K[1]]
ub = [G.a[2], 0.999, +Inf, +Inf, +Inf, +Inf, G.K[2]]

# Mmap file where all orbits (and scores) are saved
mmap_file = "./test/saves/mmap_orbs.bin"

# Maximum number of orbits to optimize
Nopt = 200

## RUNNING PACOME
println("Starting search grid...")
ΔT = @elapsed res = Pacome.PACOME_MT_mmap(data, G, ker,
                                          Threads.nthreads(),
                                          mmap_file;
                                          fcost_lim)

println("Computation time = $(Utils.formatTimeSeconds(ΔT))")

bestcost, bestorb = res[1], res[2:end]

## OPTIMIZATION OF THE BEST ON-GRID ORBITS

s = open(mmap_file)
m = read(s, Int)
n = read(s, Int)
all_orbs = Mmap.mmap(s, Matrix{Float64}, (m,n))
close(s)

Nrep = min(Nopt, size(all_orbs, 2))
optorbs = Array{Float64}(undef, (7, Nrep))
optcosts = Vector{Float64}(undef, Nrep)
cpt = Threads.Atomic{Int}(0)
Threads.@threads for n in 1:Nrep
   orb = collect(Orbits.orb_comb(Int(all_orbs[2,n]), G))
   optorbs[:,n], optcosts[n] = Pacome.optimize_orb_param(data,
                                                         orb,
                                                         ker,
                                                         (lb,ub);
                                                         maxeval=50_000)
   Threads.atomic_add!(cpt, 1)
   print("Ns : $(cpt[])/$Nrep\r")
end

idx = sortperm(optcosts, rev=true)
optcosts = optcosts[idx]
optorbs = optorbs[:,idx]

Display.plot_individual_snrs(data, Pacome.arr2orb(optorb[:,1]), 20)
