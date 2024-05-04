## packages

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

import MathOptInterface as MOI
import Ipopt
import ForwardDiff as FD
import Convex as cvx
import ECOS
using LinearAlgebra
using Plots
using Random
using JLD2
using Test
import MeshCat as mc
using GeometryTypes
using MeshIO
using CoordinateTransformations
using Printf
import lazy_nlp_qd as nlp

include(joinpath(@__DIR__, "utils", "fmincon.jl"))
include(joinpath(@__DIR__, "utils", "quadrotor.jl"))

## cost

function task_cost(params::NamedTuple, Z::Vector)::Real
    # compute the cost 
    xend = Z[params.idx.x[end]]
    J = (xend - params.xf)' * params.Q * (xend - params.xf)
    for k = 1:(params.N-1)
        x = Z[params.idx.x[k]]
        u = Z[params.idx.u[k]]
        J += (x - params.xf)' * params.Q * (x - params.xf) + u' * params.R * u
    end
    return J
end

# equality constraint

function hermite_simpson(params::NamedTuple, x1::Vector, x2::Vector, u, dt::Real)::Vector
    # TODO: input hermite simpson implicit integrator residual 
    dynamics = combined_dynamics
    x_mid =
        0.5 * (x1 + x2) +
        0.125 * dt * (dynamics(params.model, x1, u) - dynamics(params.model, x2, u))
    return x1 +
           1 / 6 *
           dt *
           (
               dynamics(params.model, x1, u) +
               4 * dynamics(params.model, x_mid, u) +
               dynamics(params.model, x2, u)
           ) - x2
end

function eq_constraints(params::NamedTuple, Z::Vector)::Vector

    c_dyn = zeros(eltype(Z), 19 * (params.N - 1))
    # dynamic constrains
    for i = 1:(params.N-1)
        xi = Z[params.idx.x[i]]
        xip1 = Z[params.idx.x[i+1]]
        ui = Z[params.idx.u[i]]
        # dynamics constraints
        c_dyn[19*(i-1).+(1:18)] = hermite_simpson(params, xi, xip1, ui, params.model.dt)
        # distance constraints
        r_lift = xi[1:3]
        r_load = xi[13:15]
        c_dyn[19*(i-1)+19] = norm(r_lift - r_load)^2 - params.model.l^2
    end

    # initial condition
    x1 = Z[params.idx.x[1]]
    c_init = x1 - params.xi

    # final condition
    xf = Z[params.idx.x[end]]
    c_end = xf - params.xf

    return [c_init; c_end; c_dyn]
end

## inequality constraint

function ineq_constraints(params::NamedTuple, Z::Vector)::Vector
    c = []
    for i = 1:(params.N-1)
        u = Z[params.idx.u[i]]
        # control limits
        c = [c; u[1:4]]
        # state constraints
        x = Z[params.idx.x[i+1]]
        for pos_obs in params.obs
            c = [c; norm(x[1:2:3]-pos_obs[1:2:3])^2 - params.r_obs^2]
            c = [c; norm(x[13:2:15]-pos_obs[1:2:3])^2 - params.r_obs^2]
        end
    end
    return c
end

## task setup

function create_idx(nx, nu, N)
    # This function creates some useful indexing tools for Z 

    # our Z vector is [xi, u0, x1, u1, …, xN]
    nz = (N - 1) * nu + N * nx # length of Z 
    x = [(i - 1) * (nx + nu) .+ (1:nx) for i = 1:N]
    u = [(i - 1) * (nx + nu) .+ ((nx+1):(nx+nu)) for i = 1:(N-1)]

    # constraint indexing for the (N-1) dynamics constraints when stacked up
    c = [(i - 1) * (nx) .+ (1:nx) for i = 1:(N-1)]
    nc = (N - 1) * nx # (N-1)*nx 

    return (nx=nx, nu=nu, N=N, nz=nz, nc=nc, x=x, u=u, c=c)
end

"""
    quadrotor navigation

Function for returning collision free trajectories for 3 quadrotors. 

Outputs:
    x::Vector{Vector}  # state trajectory for quad
    u::Vector{Vector}  # control trajectory for quad
    t_vec::Vector
    params::NamedTuple

The resulting trajectories should have dt=0.2, tf = 5.0, N = 26
where all the x's are length 26, and the u's are length 25. 

"""
function quadrotor_navigation(; verbose=true)

    # problem size 
    nx_lift = 12
    nx_load = 6
    nx = 12 + 6
    nu = 4 + 1
    dt = 0.1
    tf = 2.0
    t_vec = 0:dt:tf
    N = length(t_vec)

    # indexing 
    idx = create_idx(nx, nu, N)

    # initial conditions and goal states 
    xi = zeros(18)
    xi[1] = -1.0
    xi[13] = -1.0
    xi[15] = -0.5
    xf = zeros(18)
    xf[1] = 1.0
    xf[13] = 1.0
    xf[15] = -0.5

    # load all useful things into params 
    Q = 1.0 * diagm([1.0 * ones(3); 0.1 * ones(3); 1.0 * ones(3); 1.0 * ones(3); 1.0 * ones(3); 0.1 * ones(3)])
    R = diagm([0.1 * ones(4); zeros(1)])
    model = (mass=0.5, mass_load=0.5,
        J=Diagonal([0.0023, 0.0023, 0.004]),
        gravity=[0, 0, -9.81],
        L=0.1750, # drone arm length
        l=0.5, # rope length
        kf=1.0, u_max=30.0 / 4.0,
        km=0.0245, dt=dt)
    params = (
        N=N,
        nx=nx,
        nx_lift=nx_lift,
        nx_load=nx_load,
        nu=nu,
        Q=Q,
        R=R,
        model=model,
        xi=xi,
        xf=xf,
        idx=idx, 
        r_obs=0.5, 
        obs=[[0.0, 0.0, 0.5], [0.0, 0.0, -0.6]]
    )

    c_l = zeros((4+2*2) * (N - 1))
    c_u = repeat([ones(4) .* params.model.u_max; ones(2*2).*Inf], N - 1)

    # test constrain feasibility
    # test constrains
    x_test = zeros(18)
    x_test[15] = -0.5
    u_lift_test = ones(4) .* (1.0 * 9.81 / 4.0)
    u_rope_test = [0.5 * 9.81]
    u_test = [u_lift_test; u_rope_test]
    xu_test = [x_test; u_test]
    Z_test = repeat(xu_test, N - 1)
    Z_test = [Z_test; x_test]

    @test all(eq_constraints(params, Z_test)[37:end] .≈ 0.0)
    @test all(ineq_constraints(params, Z_test) ≤ c_u)
    @test all(ineq_constraints(params, Z_test) ≥ c_l)

    x_guess = range(params.xi, params.xf, length=params.N)
    z0 = zeros(params.idx.nz)
    for i = 1:params.N
        z0[params.idx.x[i][1]] = x_guess[i][1]
        z0[params.idx.x[i][3]] = x_guess[i][3]
        z0[params.idx.x[i][13]] = x_guess[i][13]
        z0[params.idx.x[i][15]] = x_guess[i][15]
        if i < params.N
            z0[params.idx.u[i]][1:4] = ones(4) .* (1.0 * 9.81 / 4.0)
            z0[params.idx.u[i]][5] = 0.5 * 9.81
        end
    end

    # load file .jld2 as initial guess
    z0 = load("Z.jld2")["Z"]

    diff_type = :auto

    Z = fmincon(
        task_cost,
        eq_constraints,
        ineq_constraints,
        ones(params.idx.nz) .* -Inf,
        ones(params.idx.nz) .* Inf,
        c_l,
        c_u,
        z0,
        params,
        diff_type;
        tol=1e-5,
        c_tol=1e-5,
        max_iters=1_000,
        verbose=verbose
    )

    # save z to file
    save("Z.jld2", "Z", Z)

    # return the trajectories 
    xs = [zeros(18) for _ = 1:params.N]
    us = [zeros(5) for _ = 1:(params.N-1)]
    for i = 1:params.N
        x = Z[params.idx.x[i]]
        xs[i] = x[1:18]
        if i < params.N
            us[i] = Z[params.idx.u[i]]
        end
    end

    return xs, us, t_vec, params
end

## solve


@time xs, us, t_vec, params = quadrotor_navigation()
# save xs, us
# save("quadrotor_navigation.jld2", "xs", xs, "us", us, "t_vec", t_vec, "params", params)

# visualize

animate_quadrotor_load(xs, xs, params.model.dt)
sleep(3.0)

# plot xs and us
# plot(t_vec, [x[1] for x in xs], label="x")
# plot!(t_vec, [x[2] for x in xs], label="y")
# plot!(t_vec, [x[3] for x in xs], label=`"z")
# plot!(t_vec[1:end-1], [u[5] for u in us], label="u5")
# plot!(t_vec[1:end-1], [u[6] for u in us], label="u6")
# plot!(t_vec[1:end-1], [u[7] for u in us], label="u7")
# plot!(t_vec[1:end-1], [u[8] for u in us], label="u8")
# plot!(t_vec[1:end-1], [u[9] for u in us], labl="u9")
# plot!(t_vec[1:end-1], [u[10] for u in us], label="u10")
# # save to file
# savefig("quadrotor_navigation.png")