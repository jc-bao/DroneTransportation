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

    c_dyn = zeros(eltype(Z), 22 * (params.N - 1))
    # dynamic constrains
    for i = 1:(params.N-1)
        xi = Z[params.idx.x[i]]
        xip1 = Z[params.idx.x[i+1]]
        ui = Z[params.idx.u[i]]
        # dynamics constraints
        c_dyn[22*(i-1).+(1:18)] = hermite_simpson(params, xi, xip1, ui, params.model.dt)
        # rope force constraints
        ui_lift = ui[1:7]
        ui_load = ui[8:10]
        c_dyn[22*(i-1).+(19:21)] = ui_lift[5:7] + ui_load
        # distance constraints
        r_lift = xi[1:3]
        r_load = xi[13:15]
        c_dyn[22*(i-1)+22] = norm(r_lift - r_load)^2 - params.model.l^2
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
    c = zeros(eltype(Z), 5 * (params.N - 1))
    for i = 1:(params.N-1)
        u = Z[params.idx.u[i]]
        u_lift = u[1:7]
        # control limits
        c[5*(i-1).+(1:4)] = u_lift[1:4]
        # state constraints
        x = Z[params.idx.x[i+1]]
        # ellipse_center_dist = sqrt(x[2]^2 + x[3]^2) 
        # c[5*(i-1)+5] = ((ellipse_center_dist - 1.25) / 1.0)^2 + (x[1] / 0.3)^2 - 1
        c[5*(i-1)+5] = x[1]^2 + x[2]^2 + (x[3]-0.3)^2 - 0.5^2

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
    nu_lift = 7
    nu_load = 3
    nu = 7 + 3
    dt = 0.2
    tf = 6.0
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
    R = diagm([0.1 * ones(4); zeros(3); zeros(3)])
    model = (mass=0.5, mass_load=0.5,
        J=Diagonal([0.0023, 0.0023, 0.004]),
        gravity=[0, 0, -9.81],
        L=0.1750, # drone arm length
        l=0.5, # rope length
        kf=1.0, u_max=20.0 / 4.0,
        km=0.0245, dt=dt)
    params = (
        N=N,
        nx=nx,
        nx_lift=nx_lift,
        nx_load=nx_load,
        nu=nu,
        nu_lift=nu_lift,
        nu_load=nu_load,
        Q=Q,
        R=R,
        model=model,
        xi=xi,
        xf=xf,
        idx=idx
    )

    c_l = zeros(5 * (N - 1))
    c_u = repeat([ones(4) .* params.model.u_max; ones(1)*Inf], N - 1)

    # test constrain feasibility
    # test constrains
    x_test = zeros(18)
    x_test[15] = -0.5
    u_lift1_test = ones(4) .* (1.0 * 9.81 / 4.0)
    u_lift2_test = [0.0, 0.0, -0.5 * 9.81]
    u_load_test = [0.0, 0.0, 0.5 * 9.81]
    u_test = [u_lift1_test; u_lift2_test; u_load_test]
    xu_test = [x_test; u_test]
    Z_test = repeat(xu_test, N - 1)
    Z_test = [Z_test; x_test]
    eq_cons = eq_constraints(params, Z_test)

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
            z0[params.idx.u[i]][7] = -0.5 * 9.81
            z0[params.idx.u[i]][10] = 0.5 * 9.81
        end
    end

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

    # return the trajectories 
    xs = [zeros(18) for _ = 1:params.N]
    us = [zeros(10) for _ = 1:(params.N-1)]
    for i = 1:params.N
        x = Z[params.idx.x[i]]
        xs[i] = x[1:18]
        if i < params.N
            us = Z[params.idx.u[i]]
        end
    end

    return xs, us, t_vec, params
end

## solve


@time xs, us, t_vec, params = quadrotor_navigation()
# save xs, us
save("quadrotor_navigation.jld2", "xs", xs, "us", us, "t_vec", t_vec, "params", params)

# visualize

animate_quadrotor_load(xs, xs, params.model.dt)
sleep(3.0)

# plot xs and us
# plot(t_vec, [x[1] for x in xs], label="x")
# plot!(t_vec, [x[2] for x in xs], label="y")
# plot!(t_vec, [x[3] for x in xs], label="z")
# plot!(t_vec[1:end-1], [u[5] for u in us], label="u5")
# plot!(t_vec[1:end-1], [u[6] for u in us], label="u6")
# plot!(t_vec[1:end-1], [u[7] for u in us], label="u7")
# plot!(t_vec[1:end-1], [u[8] for u in us], label="u8")
# plot!(t_vec[1:end-1], [u[9] for u in us], label="u9")
# plot!(t_vec[1:end-1], [u[10] for u in us], label="u10")
# # save to file
# savefig("quadrotor_navigation.png")