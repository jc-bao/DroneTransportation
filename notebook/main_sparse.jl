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
using SparseArrays
using DiffResults
using Rotations
import lazy_nlp_qd as nlp

include(joinpath(@__DIR__, "utils", "fmincon.jl"))
include(joinpath(@__DIR__, "utils", "quadrotor.jl"))

## cost

function cost(params::NamedTuple, Z::Vector)::Real
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

function cost_gradient!(params::NamedTuple, grad::Vector, Z::Vector)
    grad .= FD.gradient(_Z -> cost(params, _Z), Z)
    return nothing
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

## converted constrains for sparse problem

function constraint!(params, cval, Z)
    # equality constraints
    for i = 1:(params.N-1)
        xi = Z[params.idx.x[i]]
        xip1 = Z[params.idx.x[i+1]]
        ui = Z[params.idx.u[i]]
        # dynamics constraints
        cval[params.idx.c_dyn[i]] .= hermite_simpson(params, xi, xip1, ui, params.model.dt)
    end

    for i = 2:(params.N-1)
        xi = Z[params.idx.x[i]]
        r_lift = xi[1:3]
        r_load = xi[13:15]
        cval[params.idx.c_dist[i-1]] = (norm(r_lift - r_load)^2 - params.model.l^2)
    end

    for i = 1:params.nkey
        idx_rlift_i = params.idx.x[params.keyframe_t_lift[i]][1:3]
        r_lift = Z[idx_rlift_i]
        cval[params.idx.c_frame_lift[i]] = norm(r_lift - params.keyframe_r_lift[i])^2 - params.frame_err^2
        idx_rload_i = params.idx.x[params.keyframe_t_load[i]][13:15]
        r_load = Z[idx_rload_i]
        cval[params.idx.c_frame_load[i]] = norm(r_load - params.keyframe_r_load[i])^2 - params.frame_err^2
    end

    return nothing
end

function constraint_jacobian!(params::NamedTuple, conjac::SparseMatrixCSC, Z::Vector)
    # dynamic constrains
    for i = 1:(params.N-1)
        xi = Z[params.idx.x[i]]
        xip1 = Z[params.idx.x[i+1]]
        ui = Z[params.idx.u[i]]
        # dynamics constraints
        conjac[params.idx.c_dyn[i], params.idx.x[i]] .= FD.jacobian(_xi -> hermite_simpson(params, _xi, xip1, ui, params.model.dt), xi)
        conjac[params.idx.c_dyn[i], params.idx.x[i+1]] .= FD.jacobian(_xip1 -> hermite_simpson(params, xi, _xip1, ui, params.model.dt), xip1)
        conjac[params.idx.c_dyn[i], params.idx.u[i]] .= FD.jacobian(_ui -> hermite_simpson(params, xi, xip1, _ui, params.model.dt), ui)
    end
    # distance constraints
    for i = 2:(params.N-1)
        xi = Z[params.idx.x[i]]
        r_lift = xi[1:3]
        r_load = xi[13:15]
        conjac[params.idx.c_dist[i-1], params.idx.x[i][1:3]] .= 2 * (r_lift - r_load)
        conjac[params.idx.c_dist[i-1], params.idx.x[i][13:15]] .= 2 * (r_load - r_lift)
    end
    # frame lift constraints
    for i = 1:params.nkey
        idx_rlift_i = params.idx.x[params.keyframe_t_lift[i]][1:3]
        r_lift = Z[idx_rlift_i]
        conjac[params.idx.c_frame_lift[i], idx_rlift_i] .= 2 * (r_lift - params.keyframe_r_lift[i])
        idx_rload_i = params.idx.x[params.keyframe_t_load[i]][13:15]
        r_load = Z[idx_rload_i]
        conjac[params.idx.c_frame_load[i], idx_rload_i] .= 2 * (r_load - params.keyframe_r_load[i])
    end
    return nothing
end

function create_conjac(params::NamedTuple)
    conjac = spzeros(params.idx.nc, params.idx.nz)
    for i = 1:(params.N-1)
        conjac[params.idx.c_dyn[i], params.idx.x[i]] .= 1.0
        conjac[params.idx.c_dyn[i], params.idx.x[i+1]] .= 1.0
        conjac[params.idx.c_dyn[i], params.idx.u[i]] .= 1.0
    end
    for i = 2:(params.N-1)
        conjac[params.idx.c_dist[i-1], params.idx.x[i][1:3]] .= 1.0
        conjac[params.idx.c_dist[i-1], params.idx.x[i][13:15]] .= 1.0
    end
    for i = 1:params.nkey
        idx_rlift_i = params.idx.x[params.keyframe_t_lift[i]][1:3]
        conjac[params.idx.c_frame_lift[i], idx_rlift_i] .= 1.0
        idx_rload_i = params.idx.x[params.keyframe_t_load[i]][13:15]   
        conjac[params.idx.c_frame_load[i], idx_rload_i] .= 1.0
    end
    return conjac
end

## task setup

function create_idx(nx, nu, nkey, N)
    # This function creates some useful indexing tools for Z 

    # our Z vector is [xi, u0, x1, u1, …, xN]
    nz = (N - 1) * nu + N * nx # length of Z 
    x = [(i - 1) * (nx + nu) .+ (1:nx) for i = 1:N]
    u = [(i - 1) * (nx + nu) .+ ((nx+1):(nx+nu)) for i = 1:(N-1)]

    # constraint indexing for the (N-1) dynamics constraints when stacked up
    c_dyn = [(i - 1) * (nx) .+ (1:nx) for i = 1:(N-1)]
    c_dist = (1:(N-2)) .+ nx*(N-1)
    c_frame_lift = (1:nkey) .+ nx*(N-1) .+ (N-2)
    c_frame_load = (1:nkey) .+ nx*(N-1) .+ nkey .+ (N-2)
    nc = nx * (N - 1) + N - 2 + nkey * 2

    return (nx=nx, nu=nu, N=N, nz=nz, nc=nc, x=x, u=u, c_dyn=c_dyn, c_dist=c_dist, c_frame_lift=c_frame_lift, c_frame_load=c_frame_load)
end

function quadrotor_navigation(; verbose=true)

    # problem size 
    nx_lift = 12
    nx_load = 6
    nx = 12 + 6
    nu = 4 + 1
    dt = 0.05
    tf = 7.0
    rad_traj = 2.0
    t_vec = 0:dt:tf
    N = length(t_vec)
    keyframe_r_lift = []
    keyframe_t_lift = []
    nkey = 5
    for i = 1:nkey
        theta = 2 * pi * i / (nkey + 1)
        r = [rad_traj * cos(theta), rad_traj * sin(theta), 0.0]
        push!(keyframe_r_lift, r)
        push!(keyframe_t_lift, Int(div(tf * i / (nkey + 1), dt)))
    end
    keyframe_r_load = keyframe_r_lift
    # keyframe_t_lift = [Int(div(tf/3, dt)), Int(div(tf/3*2, dt))]
    dt_obj_drone = 0.2
    dstep = Int(div(dt_obj_drone, dt))
    # keyframe_t_load = keyframe_t_lift .+ dstep
    keyframe_t_load = zeros(Int, nkey)
    for i = 1:nkey
        if i % 2 == 0
            keyframe_t_load[i] = keyframe_t_lift[i] - dstep
        else
            keyframe_t_load[i] = keyframe_t_lift[i] + dstep
        end
    end
    frame_err = 0.1

    # indexing 
    idx = create_idx(nx, nu, nkey, N)

    # initial conditions and goal states 
    xi = zeros(nx)
    xi[1] = rad_traj
    xi[13] = rad_traj
    xi[15] = -0.5
    xf = xi
    # xf = zeros(18)
    # xf[1] = dist
    # xf[13] = dist
    # xf[15] = -0.5

    # load all useful things into params 
    Q = diagm([0.0 * ones(3); 0.1 * ones(3); [0.1, 0.1, 1.0]; 0.1 * ones(3); 0.0 * ones(3); 0.1 * ones(3)])
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
        keyframe_r_lift = keyframe_r_lift,
        keyframe_r_load = keyframe_r_load,
        keyframe_t_lift = keyframe_t_lift,
        keyframe_t_load = keyframe_t_load,
        frame_err = frame_err,
        nkey = nkey,
        # r_obs=0.5,
        # obs=[[0.0, 0.0, 0.5], [0.0, 0.0, -0.6]]
    )

    # primal bounds
    x_min = -Inf * ones(nx)
    x_max = Inf * ones(nx)
    u_min = [ones(4) .* -model.u_max; 0.0]
    u_max = [ones(4) .* model.u_max; Inf]
    Z_l = -Inf * ones(idx.nz)
    Z_u = Inf * ones(idx.nz)
    Z_l[idx.x[1]] .= xi
    Z_u[idx.x[1]] .= xi
    for i = 1:(N-1)
        Z_l[idx.u[i]] .= u_min
        Z_u[idx.u[i]] .= u_max
    end
    for i = 2:(N-1)
        Z_l[idx.x[i]] .= x_min
        Z_u[idx.x[i]] .= x_max
    end
    Z_l[idx.x[N]] .= xf
    Z_u[idx.x[N]] .= xf

    # constrains bounds
    c_l = zeros(idx.nc)
    c_u = zeros(idx.nc)

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

    con = zeros(idx.nc)
    constraint!(params, con, Z_test)
    result = DiffResults.JacobianResult(con, Z_test)
    FD.jacobian!(result, (_con, _Z) -> constraint!(params, _con, _Z), con, Z_test)
    conjac_FD = DiffResults.jacobian(result)
    conjac_test = spzeros(idx.nc, idx.nz)
    constraint_jacobian!(params, conjac_test, Z_test)
    @test size(conjac_FD) == size(conjac_test)
    @test all(con[1:idx.nc-nkey*2] .≈ 0.0)
    # compare conjac_FD and conjac_test and show the difference
    # conjac_diff = conjac_FD .- conjac_test
    # show the difference item index
    # display(conjac_FD[1:10, 1:10])
    # display(conjac_test[1:10, 1:10])
    # display(sparse(conjac_diff))
    # display(sparse(conjac_FD))
    # display(sparse(conjac_test))
    @test all(conjac_FD .≈ conjac_test)

    # r_guess = range(params.xi, params.xf, length=params.N)
    r_guess = zeros(params.N, 3)
    for i = 1:params.N
        theta = 2 * pi * i / params.N
        r_guess[i, 1] = rad_traj * cos(theta)
        r_guess[i, 2] = rad_traj * sin(theta)
        r_guess[i, 3] = 0.0
    end
    Z0 = zeros(params.idx.nz)
    for i = 1:params.N
        Z0[params.idx.x[i][1:3]] = r_guess[i, :]
        Z0[params.idx.x[i][13:15]] = r_guess[i, :] - [0.0, 0.0, model.l]
        if i < params.N
            Z0[params.idx.u[i]][1:4] = ones(4) .* (1.0 * 9.81 / 4.0)
            Z0[params.idx.u[i]][5] = 0.5 * 9.81
        end
    end

    conjac_tmp = create_conjac(params)
    display(sparse(conjac_tmp))

    Z = nlp.sparse_fmincon(cost::Function,
        cost_gradient!::Function,
        constraint!::Function,
        constraint_jacobian!::Function,
        conjac_tmp,
        Z_l::Vector,
        Z_u::Vector,
        c_l::Vector,
        c_u::Vector,
        Z0::Vector,
        params::NamedTuple;
        tol=1e-4,
        c_tol=1e-4,
        max_iters=1_000,
        verbose=true)
    # Z = Z0

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

animate_quadrotor_load(xs, xs, params.model.dt, params)
sleep(20.0)

# plot xyz
# plot(t_vec, [x[1] for x in xs], label="x")
# plot!(t_vec, [x[2] for x in xs], label="y")
# plot!(t_vec, [x[3] for x in xs], label="z")
# savefig("quadrotor_navigation_xyz.png")
# plot!(t_vec[1:end-1], [u[5] for u in us], label="u5")
# plot!(t_vec[1:end-1], [u[6] for u in us], label="u6")
# plot!(t_vec[1:end-1], [u[7] for u in us], label="u7")
# plot!(t_vec[1:end-1], [u[8] for u in us], label="u8")
# plot!(t_vec[1:end-1], [u[9] for u in us], labl="u9")
# plot!(t_vec[1:end-1], [u[10] for u in us], label="u10")
# # save to file
# savefig("quadrotor_navigation.png")