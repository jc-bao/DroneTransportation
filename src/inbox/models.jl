include(joinpath(@__DIR__, "quaternions.jl"))

"""
Model Parameters
"""

load_params = (m=0.9,
    gravity=SVector(0, 0, -9.81),
    radius=0.2)

quad_params = (m=1.0,
    J=SMatrix{3,3}(Diagonal([0.0023, 0.0023, 0.004])),
    Jinv=SMatrix{3,3}(Diagonal(1.0 ./ [0.0023, 0.0023, 0.004])),
    gravity=SVector(0, 0, -9.81),
    motor_dist=0.175,
    kf=1.0, # u to thrust
    km=0.0245, # u to torque
    radius=0.275)

"""
Dynamic Model
"""

function load_dynamics!(ẋ, x, u)
    ẋ[1:3] = x[4:6]
    ẋ[4:6] = u[1:3]
    ẋ[6] -= 9.81 # gravity
end

function lift_dynamics!(ẋ, x, u, params)

    q = normalize(Quaternion(view(x, 4:7)))
    v = view(x, 8:10)
    omega = view(x, 11:13)

    # Parameters
    m = params[:m] # mass
    J = params[:J] # inertia matrix
    Jinv = params[:Jinv] # inverted inertia matrix
    g = params[:gravity] # gravity
    L = params[:motor_dist] # distance between motors

    w1 = u[1]
    w2 = u[2]
    w3 = u[3]
    w4 = u[4]

    kf = params[:kf] # 6.11*10^-8;
    F1 = kf * w1
    F2 = kf * w2
    F3 = kf * w3
    F4 = kf * w4
    F = @SVector [0.0, 0.0, F1 + F2 + F3 + F4] #total rotor force in body frame

    km = params[:km]
    M1 = km * w1
    M2 = km * w2
    M3 = km * w3
    M4 = km * w4
    # Macro for static array
    tau = @SVector [L * (F2 - F4), L * (F3 - F1), (M1 - M2 + M3 - M4)] #total rotor torque in body frame

    ẋ[1:3] = v # velocity in world frame
    ẋ[4:7] = SVector(0.5 * q * Quaternion(zero(x[1]), omega...))
    ẋ[8:10] = g + (1 / m) * (q * F + u[5:7]) #acceleration in world frame
    ẋ[11:13] = Jinv * (tau - cross(omega, J * omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
    return tau, omega, J, Jinv
end

function batch_dynamics!(ẋ,x,u, params)
    """
    generate dynamics for the joint control problem
    """
    # Unpack parameters
    lift_params = params.lift
    load_params = params.load
    load_mass = load_params.m
    n_batch = length(x)
    num_lift = (n_batch - 6) ÷ 13

    # Unpack indices
    lift_inds = [(1:13) .+ i for i in (0:13:13*num_lift-1)]
    load_inds = (1:6) .+ 13*num_lift
    r_inds = [(1:3) .+ i for i in (0:13:13*num_lift)] # position indices
    s_inds = 5:5:5*num_lift # rope force indices
    u_inds = [(1:4) .+ i for i in (0:5:5*num_lift-1)] # quadrotor control indices

    # Get 3D positions
    r = [x[inds] for inds in r_inds]
    rlift = r[1:num_lift]
    rload = r[end]

    # Get control values
    u_quad = [u[ind] for ind in u_inds]
    u_load = u[(1:num_lift) .+ 5*num_lift]
    s_quad = u[s_inds]

    dir = [(rload - r)/norm(rload - r) for r in rlift]

    lift_control = [[u_quad[i]; s_quad[i]*dir[i]] for i = 1:num_lift]
    u_slack_load = -1.0*sum(dir .* u_load)

    for i = 1:num_lift
        inds = lift_inds[i]
        lift_dynamics!(view(ẋ,inds), x[inds], lift_control[i], lift_params) # view is a reference to the array, so ẋ is modified
    end

    load_dynamics!(view(ẋ, load_inds), x[load_inds], u_slack_load/load_mass)

    return nothing
end