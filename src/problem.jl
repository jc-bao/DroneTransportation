using Combinatorics
using TrajectoryOptimization
include("models.jl")


function gen_prob(agent, quad_params, load_params, r0_load=[0, 0, 0.25];
    num_lift=3, N=51, quat=false, scenario=:doorway)
    """
    generate optimization problem
    agent: problem to solve, either :load or :batch or lift_idx
    quad_params: parameters for the quadrotor
    load_params: parameters for the load
    r0_load: initial location of the load
    num_lift: number of quadrotors
    N: number of knot points
    quat: if system involves quaternions
    """

    """
    unpack parameters
    """

    scenario == :doorway ? obs = true : obs = false

    # parameters for dynamics
    dt = 0.2

    # parameters for tasks
    goal_dist = 0.0 ? scenario == :hover : 6.0
    d = 1.55 # rope length
    lift_radius = 0.275
    load_radius = 0.2
    mass_load = load_params.m::Float64
    mass_lift = quad_params.m::Float64

    # parameters for initial configuration
    α = asin(r_config / d) # angle between the ropes and the vertical for the initial configuration
    r_config = 1.2 # initially, drones is forming a circle, this is the radius of the circle
    if num_lift > 6
        d *= 2.5
        r_config *= 2.5
    end

    # parameters for obstacle avoidance
    β = deg2rad(50)  # fan angle (radians) for mid point formation, to arrange drones to a-z plane, such that it can pass through the doorway
    Nmid = convert(Int, floor(N / 2)) + 1 # middle knot point, mainly for the obstacle avoidance
    r_cylinder = 0.5 # radius of the cylinder obstacle
    ceiling = 2.1 # height of the ceiling

    # parameters for constraints
    n_lift = 13 # lift state dimension
    m_lift = 5 # lift control dimension
    n_load = 6 # load state dimension
    m_load = num_lift # load control dimension, each rope has a force
    n_batch = num_lift * n_lift + n_load # batch state dimension
    m_batch = num_lift * m_lift + m_load # batch control dimension

    # parameters for initial configuration
    xlift0, xload0 = get_states(r0_load, n_lift, n_load, num_lift, d, α)
    # initial state
    x0 = vcat(xlift0...,xload0)
    
    # parameters for final configuration
    rf_load = [goal_dist, 0, r0_load[3]] # final location of the load
    xliftf, xloadf = get_states(rf_load, n_lift, n_load, num_lift, d, α)
    # parameters for final state guess
    ulift_static, uload_static = calc_static_forces(α, quad_params.m, mass_load, num_lift)
    # NOTE: now only works for num_lift = 3
    q10 = [0.99115, 4.90375e-16, 0.132909, -9.56456e-17]
    u10 = [3.32131, 3.32225, 3.32319, 3.32225, 4.64966]
    q20 = [0.99115, -0.115103, -0.0664547, 1.32851e-17]
    u20 = [3.32272, 3.32144, 3.32178, 3.32307, 4.64966]
    q30 = [0.99115, 0.115103, -0.0664547, 1.92768e-16]
    u30 = [3.32272, 3.32307, 3.32178, 3.32144, 4.64966]
    uload = [4.64966, 4.64966, 4.64966]
    q_lift_static3 = [q10, q20, q30]
    ulift3 = [u10, u20, u30]
    if num_lift == 3
        for i = 1:num_lift
            xlift0[i][4:7] = q_lift_static3[i]
            xliftf[i][4:7] = q_lift_static3[i]
        end
    end
    # final state
    xf = vcat(xliftf...,xloadf)
    uf = vcat(ulift_static...,uload_static) # for final cost
    
    # parameters for mid point formation
    rm_load = [goal_dist / 2, 0, r0_load[3]] # mid point location of the load
    rm_lift = get_quad_locations(rm_load, d, β, num_lift, config=:doorway)
    ulift_static_mid, uload_static_mid = calc_static_forces(α, quad_params.m, mass_load, num_lift)
    xliftmid = [zeros(n_lift) for i = 1:num_lift]
    for i = 1:num_lift
        xliftmid[i][1:3] = rm_lift[i]
        xliftmid[i][4] = 1.0
    end
    if num_lift == 3
        xliftmid[2][2] = 0.01
        xliftmid[3][2] = -0.01
    end
    xloadm = zeros(n_load)
    xloadm[1:3] = rm_load
    xm = vcat(xliftmid...,xloadm)
    um = vcat(ulift_static_mid...,uload_static_mid) 

    # objective functions parameters
    q_lift, r_lift, qf_lift = quad_costs(n_lift, m_lift, scenario)
    q_load, r_load, qf_load = load_costs(n_load, m_load, scenario)
    q_lift_mid = copy(q_lift)
    q_load_mid = copy(q_load)
    q_lift_mid[1:3] .= 10
    q_load_mid[1:3] .= 10
    # convert to diagonal
    # for lift
    Q_lift = Diagonal(q_lift)
    Qf_lift = Diagonal(qf_lift)
    R_lift = Diagonal(r_lift)
    # for load
    Q_load = Diagonal(q_load)
    Qf_load = Diagonal(qf_load)
    R_load = Diagonal(r_load)
    # for batch system
    Q = Diagonal([repeat(q_lift, num_lift); q_load])
    Qf = Diagonal([repeat(qf_lift, num_lift); qf_load])
    R = Diagonal([repeat(r_lift, num_lift); r_load])
    Q_mid = Diagonal([repeat(q_lift_mid, num_lift); q_load_mid])
    # if involves quaternions, convert with quaternion jacobian
    if quat:
        Gf = state_diff_jacobian(model, xf)
        Qf = Gf' * Qf * Gf
        Q = Gf' * Q * Gf
        Q_mid = Gf' * Q_mid * Gf

    # constraints
    u_min_lift = [0,0,0,0,-Inf]
    u_max_lift = ones(m_lift)*(mass_load + mass_lift)*9.81/4
    u_max_lift[end] = Inf
    x_min_lift = -Inf*ones(n_lift)
    x_min_lift[3] = 0
    x_max_lift = Inf*ones(n_lift)
    if scenario == :doorway
        x_max_lift[3] = ceiling
    end
    u_min_load = zeros(num_lift)
    u_max_load = ones(m_load)*Inf
    x_min_load = -Inf*ones(n_load)
    x_min_load[3] = 0
    x_max_load = Inf*ones(n_load)
    if scenario == :doorway
        x_max_load[3] = ceiling
    end

    # obstacles
    _cyl = door_obstacles(r_cylinder, goal_dist/2)


    """
    constrains
    """
    # obstacle constrains
    function cI_cylinder_lift(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*lift_radius)
        end
    end
    function cI_cylinder_load(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*load_radius)
        end
    end
    function cI_cylinder(c,x,u)
        # cylinder obstacle constraint
        c_shift = 1
        n_slack = 3
        for p = 1:length(_cyl)
            n_shift = 0
            for i = 1:num_lift
                idx_pos = (n_shift .+ (1:13))[1:3]
                c[c_shift] = circle_constraint(x[idx_pos],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*lift_radius)
                c_shift += 1
                n_shift += 13
            end
            c[c_shift] = circle_constraint(x[num_lift*13 .+ (1:3)],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*lift_radius)
            c_shift += 1
        end
    end

    # state constraints
    r_inds = [(1:3) .+ i for i in (0:13:13*num_lift)] # drone + load position index
    s_inds = 5:5:5*num_lift # rope force index
    s_load = (1:num_lift) .+ 5*num_lift # load force index
    function distance_constraint(c,x,u=zeros(m_batch))
        # distance between the load and the drones
        r_load = x[r_inds[end]]
        for i = 1:num_lift
            r_lift = x[r_inds[i]]
            c[i] = norm(r_lift - r_load)^2 - d^2
        end
        return nothing
    end
    quad_pairs = combinations(1:num_lift, 2)
    function collision_constraint(c,x,u=zeros(m_batch))
        # drone collision constraint
        r_lift = [x[inds] for inds in r_inds]
        for (p,pair) in enumerate(quad_pairs)
            i,j = pair
            c[p] = circle_constraint(r_lift[i], r_lift[j][1], r_lift[j][2], 2*lift_radius)
        end
        return nothing
    end
    

    # control constrains
    function force_constraint(c,x,u)
        # rope force constrains
        u_load = u[s_load]
        for i = 1:num_lift
            c[i] = u[s_inds[i]] - u_load[i]
        end
        return nothing
    end

    """
    generate problem
    """

    if agent == :load
        # raise error
        error("Not implemented yet")
    elseif agent ∈ 1:num_lift
        error("Not implemented yet")
    elseif agent == :batch
        # dynamics
        info = Dict{Symbol,Any}()
        if quat
            info[:quat] = [(4:7) .+ i for i in 0:n_lift:n_lift*num_lift-1]
        end
        batch_params = (lift=quad_params, load=load_params)
        model_batch = Model(batch_dynamics!, n_batch, m_batch, batch_params, info)

        # objective function
        obj = LQRObjective(Q, R, Qf, xf, N, uf)
        if obs:
            obj.cost[Nmid] = LQRCost(Q_mid, R, xm, um)

        # control bounds
        # Bound Constraints
        u_l = [repeat(u_min_lift, num_lift); u_min_load]
        u_u = [repeat(u_max_lift, num_lift); u_max_load]
        x_l = [repeat(x_min_lift, num_lift); x_min_load]
        x_u = [repeat(x_max_lift, num_lift); x_max_load]
        # also constrain for object final states
        # NOTE: note sure how does it compared with equality constrains
        x_l_N = copy(x_l)
        x_u_N = copy(x_u)
        x_l_N[end-(n_load-1):end] = [rf_load;zeros(3)]
        x_u_N[end-(n_load-1):end] = [rf_load;zeros(3)]

        bnd = BoundConstraint(n_batch,m_batch,u_min=u_l,u_max=u_u, x_min=x_l, x_max=x_u)
        bndN = BoundConstraint(n_batch,m_batch,x_min=x_l_N,x_max=x_u_N)

        # Constraints
        cyl = Constraint{Inequality}(cI_cylinder,n_batch,m_batch,(num_lift+1)*length(_cyl),:cyl)
        dist_con = Constraint{Equality}(distance_constraint,n_batch,m_batch, num_lift, :distance)
        for_con = Constraint{Equality}(force_constraint,n_batch,m_batch, num_lift, :force)
        col_con = Constraint{Inequality}(collision_constraint,n_batch,m_batch, binomial(num_lift, 2), :collision)
        con = Constraints(N)
        for k = 1:N-1
            con[k] += dist_con + for_con + bnd + col_con
            if obs
                con[k] += cyl
            end
        end
        con[N] +=  col_con  + dist_con + bndN
        if obs
            con[N] += cyl
        end

        # Problem
        prob = Problem(model_batch, obj, constraints=con,
                dt=dt, N=N, xf=xf, x0=x0,
                integration=:midpoint)
        # Initial controls
        U0 = [uf for k = 1:N-1]
        initial_controls!(prob, U0)

        return prob
    end

end

function calc_static_forces(α::Float64, lift_mass, load_mass, num_lift)
    thrust = 9.81*(lift_mass + load_mass/num_lift)/4
    f_mag = load_mass*9.81/(num_lift*cos(α))
    ulift = [[thrust; thrust; thrust; thrust; f_mag] for i = 1:num_lift]
    uload = ones(num_lift)*f_mag
    return ulift, uload
end

function get_quad_locations(x_load::Vector, d::Real, α=π / 4, num_lift=3;
    config=:default, r_cables=[zeros(3) for i = 1:num_lift], ϕ=0.0)
    """
    Get the initial locations of the quadrotors

    x_load: initial location of the load
    d: rope length
    α: angle between the ropes and the vertical
    num_lift: number of quadrotors
    config: :default or :doorway
    r_cables: attachment locations of the quadrotors
    ϕ: angle rotate around the z-axis for the overall formation
    """
    if config == :default
        # get the initial locations of the quadrotors in spherical coordinates
        h = d * cos(α)
        r = d * sin(α)
        z = x_load[3] + h
        circle(θ) = [x_load[1] + r * cos(θ), x_load[2] + r * sin(θ)] # lambda function
        θ = range(0, 2π, length=num_lift + 1) .+ ϕ
        x_lift = [zeros(3) for i = 1:num_lift]
        for i = 1:num_lift
            if num_lift == 2
                x_lift[i][1:2] = circle(θ[i] + pi / 2)
            else
                x_lift[i][1:2] = circle(θ[i])
            end
            x_lift[i][3] = z
            x_lift[i] += r_cables[i]  # Shift by attachment location
        end
    elseif config == :doorway
        # in doorway case, place the quadrotors in x-z plane
        y = x_load[2]
        fan(θ) = [x_load[1] - d * sin(θ), y, x_load[3] + d * cos(θ)]
        θ = range(-α, α, length=num_lift)
        x_lift = [zeros(3) for i = 1:num_lift]
        for i = 1:num_lift
            x_lift[i][1:3] = fan(θ[i])
        end
    end
    return x_lift
end

function get_states(r_load, n_lift, n_load, num_lift, d=1.55, α=deg2rad(50))
    r_lift = get_quad_locations(r_load, d, α, num_lift)
    x_lift = [zeros(n_lift) for i = 1:num_lift]
    for i = 1:num_lift
        x_lift[i][1:3] = r_lift[i]
        x_lift[i][4] = 1.0
    end

    x_load = zeros(n_load)
    x_load[1:3] = r_load
    return x_lift, x_load
end

function quad_costs(n_lift, m_lift, scenario=:doorway)
    if scenario == :hover
        q_diag = 10.0*ones(n_lift)
        q_diag[4:7] .= 1e-6

        r_diag = 1.0e-3*ones(m_lift)
        r_diag[end] = 1

        qf_diag = copy(q_diag)*10.0
    elseif scenario == :p2pa
        q_diag = 1.0*ones(n_lift)
        q_diag[1] = 1e-5

        r_diag = 1.0e-3*ones(m_lift)
        r_diag[end] = 1

        qf_diag = 100*ones(n_lift)
    else
        q_diag = 1e-1*ones(n_lift)
        q_diag[1] = 1e-3 # x position
        q_diag[4:7] .*= 25.0

        r_diag = 2.0e-3*ones(m_lift)

        r_diag[end] = 1

        qf_diag = 100*ones(n_lift)
    end
    return q_diag, r_diag, qf_diag
end

function load_costs(n_load, m_load, scenario=:doorway)
    if scenario == :hover
        q_diag = 10.0*ones(n_load) #

        r_diag = 1*ones(m_load)
        qf_diag = 10.0*ones(n_load)
    elseif scenario == :p2p
        q_diag = 1.0*ones(n_load) #

        q_diag[1] = 1.0e-5
        r_diag = 1*ones(m_load)
        qf_diag = 0.0*ones(n_load)
    else
        q_diag = 0.5e-1*ones(n_load) #

        r_diag = 1*ones(m_load)
        qf_diag = 0.0*ones(n_load)
    end
    return q_diag, r_diag, qf_diag
end

function door_obstacles(r_cylinder=0.5, x_door=3.0)
    _cyl = NTuple{3,Float64}[]

    push!(_cyl,(x_door, 1.,r_cylinder))
    push!(_cyl,(x_door,-1.,r_cylinder))
    push!(_cyl,(x_door-0.5, 1.,r_cylinder))
    push!(_cyl,(x_door-0.5,-1.,r_cylinder))
    return _cyl
end