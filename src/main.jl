using TrajectoryOptimization

include(joinpath(@__DIR__, "visualization.jl"))
include(joinpath(@__DIR__, "problem.jl"))

function get_trajs(; num_lift=3, verbose=true, visualize=true)
    # task parameters
    r0_load = [0,0,0.25]
    quat = true # task has quaternion

    # solver
    opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
        opts_uncon=opts_ilqr,
        cost_tolerance=1.0e-5,
        constraint_tolerance=1.0e-3,
        cost_tolerance_intermediate=1.0e-4,
        iterations=50,
        penalty_scaling=10.0,
        penalty_initial=1.0e-3)

    # initial guess by solving hovering problem
    prob_batch_trim = gen_prob(:batch, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=:hover)
    trim, trim_solver = solve(prob_batch_trim,opts_al)
    prob_batch = gen_prob(:batch, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=:p2p)
    initial_controls!(prob_batch,[trim.U[end] for k = 1:prob_batch.N-1])
    # initialize quaternion for LQR cost
    for i = 1:num_lift
        prob_batch.x0[(i-1)*13 .+ (4:7)] = trim.X[end][(i-1)*13 .+ (4:7)]
        prob_batch.xf[(i-1)*13 .+ (4:7)] = trim.X[end][(i-1)*13 .+ (4:7)]
    end

    # solve problem
    prob_batch, solver = solve(prob_batch, opts_al)

    # visualize
    if visualize
        vis = Visualizer()
        open(vis)
        visualize_batch(vis,prob_batch,true,num_lift)
    end    
    @info "stats" solver.stats[:iterations] max_violation(prob)   
    return prob 
end

get_trajs()