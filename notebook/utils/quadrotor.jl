function dcm_from_mrp(p)
    p1,p2,p3 = p
    den = (p1^2 + p2^2 + p3^2 + 1)^2
    a = (4*p1^2 + 4*p2^2 + 4*p3^2 - 4)
    [
    (-((8*p2^2+8*p3^2)/den-1)*den)   (8*p1*p2 + p3*a)     (8*p1*p3 - p2*a);
    (8*p1*p2 - p3*a) (-((8*p1^2 + 8*p3^2)/den - 1)*den)   (8*p2*p3 + p1*a);
    (8*p1*p3 + p2*a)  (8*p2*p3 - p1*a)  (-((8*p1^2 + 8*p2^2)/den - 1)*den)
    ]/den
end
function skew(ω::Vector)
    return [0    -ω[3]  ω[2];
            ω[3]  0    -ω[1];
           -ω[2]  ω[1]  0]
end
function quadrotor_dynamics(model::NamedTuple,x,u)
    # quadrotor dynamics with an MRP for attitude
    # and velocity in the world frame (not body frame)
    
    r = x[1:3]     # position in world frame 
    v = x[4:6]     # position in body frame 
    p = x[7:9]     # n_p_b (MRP) attitude 
    ω = x[10:12]   # angular velocity 

    Q = dcm_from_mrp(p) # DCM from MRP to body frame

    mass=model.mass
    J = model.J
    gravity= model.gravity
    L= model.L
    kf=model.kf
    km=model.km

    w1 = u[1]
    w2 = u[2]
    w3 = u[3]
    w4 = u[4]
    f_rope = u[5:7]

    F1 = max(0,kf*w1)
    F2 = max(0,kf*w2)
    F3 = max(0,kf*w3)
    F4 = max(0,kf*w4)
    F = [0., 0., F1+F2+F3+F4] #total rotor force in body frame

    M1 = km*w1
    M2 = km*w2
    M3 = km*w3
    M4 = km*w4
    τ = [L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame

    f = mass*gravity + Q*F + f_rope # forces in world frame

    # this is xdot 
    [
        v
        f/mass
        ((1+norm(p)^2)/4) *(   I + 2*(skew(p)^2 + skew(p))/(1+norm(p)^2)   )*ω # integrate MRP
        J\(τ - cross(ω,J*ω))
    ]
end
function rk4(model,ode,x,u,dt)
    # rk4 
    k1 = dt*ode(model,x, u)
    k2 = dt*ode(model,x + k1/2, u)
    k3 = dt*ode(model,x + k2/2, u)
    k4 = dt*ode(model,x + k3, u)
    x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
end
function vis_traj!(vis, name, X; R = 0.1, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))
    # visualize a trajectory expressed with X::Vector{Vector}
    for i = 1:(length(X)-1)
        a = X[i][1:3]
        b = X[i+1][1:3]
        cyl = mc.Cylinder(mc.Point(a...), mc.Point(b...), R)
        mc.setobject!(vis[name]["p"*string(i)], cyl, mc.MeshPhongMaterial(color=color))
    end
    for i = 1:length(X)
        a = X[i][1:3]
        sph = mc.HyperSphere(mc.Point(a...), R)
        mc.setobject!(vis[name]["s"*string(i)], sph, mc.MeshPhongMaterial(color=color))
    end
end

function cable_transform(y,z)
    v1 = [0,0,1]
    v2 = y[1:3,1] - z[1:3,1]
    normalize!(v2)
    ax = cross(v1,v2)
    ang = acos(v1'v2)
    R = AngleAxis(ang,ax...)
    compose(Translation(z),LinearMap(R))
end

function animate_quadrotor_load(Xsim, Xref, dt)
    # animate quadrotor, show Xref with vis_traj!, and track Xref with the green sphere
    vis = mc.Visualizer()
    robot_obj = mc.MeshFileGeometry(joinpath(@__DIR__,"quadrotor.obj"))
    mc.setobject!(vis[:vic], robot_obj, mc.MeshPhongMaterial(color = mc.RGBA(0.1,0.1,0.1,1.0)))
    load_obj = mc.HyperSphere(mc.Point(0,0,0.0),0.03)
    mc.setobject!(vis[:load], load_obj, mc.MeshPhongMaterial(color = mc.RGBA(1.0,0.5,1.0,1.0)))

    gate = mc.MeshFileGeometry(joinpath(@__DIR__,"donut.obj"))
    mc.setobject!(vis[:gate1], gate, mc.MeshPhongMaterial(color = mc.RGBA(0.0,1.0,0.0,1.0)))
    # rotate gate along x axis by [pi/2, 0, 0] and scale by 2.0
    mc.settransform!(vis[:gate1], mc.compose(mc.Translation([1.0,0,0]), mc.LinearMap(AngleAxis(pi/2,0,1,0).*2.0)))
    mc.settransform!(vis[:gate2], mc.compose(mc.Translation([3.0,0,0]), mc.LinearMap(AngleAxis(pi/2,0,1,0).*2.0)))
    
    anim = mc.Animation(floor(Int,1/dt))
    for k = 1:length(Xsim)
        mc.atframe(anim, k) do
            r = Xsim[k][1:3]
            p = Xsim[k][7:9]
            r_load = Xsim[k][13:15]
            mc.settransform!(vis[:vic], mc.compose(mc.Translation(r),mc.LinearMap(0.5*(dcm_from_mrp(p)))))
            mc.settransform!(vis[:load], mc.Translation(r_load))
            # settransform!(vis["cable"], cable_transform(r,r_load))
            # mc.settransform!(vis[:target], mc.Translation(Xref[k][1:3]))
            # mc.settransform!(vis[:target_load], mc.Translation(Xref[k][13:15]))

            # place hole at the origin (0,0,0) orient to +x
            # mc.settransform!(vis[:cyl], [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
        end
    end
    mc.setanimation!(vis, anim)

    return (mc.render(vis))
end

function load_dynamics(model::NamedTuple,x,u)
    # load dynamics
    r = x[1:3]     # position in world frame
    v = x[4:6]     # position in body frame
    a = u[1:3]/model.mass_load + model.gravity     # acceleration in world frame
    x_dot = [
        v
        a
    ]
    return x_dot
end

function combined_dynamics(model::NamedTuple,x,u)
    # combined dynamics
    x_lift = x[1:12]
    x_load = x[13:18]
    u_rotor = u[1:4]
    f_rope = u[5]
    pos_lift = x_lift[1:3]
    pos_load = x_load[1:3]
    load2lift = pos_load - pos_lift
    n_load2lift = load2lift/norm(load2lift)

    u_lift = [u_rotor; f_rope.*n_load2lift]
    u_load = -f_rope.*n_load2lift

    x_lift_dot = quadrotor_dynamics(model,x_lift,u_lift)
    x_load_dot = load_dynamics(model,x_load,u_load)

    return [x_lift_dot; x_load_dot]
end