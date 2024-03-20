import Pkg; Pkg.add("Plots")

using Plots

function is_outside_donut(pos::Vector{Float64})
    a = 1.0  # major axis length
    b = 0.3  # minor axis length
    r = 1.25  # rotating radius

    x, y, z = pos[1], pos[2], pos[3]
    ellipse_center_dist = sqrt(y^2 + z^2)

    outside_ellipse1 = ((((ellipse_center_dist - r) / a)^2 + (x / b)^2) > 1)
    outside_ellipse2 = ((((ellipse_center_dist + r) / a)^2 + (x / b)^2) > 1)

    return outside_ellipse1 && outside_ellipse2
end

# Generate meshgrid from -2.5 to 2.5 with 100 points
x = range(-1.0, 1.0, length=100)
z = range(-1.0, 1.0, length=100)
y = range(-1.0, 1.0, length=100)
donut_points = [[x[i], y[j], z[k]] for i = 1:100, j = 1:100, k = 1:100]
is_outside_donut.(donut_points)

# Plot donut points
scatter(
    [point[1] for point in donut_points if is_outside_donut(point)],
    [point[2] for point in donut_points if is_outside_donut(point)],
    [point[3] for point in donut_points if is_outside_donut(point)],
    label="Outside",
    color="red",
    marker=:circle,
    markersize=2,
    xlabel="x",
    ylabel="y",
    zlabel="z",
    title="Donut",
)
scatter!(
    [point[1] for point in donut_points if !is_outside_donut(point)],
    [point[2] for point in donut_points if !is_outside_donut(point)],
    [point[3] for point in donut_points if !is_outside_donut(point)],
    label="Inside",
    color="blue",
    marker=:circle,
    markersize=2,
)
savefig("donut.png")

# Test points
test_points = [
    [0.0, 0.0, 0.0],  # outside
    [1.0, 0.0, 0.0],  # outside
    [0.0, 1.5, 0.0],  # inside
    [0.0, 0.0, 0.5],  # inside
    [0.0, -1.5, 0.0],  # inside
    [0.0, 0.0, -0.5],  # inside
    [0.0, 0.0, 0.0], # outside
]

# Verify the functionality
for point in test_points
    if is_outside_donut(point)
        println("Point $point is outside the donut.")
    else
        println("Point $point is inside the donut.")
    end
end