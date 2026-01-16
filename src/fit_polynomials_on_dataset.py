from fitting.fit_polynomial import fit_poly_x_of_y
from fitting.load_data import get_current_data
from fitting.plot_label import get_points_of_all_3_lanes

if __name__ == "__main__":
    data = get_current_data()
    sample_data = data.sample(n=100)

    for index, row in sample_data.iterrows():
        (
            left_lane_points,
            center_lane_points,
            right_lane_points,
        ) = get_points_of_all_3_lanes(row)
        print("Points:")
        print(left_lane_points)
        print(center_lane_points)
        print(right_lane_points)

        poly_left_d2 = fit_poly_x_of_y(left_lane_points, 2)
        poly_center_d2 = fit_poly_x_of_y(center_lane_points, 2)
        poly_right_d2 = fit_poly_x_of_y(right_lane_points, 2)
        print("Polynomials:")
        print(poly_left_d2)
        print(poly_center_d2)
        print(poly_right_d2)
        print(f"Index: {index} done")
