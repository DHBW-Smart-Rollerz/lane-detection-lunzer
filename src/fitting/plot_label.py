import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from load_data import get_current_data
from visualize_original_label import (
    draw_lanes,
    load_image,
    send_frame_to_server,
    split_point_string_to_points,
)


def get_points_of_all_3_lanes(data_row):
    left_lane_data = split_point_string_to_points(data_row["left lane"])
    center_lane_data = split_point_string_to_points(data_row["center lane"])
    right_lane_data = split_point_string_to_points(data_row["right lane"])
    return left_lane_data, center_lane_data, right_lane_data


def create_xy_arrays(points):
    x_array = []
    y_array = []
    for xy in points:
        if xy is None or len(xy) != 2:
            print(f"Koordinate {xy} hat Probleme verursacht und wird übersprungen")
            continue
        x, y = xy[0], xy[1]
        x_array.append(x)
        y_array.append(y)
    return x_array, y_array


def create_lane_polynomials(row):
    left_lane_data, center_lane_data, right_lane_data = get_points_of_all_3_lanes(row)
    if left_lane_data:
        x_array, y_array = create_xy_arrays(left_lane_data)
        left_poly_pred = np.polynomial.polynomial.polyfit(x_array, y_array, 1)
    else:
        left_lane_data = None

    if center_lane_data:
        x_array, y_array = create_xy_arrays(center_lane_data)
        center_poly_pred = np.polynomial.polynomial.polyfit(x_array, y_array, 1)
    else:
        center_lane_data = None

    if right_lane_data:
        x_array, y_array = create_xy_arrays(right_lane_data)
        right_poly_pred = np.polynomial.polynomial.polyfit(x_array, y_array, 1)
    else:
        right_lane_data = None

    return left_poly_pred, center_poly_pred, right_poly_pred


def plot_lanes(row):
    left_lane_data, center_lane_data, right_lane_data = get_points_of_all_3_lanes(row)
    fig = go.Figure()
    if left_lane_data:
        x_array, y_array = create_xy_arrays(left_lane_data)
    else:
        left_lane_data = None
    fig.add_scatter(
        x=x_array,
        y=y_array,
        mode="markers",
        marker=dict(size=6, symbol="circle", color="red"),
    )
    if center_lane_data:
        x_array, y_array = create_xy_arrays(center_lane_data)
    else:
        center_lane_data = None
    fig.add_scatter(
        x=x_array,
        y=y_array,
        mode="markers",
        marker=dict(size=6, symbol="circle", color="green"),
    )
    if right_lane_data:
        x_array, y_array = create_xy_arrays(right_lane_data)
    else:
        right_lane_data = None
    fig.add_scatter(
        x=x_array,
        y=y_array,
        mode="markers",
        marker=dict(size=6, symbol="circle", color="blue"),
    )

    fig.update_yaxes(
        autorange="reversed"
    )  # Ans CV2 Koordinatensystem anpassen (Y invertiert)
    fig.update_layout(
        yaxis_rangemode="tozero", xaxis_rangemode="tozero"
    )  # Auf 0 Legen ohne negative abschnitte im Graph
    return fig


if __name__ == "__main__":
    data = get_current_data()
    sample_data = data.sample(n=1)

    for index, row in sample_data.iterrows():
        print(create_lane_polynomials(row))
        left_lane_data, center_lane_data, right_lane_data = get_points_of_all_3_lanes(
            row
        )
        fig = plot_lanes(row)
        fig.show()
        img = load_image(row["image_path"])
        img = draw_lanes(img, row)
        send_frame_to_server(img)
