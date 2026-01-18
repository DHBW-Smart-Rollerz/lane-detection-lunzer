import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .common_helpers import get_points_of_all_3_lanes, to_xy_arrays


def plot_lanes(row):
    """
    Plot lane annotation points for a single sample using Plotly.

    Creates a scatter plot containing the lane point sets (left/center/right) as
    separate traces. The y-axis is reversed to match the OpenCV image coordinate
    system (origin at top-left).

    Args:
        row:
            A pandas Series or dict-like row containing lane label fields.

    Returns:
        plotly.graph_objects.Figure:
            Plotly figure containing the lane point scatter traces.
    """
    left_lane_data, center_lane_data, right_lane_data = get_points_of_all_3_lanes(row)
    fig = go.Figure()
    if left_lane_data:
        x_array, y_array = to_xy_arrays(left_lane_data)
    else:
        left_lane_data = None
    fig.add_scatter(
        x=x_array,
        y=y_array,
        mode="markers",
        marker=dict(size=6, symbol="circle", color="red"),
    )
    if center_lane_data:
        x_array, y_array = to_xy_arrays(center_lane_data)
    else:
        center_lane_data = None
    fig.add_scatter(
        x=x_array,
        y=y_array,
        mode="markers",
        marker=dict(size=6, symbol="circle", color="green"),
    )
    if right_lane_data:
        x_array, y_array = to_xy_arrays(right_lane_data)
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
