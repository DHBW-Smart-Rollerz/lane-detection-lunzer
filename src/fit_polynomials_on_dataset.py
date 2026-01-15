from fitting.load_data import get_current_data
from fitting.plot_label import get_points_of_all_3_lanes

if __name__ == "__main__":
    data = get_current_data()
    sample_data = data.sample(n=1)
    
    for index, row in sample_data.iterrows():
        left_lane_data, center_lane_data, right_lane_data = get_points_of_all_3_lanes(
            row
        )
        print(left_lane_data)
    
    