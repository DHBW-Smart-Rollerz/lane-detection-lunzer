from fitting.load_data import get_current_data
from fitting.plot_label import plot_lanes
from fitting.visualize import draw_lanes, load_image, send_frame_to_server

if __name__ == "__main__":
    data = get_current_data()
    sample_data = data.sample(n=1)

    for _, row in sample_data.iterrows():
        fig = plot_lanes(row)
        fig.show()

        img = load_image(row["image_path"])
        img = draw_lanes(img, row)
        send_frame_to_server(img)
