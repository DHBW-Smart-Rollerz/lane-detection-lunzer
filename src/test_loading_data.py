from fitting.load_data import get_current_data

if __name__ == "__main__":
    data = get_current_data()
    data.to_excel("artifacts/current_data.xlsx")
    print(data)
