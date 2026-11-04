from src.data.build import load_panel
if __name__ == "__main__":
    df = load_panel()
    print(df.head())