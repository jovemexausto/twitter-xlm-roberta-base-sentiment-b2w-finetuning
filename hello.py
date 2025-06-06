import pandas as pd

def main():
    # Load the CSV file
    df = pd.read_csv('./data/b2w.csv')
    # Filter rows where rating == 3
    filtered = df[df['rating'] == 3]
    # Show top 10 rows
    print(filtered.tail(10))

if __name__ == "__main__":
    main()
