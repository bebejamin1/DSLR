import sys

def load_csv():
    pass

def get_rows():
    pass

def main():
    print(len(sys.argv))
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <dataset.csv>", file=sys.stderr)
        sys.exit(1)
    filepath = sys.argv[1]
    header, rows = load_csv(filepath)

if __name__ == "__main__":
    main()
