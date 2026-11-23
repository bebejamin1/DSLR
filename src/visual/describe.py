import sys
import csv

def load_csv(filepath):
    try:
        with open(filepath, mode='r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in reader]
            return header, rows
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}", file=sys.stderr)
        sys.exit(1)

def rows_to_dict(header, rows):
    columns = {col: [] for col in header}
    for row in rows:
        for col_name, val in zip(header, row):
            if val == "":
                columns[col_name].append(None)
            else:
                try:
                    columns[col_name].append(float(val))
                except ValueError:
                    columns[col_name].append(val)
    return columns

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <dataset.csv>", file=sys.stderr)
        sys.exit(1)
    filepath = sys.argv[1]
    header, rows = load_csv(filepath)
    print(rows_to_dict(header, rows))

if __name__ == "__main__":
    main()
