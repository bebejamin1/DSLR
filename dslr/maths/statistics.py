"""Module fournissant les calculs statistiques et mathématiques pour DSLR."""


def extract_subjects(students: list[dict]):
    """Regroupe les notes par matière et par maison."""
    subjects_marks = {}
    subjects_house_marks = {}
    for student in students:
        house = student.get('Hogwarts House')
        for subject, mark in student.items():
            if isinstance(mark, (int, float)):
                if subject not in subjects_marks:
                    subjects_marks[subject] = []
                    subjects_house_marks[subject] = {}
                subjects_marks[subject].append(float(mark))
                if house:
                    if house not in subjects_house_marks[subject]:
                        subjects_house_marks[subject][house] = []
                    subjects_house_marks[subject][house].append(float(mark))
    return subjects_marks, subjects_house_marks


def calculate_mean(marks: list[float]) -> float:
    """Calcule la note moyenne d'une matière."""
    if not marks:
        return 0.0
    return sum(marks) / len(marks)


def calculate_std(marks: list[float], mean: float) -> float:
    """Calcule l'écart-type d'échantillon des notes par rapport à la moyenne."""
    count = len(marks)
    if count <= 1:
        return 0.0
    squared_differences = 0.0
    for mark in marks:
        difference = mark - mean
        squared_differences += difference * difference
    variance = squared_differences / (count - 1)
    return variance ** 0.5


def compute_statistics(students: list[dict]) -> dict[str, dict]:
    """Calcule les statistiques descriptives complètes pour chaque matière."""
    subjects_marks, subjects_houses = extract_subjects(students)
    subjects_statistics = {}
    for subject, marks in subjects_marks.items():
        house_means = {}
        for house, h_marks in subjects_houses.get(subject, {}).items():
            house_means[house] = calculate_mean(h_marks)
        marks.sort()
        count = len(marks)
        mean = calculate_mean(marks)
        subjects_statistics[subject] = {
            "count": count,
            "mean": mean,
            "std": calculate_std(marks, mean),
            "min": marks[0],
            "25%": marks[int(count / 4)],
            "50%": marks[int(count / 2)],
            "75%": marks[int(count * 3 / 4)],
            "max": marks[count - 1],
            list(house_means.keys())[0]: house_means[list(house_means.keys())[0]],
            list(house_means.keys())[1]: house_means[list(house_means.keys())[1]],
            list(house_means.keys())[2]: house_means[list(house_means.keys())[2]],
            list(house_means.keys())[3]: house_means[list(house_means.keys())[3]],
        }
    return subjects_statistics
