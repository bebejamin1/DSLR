from dslr.maths.statistics import compute_statistics


def replace_empty_marks_with_means(
    students: list[dict],
    subjects_statistics: dict[str, dict]
) -> list[dict]:
    """Remplace les notes manquantes d'un élève par la moyenne de la
    matière."""
    for student in students:
        for subject, mark in student.items():
            if (mark is None or mark == "") and subject in subjects_statistics:
                student[subject] = subjects_statistics[subject]['mean']
    return students


def standardise_marks(
    students: list[dict],
    subjects_statistics: dict[str, dict]
) -> list[dict]:
    """Standardise chaque note selon la
    formule Z-score: (note - mean) / std."""
    for student in students:
        for subject, mark in student.items():
            if subject in subjects_statistics:
                mean = subjects_statistics[subject]['mean']
                std = subjects_statistics[subject]['std']
                if std != 0:
                    student[subject] = (student[subject] - mean) / std
                else:
                    student[subject] = 0.0
    return students


def normalise(students: list[dict]) -> list[dict]:
    """Nettoie (imputation) et standardise les notes de tous les élèves."""
    subjects_statistics = compute_statistics(students)
    students = replace_empty_marks_with_means(students, subjects_statistics)
    students = standardise_marks(students, subjects_statistics)
    return students
