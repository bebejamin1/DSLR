
from dslr.visual.describe import calculate_describe_notes


def replace_empty_notes_with_means(data: list, subjects_infos: dict) -> list:
    for student in data:
        for subject, note in student.items():
            if (note is None or note == "") and subject in subjects_infos:
                student[subject] = subjects_infos[subject]['mean']
    return data


def standardise(data: list, subjects_infos: dict) -> list:
    for student in data:
        for subject, note in student.items():
            if subject in subjects_infos:
                mean = subjects_infos[subject]['mean']
                std = subjects_infos[subject]['std']
                if std != 0:
                    student[subject] = (student[subject] - mean) / std 
                else: 
                    student[subject] = 0.0
    return data


def normalise(data: list) -> list:
    subjects_infos = calculate_describe_notes(data)
    data = replace_empty_notes_with_means(data, subjects_infos)
    data = standardise(data, subjects_infos)
    print(data)
    return data
