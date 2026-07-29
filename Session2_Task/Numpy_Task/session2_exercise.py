import numpy as np

grades = np.random.randint(40, 100, size=(5, 6))
students = np.array(['Layla', 'Omar', 'Sara', 'Kareem', 'Nour'])  # 5 students : rows
subjects = np.array(['Math', 'Science', 'English', 'History', 'Art', 'PE'])  # 6 subjects : columns
print("Grades:\n", grades)

# Task 2 — curve grades below 50
grades = np.where(grades < 50, grades + 5, grades)
print("Curved grades:\n", grades)

# Task 3 — row/column averages
student_avg = grades.mean(axis=1) #axis 1 for rows, each row(student) calculate the avg mark
subject_avg = grades.mean(axis=0) #same but for columns
print("Student averages:", dict(zip(students, student_avg)))
print("Subject averages:", dict(zip(subjects, subject_avg)))

# Task 4 — top student, hardest subject
top_student_index = student_avg.argmax()
top_student = students[top_student_index]

hardest_subject_index = subject_avg.argmin()  #lowest average mark -> hardest subject
hardest_subject = subjects[hardest_subject_index]

print("Top student:", top_student)
print("Hardest subject:", hardest_subject)

# Task 5 — Pass/Fail matrix
status = np.where(grades >= 60, 'Pass', 'Fail')
print("Status matrix:\n", status)

fail_count = np.sum(status == 'Fail')
print("Total fails:", fail_count)

# Task 6 — add Music column
music_grades = np.random.randint(40, 100, size=(5, 1))
grades = np.hstack([grades, music_grades])
subjects = np.append(subjects, 'Music')

student_avg = grades.mean(axis=1)  # recomputed with new data

# Task 7 — rank students highest to lowest
sorted_indices = np.argsort(student_avg)[::-1]
print("Ranking:")
for i in sorted_indices:
    print(f"{students[i]}: {student_avg[i]:.2f}")

# Bonus — normalize each student's grades (row-wise min-max)
row_min = grades.min(axis=1, keepdims=True)
row_max = grades.max(axis=1, keepdims=True)
normalized = (grades - row_min) / (row_max - row_min)
print("Normalized grades:\n", normalized)