% Facts: (student(StudentName, Subcode), teacher(TeacherName, Subcode))

% Students and the courses they are enrolled in
student(john, math101).
student(jane, cs102).
student(mike, math101).
student(susan, cs102).
student(david, eng202).

% Teachers and the courses they are teaching
teacher(mr_smith, math101).
teacher(mrs_jones, cs102).
teacher(mr_brown, eng202).

% Rule: Find the teacher for a given subcode
teacher_for_subcode(Subcode, Teacher) :-
    teacher(Teacher, Subcode).

% Rule: Find the students enrolled in a given subcode
students_in_course(Subcode, Student) :-
    student(Student, Subcode).

% Rule: Find which teacher is teaching a given student
teacher_for_student(Student, Teacher) :-
    student(Student, Subcode),
    teacher(Teacher, Subcode).
