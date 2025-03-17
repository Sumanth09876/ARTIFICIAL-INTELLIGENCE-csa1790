% Define the current date (this can be set manually or dynamically)
current_date(2025, 3, 17).

% Rule to calculate the Date of Birth (DOB) from the current date and age
dob_from_age(Age, YearOfBirth, MonthOfBirth, DayOfBirth) :-
    current_date(CurrentYear, CurrentMonth, CurrentDay),
    YearOfBirth is CurrentYear - Age,
    % Check if the current date has already passed the birthday this year
    (CurrentMonth > MonthOfBirth ; 
    (CurrentMonth = MonthOfBirth, CurrentDay >= DayOfBirth)).

dob_from_age(Age, YearOfBirth, MonthOfBirth, DayOfBirth) :-
    current_date(CurrentYear, CurrentMonth, CurrentDay),
    YearOfBirth is CurrentYear - Age - 1,
    % Handle cases where the birthday hasn't occurred yet in the current year
    (CurrentMonth < MonthOfBirth ; 
    (CurrentMonth = MonthOfBirth, CurrentDay < DayOfBirth)).
