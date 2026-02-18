class User:
    def __init__(self, uid, name):
        self.__id = uid
        self.__name = name
    def get_id(self):
        return self.__id
    def get_name(self):
        return self.__name
    def display_details(self):
        print(self.__id, self.__name)
# STUDENT
class Student(User):
    def __init__(self, uid, name):
        super().__init__(uid, name)
        self.course = None
    def enroll(self, course):
        self.course = course
    def display_details(self):
        print("Student:", self.get_id(), self.get_name(), "Course:", self.course)
# MENTOR
class Mentor(User):
    def __init__(self, uid, name):
        super().__init__(uid, name)
        self.students = []
    def assign_student(self, student):
        self.students.append(student)
    def display_details(self):   # Polymorphism
        print("Mentor:", self.get_id(), self.get_name())
        print("Students:")
        for s in self.students:
            print("-", s.get_name())
# ADMIN
class Admin(User):
    def view_all(self, students, mentors):
        print("\nAll Students:")
        for s in students:
            s.display_details()
        print("\nAll Mentors:")
        for m in mentors:
            m.display_details()
# MENU
students = []
mentors = []
admin = Admin(1, "Admin")
while True:
    print("\n1.Add Student  2.Add Mentor  3.Enroll Course")
    print("4.Assign Mentor  5.View Mentor  6.Admin View  7.Exit")
    ch = input("Enter choice: ")
    if ch == "1":
        students.append(Student(input("ID: "), input("Name: ")))
    elif ch == "2":
        mentors.append(Mentor(input("ID: "), input("Name: ")))
    elif ch == "3":
        sid = input("Student ID: ")
        for s in students:
            if s.get_id() == sid:
                s.enroll(input("Course: "))
    elif ch == "4":
        sid = input("Student ID: ")
        mid = input("Mentor ID: ")
        for s in students:
            for m in mentors:
                if s.get_id() == sid and m.get_id() == mid:
                    m.assign_student(s)
    elif ch == "5":
        mid = input("Mentor ID: ")
        for m in mentors:
            if m.get_id() == mid:
                m.display_details()
    elif ch == "6":
        admin.view_all(students, mentors)
    elif ch == "7":
        break