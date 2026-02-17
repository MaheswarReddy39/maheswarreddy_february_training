# Smart Login System
correct_username = "admin"
correct_password = "pass@123"
active = True
username = input("Enter username: ")
password = input("Enter password: ")
if username == correct_username:
    if password == correct_password:
        if active:
            print("Login Successful")
        else:
            print("Account Disabled")    
    else:
        print("Wrong Credentials")
else:
    print("Wrong Credentials")