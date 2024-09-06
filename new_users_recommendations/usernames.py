import random
import csv

# List of possible first names
first_names = ["Rahul", "Daksh", "Sachin", "Rohan", "Ayush", "Kunal", "Harsh", "Vishal", "Abhishek", "Sagar"]

# Generate 10 random usernames
usernames = [f"{random.choice(first_names)}{random.randint(1, 100)}" for _ in range(10)]

# Save the usernames to a CSV file
with open('usernames.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["username"])  # header
    for username in usernames:
        writer.writerow([username])

print("Usernames saved to usernames.csv")


