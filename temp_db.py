import mysql.connector as db
con=db.connect(
    host="localhost",
    user="root",
    password="",
    database="student")
cursor=con.cursor()
q="select * from studentdetails"
cursor.execute(q)
j=cursor.fetchall()
for i in j:
    print(i[0])
