import mysql.connector


def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Truong@1866",
        database="face_recognition"
    )
def insert_student(student_id, full_name, major):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO students (student_id, full_name, major) VALUES (%s, %s, %s)",
        (student_id, full_name, major)
    )
    conn.commit()
    cursor.close()
    conn.close()
def get_student(student_id):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute(
        "SELECT student_id, full_name, major FROM students WHERE student_id=%s",
        (student_id,)
    )

    row = cur.fetchone()

    cur.close()
    conn.close()
    return row
