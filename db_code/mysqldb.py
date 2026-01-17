import mysql.connector
from mysql.connector import pooling

# Tạo một Pool duy nhất cho toàn ứng dụng
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "Truong@1866",
    "database": "face_recognition",
    "charset": "utf8"
}

connection_pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    **db_config
)

def get_connection():
    # Lấy một kết nối có sẵn từ pool (Cực nhanh)
    return connection_pool.get_connection()

def insert_box_user(user_id, box_number):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO boxes_users (user_id, box_number) VALUES (%s, %s)",
        (user_id, box_number)
    )
    conn.commit()
    cursor.close()
    conn.close()

def get_box_user(user_id):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute(
        "SELECT user_id, box_number, time_in FROM boxes_users WHERE user_id = %s",
        (user_id,)
    )

    row = cur.fetchone()

    cur.close()
    conn.close()
    return row