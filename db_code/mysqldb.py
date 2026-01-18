import mysql
import mysql.connector as conn
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

def insert_box_user(user_id):
    insert_result = False
    connection = get_connection()
    cursor = connection.cursor()
    try:
        query_data = "SELECT * FROM boxes_users WHERE user_id = %s"
        cursor.execute(query_data, (user_id,))
        data = cursor.fetchone()
    except mysql.connector.Error as e:
        print("Xuất hiện lỗi khi truy vấn dữ liệu" + e)
        data =None
    try:
        if data is None:
            insert_result = True
            query_box_id = "SELECT box_id FROM queue WHERE id = '1'"
            cursor.execute(query_box_id)
            result_query_box_id = cursor.fetchone()
            if result_query_box_id:
                box_id = int(result_query_box_id[0])
                query_insert = "INSERT INTO boxes_users (user_id, box_number) VALUES (%s, %s)"
                cursor.execute(query_insert, (user_id,box_id))
                connection.commit()
                delete_box_id = "DELETE FROM queue WHERE id = '1'"
                cursor.execute(delete_box_id)
                connection.commit()
    except mysql.connector.Error as err:
        print("Lỗi khi thêm người mới" + err)
    finally:
        cursor.close()
        connection.close()
        return insert_result

def delete_box_user(user_id):
    delete_result = False
    connection = get_connection()
    cursor = connection.cursor()
    try:
        query_data ="SELECT box_number FROM boxes_users WHERE user_id = %s"
        cursor.execute(query_data, (user_id,))
        result_query = cursor.fetchone()
        if result_query:
            delete_result = True
            box_target = result_query[0]
            insert_box = "INSERT INTO queue (box_id) VALUES (%s)"
            cursor.execute(insert_box, (box_target,))
            connection.commit()
            query_delete = "DELETE FROM boxes_users WHERE user_id = %s"
            cursor.execute(query_delete, (user_id,))
            connection.commit()
    except mysql.connector.Error as err:
        print(err)
    finally:
        cursor.close()
        connection.close()
        return delete_result
