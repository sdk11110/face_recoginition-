-- 删除已有数据库
DROP DATABASE IF EXISTS attendance_db;

-- 重新创建数据库
CREATE DATABASE attendance_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE attendance_db;

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL,
    name VARCHAR(100) NOT NULL,
    role ENUM('admin', 'user') NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建考勤记录表
CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    check_in_time DATETIME NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建人脸编码表
CREATE TABLE IF NOT EXISTS known_faces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    face_encoding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建索引
CREATE INDEX idx_check_in_time ON attendance(check_in_time);
CREATE INDEX idx_name ON attendance(name);
CREATE INDEX idx_known_faces_name ON known_faces(name);

-- 创建获取下一个可用ID的函数
DELIMITER //
CREATE FUNCTION get_next_id(table_name VARCHAR(50)) 
RETURNS INT
READS SQL DATA
BEGIN
    DECLARE next_id INT;
    
    IF table_name = 'users' THEN
        SELECT COALESCE(MIN(t1.id + 1), 1) INTO next_id
        FROM users t1
        LEFT JOIN users t2 ON t1.id + 1 = t2.id
        WHERE t2.id IS NULL;
    ELSEIF table_name = 'attendance' THEN
        SELECT COALESCE(MIN(t1.id + 1), 1) INTO next_id
        FROM attendance t1
        LEFT JOIN attendance t2 ON t1.id + 1 = t2.id
        WHERE t2.id IS NULL;
    ELSEIF table_name = 'known_faces' THEN
        SELECT COALESCE(MIN(t1.id + 1), 1) INTO next_id
        FROM known_faces t1
        LEFT JOIN known_faces t2 ON t1.id + 1 = t2.id
        WHERE t2.id IS NULL;
    ELSE
        SET next_id = 1;
    END IF;
    
    RETURN COALESCE(next_id, 1);
END //
DELIMITER ;

-- 创建插入用户的存储过程
DELIMITER //
CREATE PROCEDURE insert_user(
    IN p_username VARCHAR(50),
    IN p_password VARCHAR(100),
    IN p_name VARCHAR(100),
    IN p_role VARCHAR(10)
)
BEGIN
    INSERT INTO users (username, password, name, role)
    VALUES (p_username, p_password, p_name, p_role);
    
    SELECT LAST_INSERT_ID() as user_id;
END //
DELIMITER ;

-- 创建插入人脸数据的存储过程
DELIMITER //
CREATE PROCEDURE insert_face(
    IN p_user_id INT,
    IN p_name VARCHAR(100),
    IN p_face_encoding BLOB
)
BEGIN
    INSERT INTO known_faces (user_id, name, face_encoding)
    VALUES (p_user_id, p_name, p_face_encoding);
    
    SELECT LAST_INSERT_ID() as face_id;
END //
DELIMITER ;

-- 创建插入考勤记录的存储过程
DELIMITER //
CREATE PROCEDURE insert_attendance(
    IN p_user_id INT,
    IN p_name VARCHAR(100),
    IN p_check_in_time DATETIME
)
BEGIN
    INSERT INTO attendance (user_id, name, check_in_time)
    VALUES (p_user_id, p_name, p_check_in_time);
    
    SELECT LAST_INSERT_ID() as attendance_id;
END //
DELIMITER ;

-- 创建软删除用户的存储过程
DELIMITER //
CREATE PROCEDURE soft_delete_user(
    IN p_user_id INT
)
BEGIN
    UPDATE users SET is_active = FALSE WHERE id = p_user_id;
    UPDATE known_faces SET is_active = FALSE WHERE user_id = p_user_id;
    UPDATE attendance SET is_active = FALSE WHERE user_id = p_user_id;
END //
DELIMITER ;

-- 插入默认管理员账户
CALL insert_user('admin', '123456', '管理员', 'admin');