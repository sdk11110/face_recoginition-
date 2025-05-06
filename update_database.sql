USE attendance_db;

-- 检查列是否存在，如果不存在则添加
SET @dbname = 'attendance_db';
SET @tablename = 'known_faces';
SET @columnname = 'feature_dim';
SET @preparedStatement = (SELECT IF(
  (
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    WHERE
      (table_name = @tablename)
      AND (table_schema = @dbname)
      AND (column_name = @columnname)
  ) > 0,
  'SELECT 1',
  'ALTER TABLE known_faces ADD COLUMN feature_dim INT DEFAULT 128'
));
PREPARE alterIfNotExists FROM @preparedStatement;
EXECUTE alterIfNotExists;
DEALLOCATE PREPARE alterIfNotExists;

-- 更新现有记录的特征维度
UPDATE known_faces SET feature_dim = 128 WHERE id > 0 AND feature_dim IS NULL;

-- 检查约束是否存在，如果不存在则添加
SET @constraint_name = 'chk_feature_dim';
SET @preparedStatement = (SELECT IF(
  (
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
    WHERE
      (table_name = @tablename)
      AND (table_schema = @dbname)
      AND (constraint_name = @constraint_name)
  ) > 0,
  'SELECT 1',
  'ALTER TABLE known_faces ADD CONSTRAINT chk_feature_dim CHECK (feature_dim IN (128, 512))'
));
PREPARE addConstraintIfNotExists FROM @preparedStatement;
EXECUTE addConstraintIfNotExists;
DEALLOCATE PREPARE addConstraintIfNotExists; 