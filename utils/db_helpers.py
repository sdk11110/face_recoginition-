# 获取下一个可用的最小ID
def get_next_available_id(cursor, table_name):
    """
    获取表中下一个可用的最小ID
    """
    cursor.execute(f"SELECT id FROM {table_name} ORDER BY id")
    # Fetch all results before proceeding
    rows = cursor.fetchall()
    used_ids = [row[0] for row in rows]

    if not used_ids:
        return 1

    # 找出第一个可用的ID
    next_id = 1
    for current_id in used_ids:
        if current_id != next_id:
            return next_id
        next_id += 1
    # If all IDs from 1 up to max(used_ids) are used, return the next one
    return next_id 