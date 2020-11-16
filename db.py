import pandas
import sqlite3

import label_edits

def create_connection(db_file):
    conn = sqlite3.connect(db_file, isolation_level=None)
    return conn

def begin_transaction(conn):
    conn.execute('BEGIN;')

def savepoint(conn, name):
    conn.execute(f'SAVEPOINT {name};')

def release(conn, name):
    conn.execute(f'RELEASE {name};')

def rollback_to(conn, name):
    conn.execute(f'ROLLBACK TO {name};')

def commit(conn):
    conn.execute('COMMIT;')

def num_lines(conn):
    sql = 'SELECT MAX(line) FROM positions'
    (max_line,) = conn.execute(sql).fetchone()
    
    return max_line + 1 if max_line else 0

def insert_line(conn, tokens, line_idx, token_type, pos_tags=None):
    select_token = 'SELECT rowid FROM tokens WHERE token = ?'
    insert_tokens = """
        INSERT OR IGNORE INTO tokens (token, token_idx, pos_tag, pos_idx)
        VALUES (?, ?, ?, ?)
    """

    select_position = 'SELECT rowid FROM positions WHERE line = ? AND idx = ?'
    insert_positions = 'INSERT OR IGNORE INTO positions (line, idx) VALUES (?, ?)'

    insert_token_positions = """
        INSERT INTO
            token_positions (token_id, position_id, token_type)
        VALUES (?, ?, ?)
    """
    for idx, token in enumerate(tokens):
        text, token_idx = token
        pos_tag = None
        pos_idx = None
        if(pos_tags):
            (_, pos_tag), pos_idx = pos_tags[idx]

        cursor = conn.cursor()
        cursor.execute(insert_tokens, (text, token_idx, pos_tag, pos_idx))
        (token_id,) = cursor.execute(select_token, (text,)).fetchone()

        cursor = conn.cursor()
        cursor.execute(insert_positions, (line_idx, idx))
        (position_id,) = cursor.execute(select_position, (line_idx, idx)).fetchone()

        cursor = conn.cursor()
        cursor.execute(insert_token_positions, (token_id, position_id, token_type))

def save_dataframe(conn, df):
    for line, row in df.iterrows():
        comp_tokens = list(zip(row['comp_tokens'], map(int, row['comp_ids'])))
        pos_tags = list(zip(row['comp_pos_tags'], map(int, row['comp_pos_ids'])))
        simp_tokens = list(zip(row['simp_tokens'], map(int, row['simp_ids'])))
        edit_labels = list(zip(row['edit_labels'], map(int, row['new_edit_ids'])))
        
        with conn:
            insert_line(conn, tokens=comp_tokens, line_idx=line, token_type="COMP", pos_tags=pos_tags)
            insert_line(conn, tokens=simp_tokens, line_idx=line, token_type="SIMP")
            insert_line(conn, tokens=edit_labels, line_idx=line, token_type="EDIT")

def load_dataframe(conn):
    data = {
        'comp_tokens': complex_tokens(conn),
        'comp_ids': complex_ids(conn),
        'comp_pos_tags': complex_pos_tags(conn),
        'comp_pos_ids': complex_pos_ids(conn),
        'simp_tokens': simple_tokens(conn),
        'simp_ids': simple_ids(conn),
        'edit_labels': edit_labels(conn),
        'new_edit_ids': edit_label_ids(conn)
    }

    return pandas.DataFrame.from_dict(data)
    

def get_tokens(conn, view_name, column='token'):
    sql = f"""
        SELECT
            GROUP_CONCAT({column}, ' ') line_text
        FROM {view_name}
        GROUP BY line
    """
    rows = conn.execute(sql).fetchall()
    return map(lambda x: x[0].split(), rows)

def complex_tokens(conn):
    return get_tokens(conn, 'complex_tokens')

def complex_ids(conn):
    return get_tokens(conn, 'complex_tokens', column='token_idx')

def complex_pos_tags(conn):
    return get_tokens(conn, 'complex_tokens', column='pos_tag')

def complex_pos_ids(conn):
    return get_tokens(conn, 'complex_tokens', column='pos_idx')

def simple_tokens(conn):
    return get_tokens(conn, 'simple_tokens')

def simple_ids(conn):
    return get_tokens(conn, 'simple_tokens', column='token_idx')

def edit_labels(conn):
    return get_tokens(conn, 'edit_labels')

def edit_label_ids(conn):
    return get_tokens(conn, 'edit_labels', column='token_idx')




def insert_edit_labels(conn):
    comp_tokens = complex_tokens(conn)
    simp_tokens = simple_tokens(conn)

    for line_idx, (comp, simp) in enumerate(zip(comp_tokens, simp_tokens)):
        edits = label_edits.sent2edit(comp, simp)
        insert_line(conn=conn, line=edits, line_idx=line_idx, token_type="EDIT")

