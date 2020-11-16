DROP TABLE IF EXISTS tokens;
CREATE TABLE tokens (
    token text NOT NULL UNIQUE,
    token_idx integer NOT NULL,
    pos_tag text,
    pos_idx integer
);

DROP TABLE IF EXISTS positions;
CREATE TABLE positions (
    line integer NOT NULL,
    idx integer NOT NULL,
    UNIQUE(line, idx)
);

DROP TABLE IF EXISTS token_positions;
CREATE TABLE token_positions (
    token_id integer NOT NULL,
    position_id integer NOT NULL,
    token_type text NOT NULL,
    PRIMARY KEY(token_id, position_id, token_type)
);

DROP VIEW IF EXISTS complex_tokens;
CREATE VIEW complex_tokens
AS
SELECT
    token,
    token_idx,
    pos_tag,
    pos_idx,
    line,
    idx
FROM tokens
INNER JOIN token_positions on token_positions.token_id = tokens.rowid
INNER JOIN positions on positions.rowid = token_positions.position_id
WHERE token_positions.token_type = 'COMP'
ORDER BY line ASC, idx ASC;

DROP VIEW IF EXISTS simple_tokens;
CREATE VIEW simple_tokens
AS
SELECT
    token,
    token_idx,
    pos_tag,
    pos_idx,
    line,
    idx
FROM tokens
INNER JOIN token_positions on token_positions.token_id = tokens.rowid
INNER JOIN positions on positions.rowid = token_positions.position_id
WHERE token_positions.token_type = 'SIMP'
ORDER BY line ASC, idx ASC;

DROP VIEW IF EXISTS edit_labels;
CREATE VIEW edit_labels
AS
SELECT
    token,
    token_idx,
    pos_tag,
    pos_idx,
    line,
    idx
FROM tokens
INNER JOIN token_positions on token_positions.token_id = tokens.rowid
INNER JOIN positions on positions.rowid = token_positions.position_id
WHERE token_positions.token_type = 'EDIT'
ORDER BY line ASC, idx ASC;
