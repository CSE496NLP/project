DROP INDEX IF EXISTS idx_token;
CREATE INDEX idx_token
ON tokens(token);

DROP INDEX IF EXISTS idx_token_type;
CREATE INDEX idx_token_type
ON token_positions(token_type);

DROP INDEX IF EXISTS idx_position;
CREATE INDEX idx_position
ON positions(line, idx);
