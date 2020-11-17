DROP TABLE IF EXISTS lines;
CREATE TABLE lines (
    comp_tokens text NOT NULL,
    comp_ids text NOT NULL,
    simp_tokens text NOT NULL,
    simp_ids text NOT NULL,
    edit_labels text NOT NULL,
    new_edit_ids text NOT NULL,
    comp_pos_tags text NOT NULL,
    comp_pos_ids NOT NULL
);
