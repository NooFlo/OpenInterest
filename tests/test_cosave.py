from openinterest.cosave import cosave_candidates
from openinterest.db import Database


def setup_pins(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    b1 = db.create_board("chairs")
    b2 = db.create_board("living-room")
    p_a = db.create_pin("http://x/a.jpg")["id"]
    p_b = db.create_pin("http://x/b.jpg")["id"]
    p_c = db.create_pin("http://x/c.jpg")["id"]
    db.add_pin_to_board(p_a, b1)
    db.add_pin_to_board(p_b, b1)
    db.add_pin_to_board(p_b, b2)
    db.add_pin_to_board(p_c, b2)
    db.add_pin_to_board(p_a, b2)
    return db, p_a, p_b, p_c


def test_cosave_returns_board_mates(tmp_path):
    db, p_a, p_b, p_c = setup_pins(tmp_path)
    cands = cosave_candidates(db, p_a)
    assert p_b in cands and p_c in cands


def test_cosave_excludes_seed(tmp_path):
    db, p_a, *_ = setup_pins(tmp_path)
    assert p_a not in cosave_candidates(db, p_a)


def test_cosave_empty_when_no_boards(tmp_path):
    db = Database(str(tmp_path / "t2.db"))
    p = db.create_pin("http://x/solo.jpg")["id"]
    assert cosave_candidates(db, p) == []
