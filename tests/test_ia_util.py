import pytest

import ia_util as util

def test_parse_page_id():
    tu = util.TextUtil()
    page_id = "sim_journal-of-thought_2004_fall_39_3_0009"
    jid, iid, page_number = tu.parse_page_id(page_id)
    assert jid == "sim_journal-of-thought"
    assert iid == "sim_journal-of-thought_2004_fall_39_3"
    assert page_number == 9

