import pytest

import excerpt as ex

def test_excerpt():
    ex1 = ex.Excerpt("jid", "pid", 0, 0)
    assert ex1.get_page_number() == 0
    pe = ex.IssueExcerpts()
    pe.put(ex1)
    ex1 = ex.Excerpt("jid", "pid", 1, 1)
    ex1.set_title("My Title")
    ex1.set_authors("John Doe")
    ex1.set_explanation("something")
    pe.put(ex1)
    assert pe.is_contiguous()
    # test save and load
    fname = "/tmp/test_issue_excerpt.jsonl"
    pe.save(fname)
    pe_loaded = ex.IssueExcerpts().load(fname)
    pe_loaded.dump()
    score, expl = pe.compare(pe_loaded)
    assert score == 1.0

    ex2 = ex.Excerpt("jid", "pid", 0, 0)
    pe = ex.IssueExcerpts()
    pe.put(ex2)
    ex2 = ex.Excerpt("jid", "pid", 3, 1)
    pe.put(ex2)
    assert not pe.is_contiguous()

    ex3 = ex.Excerpt("jid", "pid", 0, 1)
    ex4 = ex.Excerpt("jid", "pid", 0, 1)
    assert ex3.same_page(ex4)
    assert ex3.same_type(ex4)

    try:
        ex5 = ex.Excerpt("jid", "pid", 0, 0)
        ex6 = ex.Excerpt("jid", "pid", 1, 0)
        ex5.compare(ex6)  # different pages, invalid comparison
        assert False
    except KeyError:
        # correct
        pass

    try:
        ex6 = ex.Excerpt("jid", "pidA", 0, 0)
        ex7 = ex.Excerpt("jid", "pidB", 0, 0)
        ex6.compare(ex7) # different issue, hence different page, invalid comparison
    except KeyError:
        pass # correct

    #  test comparison to self and nearly matching author or title
    ex8 = ex.Excerpt("jid", "pid", 0, 1)
    ex8.set_title("title of article")
    ex8.set_authors("John Smith, Jane Doe")
    score = ex8.compare(ex8) # compare to self
    assert score == 1.0
    ex9 = ex.Excerpt("jid", "pid", 0, 1)
    ex9.set_title("title of articl")
    ex9.set_authors("John Smith, Jane Do")
    score = ex8.compare(ex9)
    assert 0.94 < score < 0.97  # close match

    #  test comparison to self and nearly matching refs
    ex10 = ex.Excerpt("jid", "pid", 0, 2)
    ex10.set_refs("Weizenbaum, J. ELIZA—a computer program for the study of natural language communication between man and machine. In Communications of the ACM, 1966.")
    score = ex10.compare(ex10)  # compare to self
    assert score == 1.0
    ex11 = ex.Excerpt("jid", "pid", 0, 2)
    ex11.set_refs("izenbaum, J. ELIZA—a computer program for the study of natural language communication between man and machine. In Communications of the ACM, 1966.")
    score = ex10.compare(ex11)
    assert 0.99 < score < 0.999  # close match




