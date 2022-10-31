import pytest
import numpy as np
import features
import sys

npdummy = np.ndarray(0)

def dump_feat(feat: features.Features, prefix: str):
    # get_label_instance
    for label in [1,2,3]:
        for j in range(0,4):
            dict = feat.get_label_instance(label_code=label, ith=j)
            sys.stderr.write(f" {prefix} label={label} ith={j}:  confidence={dict['confidence']}\n")

def test_sorting_one_confidence():
    feat = features.Features(None, "/dev/null")
    feat.start("sim_dummy_foo")
    feat.put(1, ith=0, region_size=0.1, confidence=0.2, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.2, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.finish()
    arr_feat = feat.get_label_instances_confidence_descending(1)
    assert arr_feat[0]["confidence"] == 0.2
    assert feat.is_production()

def test_sorting_one_x():
    feat = features.Features(None, "/dev/null")
    feat.start("sim_dummy_foo")
    feat.put(1, ith=0, region_size=0.1, confidence=0.2, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.2, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.finish()
    arr_feat = feat.get_label_instances_x_ascending(1)
    assert arr_feat[0]["normalized_x"] == 0
    assert arr_feat[3]["normalized_x"] == 0.2

def test_sorting_one_y():
    feat = features.Features(None, "/dev/null")
    feat.start("sim_dummy_foo")
    feat.put(1, ith=0, region_size=0.1, confidence=0.2, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.2, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.finish()
    arr_feat = feat.get_label_instances_y_ascending(1)
    assert arr_feat[0]["normalized_y"] == 0
    assert arr_feat[3]["normalized_y"] == 0.2

def test_sorting_multi_confidence():
    feat = features.Features(None, "/dev/null")
    feat.start("sim_dummy_foo")
    feat.put(1, ith=0, region_size=0.1, confidence=0.2, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.2, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.put(1, ith=1, region_size=0.1, confidence=0.4, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.2, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.put(1, ith=2, region_size=0.1, confidence=0.6, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.2, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.put(1, ith=3, region_size=0.1, confidence=0.5, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.2, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.finish()

    arr_feat = feat.get_label_instances_confidence_descending(1)

    for j in range(0,4):
        print(f" after sort: [{j}]:  confidence={arr_feat[j]['confidence']}")

    assert arr_feat[0]["confidence"] == 0.6
    assert arr_feat[3]["confidence"] == 0.2


def test_sorting_multi_x():
    feat = features.Features(None, "/dev/null")
    feat.start("sim_dummy_foo")
    feat.put(2, ith=0, region_size=0.1, confidence=0.2, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.1, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.put(2, ith=1, region_size=0.1, confidence=0.4, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.9, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.put(2, ith=2, region_size=0.1, confidence=0.6, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.5, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.put(2, ith=3, region_size=0.1, confidence=0.5, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.6, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.finish()
    arr_feat = feat.get_label_instances_x_ascending(2)
    assert arr_feat[0]["normalized_x"] == 0.1
    assert arr_feat[3]["normalized_x"] == 0.9


def test_sorting_multi_x():
    feat = features.Features(None, "/dev/null")
    feat.start("sim_dummy_foo")
    feat.put(3, ith=0, region_size=0.1, confidence=0.2, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.1, normalized_y=0.1, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.put(3, ith=1, region_size=0.1, confidence=0.4, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.9, normalized_y=0.2, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.put(3, ith=2, region_size=0.1, confidence=0.6, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.5, normalized_y=0.9, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.put(3, ith=3, region_size=0.1, confidence=0.5, aspect_ratio=1.0, page_width_fraction=0.1,
             page_height_fraction=0.1, normalized_x=0.6, normalized_y=0.8, tag_rect_x0=10, tag_rect_y0=10,
             tag_rect_x1=20, tag_rect_y1=20, tag_rect_image=npdummy, tag_comments="")
    feat.finish()
    arr_feat = feat.get_label_instances_y_ascending(3)
    assert arr_feat[0]["normalized_y"] == 0.1
    assert arr_feat[3]["normalized_y"] == 0.9
