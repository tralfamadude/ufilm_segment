"""
Abstract post processing model.
INPUT: predictions from dhSegment
OUTPUT: what kind of page { other(0), start(1), refs(2), toc(3) }
"""

class AbsPostModel:
    def train(self, X, y):
        pass

    def predict(self, X):
        pass

    def save(self):
      pass

    def load(self):
      pass

