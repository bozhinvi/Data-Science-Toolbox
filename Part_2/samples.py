
__author__="weimer"
__date__ ="$Feb 13, 2009 4:33:11 PM$"


TRAIN, TEST, PREDICT = range(3)

class Sample(object):
    def __init__(self, kind, ind = None, row=None, col=None, rowcounter = None, rawPrediction=None, prediction=None, y=None):
        self.row = row
        self.col = col
        self.ind = ind
        self.rowcounter = rowcounter
        self.rawPrediction = rawPrediction
        self.prediction = prediction
        self.y = y
        self.kind = kind
        assert(self.kind in (TRAIN, TEST, PREDICT))

    def __str__(self):
        if self.kind == TRAIN:
            k = 'TRAIN  '
        elif self.kind == TEST:
            k = 'TEST   '
        elif self.kind == PREDICT:
            k = 'PREDICT'

        return 'Sample(kind=%s, row=%s, col=%s, rawPrediction=%s, prediction=%s, y=%s)'%(k, self.row, self.col, self.rawPrediction, self.prediction, self.y)

    def hasY(self):
        return self.y is not None

    def getKind(self):
        return self.kind



# This clas is for Tensor factorization so we need an index of arbitrary length (list) 
class TSample(object):
    def __init__(self, kind, ind = None, rawPrediction = None, prediction = None, y = None):
        self.ind = ind
        self.rawPrediction = rawPrediction
        self.prediction = prediction
        self.y = y
        self.kind = kind
        assert(self.kind in (TRAIN, TEST, PREDICT))

    def __str__(self):
        if self.kind == TRAIN:
            k = 'TRAIN  '
        elif self.kind == TEST:
            k = 'TEST   '
        elif self.kind == PREDICT:
            k = 'PREDICT'

        return 'Sample(kind=%s, ind =%s, rawPrediction=%s, prediction=%s, y=%s)'%(k, self.ind, self.rawPrediction, self.prediction, self.y)
