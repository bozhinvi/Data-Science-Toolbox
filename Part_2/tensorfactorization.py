import logging
import numpy
import pdb
import math
import samples

class LossFunction(object):

    def computeLossAndGradient(self, y, f):
        raise NotImplementedError()

    def predict(self, f):
        return f

class LeastSquares(LossFunction):

    def computeLossAndGradient(self, y, f):
        v= (f-y)
        return (v**2, v+v)



class DataSource(object):

    def __iter__(self):
        raise NotImplementedError('This subclass of DataSource does not implement __iter__(). Something is wrong.')

class AdomDataSource(DataSource):

    def __init__(self, filename, kind=samples.TRAIN):
        self.__filename = filename
        self.__kind  = kind
        self.__f = open(self.__filename, 'r')

    def __iter__(self):

        for line in self.__f:
            entries = line.split(',')# its comma on these files
            #assert(len(entries) == 7)
            i = int(entries[0]) - 1
            j = int(entries[1]) - 1
            c1 = int(entries[2]) - 1
            c2  = int(entries[3]) - 1
            c3  = int(entries[4]) - 1
            c4   = int(entries[5]) - 1
            c5  = int(entries[6]) - 1
            rating  = int(entries[7])


            yield samples.Sample(kind=self.__kind, ind = [i,j, c1, c2, c3, c4, c5 ], y=rating)


class Model(object):
    def getM(self, j):
        raise NotImplementedError

    def getU(self, i):
        raise NotImplementedError

    def getUandM(self, sample):
        return (self.getU(sample.row), self.getM(sample.col))

    def getUandMandF(self, sample):
        return(self.getU(sample.row), self.getM(sample.col), self.getMF(sample.col))

    def getEmptyVector(self):
        raise NotImplementedError

    def getAllMIDs(self):
        raise NotImplementedError

    def getAllUIDs(self):
        raise NotImplementedError

    def updateUandM(self, sample, u, m):
        pass


class TensorModel(Model):
    def __init__(self, nFactors, nDimensions):
        self.__modelArrays = []
        numpy.random.seed(5)
        for nFact, nDims in zip(nFactors, nDimensions):
            self.__modelArrays.append(abs(numpy.random.normal(4.0/nFact, 1.5/nFact, size=[nDims, nFact])))

        self.__Tensor = 0.1*numpy.random.normal(0/nFact, 1.5, size=[5,5]) # this is a dummy tensor, for the real think uncomment previous line
        #self.__Tensor = numpy.ones(nFactors)
        #abusing the interface here getU returns now a list of arrays containing the rows specified in ind which is a list
    def getU(self, ind):
        return [k[i,] for (k,i) in zip(self.__modelArrays, ind)]
    #abusing the interface again j is a list of indexes in this case
    def getM(self, j):
        return self.__Tensor[tuple(j)]

    def getOne(self, i, ind):
        return self.__modelArrays[i][ind,]

    def getUandM(self, sample):
        return (self.getU(sample.ind), self.__Tensor)

    def getFullModel(self):
        return self.__modelArrays


class LearningRateStrategy(object):

    def getLearningRates(self, sample):
        raise NotImplementedError

class ConstantLearningRate(LearningRateStrategy):

    def __init__(self, lr):
        self.learningrate = lr

    def getLearningRates(self, sample):
        return (self.learningrate, self.learningrate)

class LambdaModel(object):

    def getLambdas(self, sample):
        raise NotImplementedError

class dualLambdaModel(LambdaModel):
    def __init__(self, lbda, tenlbda):
        self.__lbda = lbda
        self.__tenlbda = tenlbda

    def getLambdas(self, sample):
        return (self.__lbda, self.__tenlbda)


class ConstantLambdaModel(LambdaModel):

    def __init__(self, lbda):
        self.__lbda = lbda

    def getLambdas(self, sample):
        return (self.__lbda, self.__lbda)


def l2(v, gradient=None):
    value    = numpy.dot(v,v)
    gradient = v+v # Should not allocate memory if gradient is given of the right size
    return (value, gradient)



class DataSink(object):

    def append(self, x):
        raise NotImplementedError

    def getResults(self):
        raise NotImplementedError


class IgnoringSink(DataSink):
    def append(self, x):
        pass

    def getResults(self):
        return {}


class PrintingSink(DataSink):

    def append(self, x):
        print x

    def getResults(self):
        return {}

class FileSink(DataSink):

    def __init__(self, file):
        self.__file = file

    def append(self, x):
        self.__file.write(str(x))
        self.__file.write('\n')

    def getResults(self):
        return {}


class ContextFileSink(DataSink):

    def __init__(self, file):
        self.__file = file

    def append(self, x):
        self.__file.writelines("%s " % item for item in x.ind)
        self.__file.writelines("%s " % x.y)
        self.__file.writelines("%s\n" % x.rawPrediction)


    def getResults(self):
        return {}

class MetaSink(DataSink):
    '''
        A DataSink that sends TRAIN, TEST and PREDICT Samples into different
        sinks.
    '''
    def __init__(self, trainSink=IgnoringSink(), testSink=IgnoringSink(), predictSink=IgnoringSink()):
        self.trainSink = trainSink
        self.testSink = testSink
        self.predictSink = predictSink

    def append(self, sample):
        if sample.kind == samples.TRAIN:
            self.trainSink.append(sample)
        elif sample.kind == samples.TEST:
            self.testSink.append(sample)
        elif sample.kind == samples.PREDICT:
            self.predictSink.append(sample)
        else:
            logger.warn('The sample was of unkown kind %s: %s' % (sample.kind, sample))

    def getResults(self):
        raise NotImplementedError('Calling getResults on a MetaSink is unsupported')


class SinkChain(DataSink):
    '''
        A chain of sinks.
        Samples sent to it will on to all sinks in the chain.
    '''
    def __init__(self, sinks=[]):
        self.sinks = list(sinks)


    def addSink(self, sink):
        self.sinks.append(sink)

    def append(self, sample):
        map(lambda x: x.append(sample), self.sinks)

    def getResults(self):
        raise NotImplementedError('Calling getResults on a SinkChain is unsupported')


class MAErrorSink(DataSink):

    def __init__(self):
        self.entries = []

    def append(self, x):
        if x.hasY():
            v = abs(x.y-x.prediction)
            self.entries.append(v)

    def getMAE(self):
        return sum(self.entries) / len(self.entries)

    def clear(self):
        self.entries = []


    def getResults(self):
        return {'MAE':self.getMAE()}


    rmse = property(getMAE, doc='Mean Absolute Error')



class RMSErrorSink(DataSink):

    '''
        A datasink that computes the rooted mean squared average error.
    '''
    def __init__(self):
        self.entries = []

    def append(self, x):
        if x.hasY():
            v = (x.y-x.prediction) ** 2
            self.entries.append(v)

    def getRMSE(self):
        return numpy.sqrt(sum(self.entries) / len(self.entries))

    def clear(self):
        self.entries = []

    def getResults(self):
        return {'RMSE':self.getRMSE()}

    rmse = property(getRMSE, doc='Root Mean Squared Error')



class OnlineTensorCofi(object):

    def __init__(self, model, lossFunction, regularizer, lbdaModel, lrStrategy):
        self.__loss       = lossFunction
        self.__model      = model
        self.__lbdaModel  = lbdaModel
        self.__reg        = regularizer
        self.__lrStrategy = lrStrategy



    def updateSimple(self, sample, ma):

        lr_ma, lr_ten = self.__lrStrategy.getLearningRates(sample)
        # Update the loss
        l, g = self.__loss.computeLossAndGradient(y=sample.y, f=sample.rawPrediction)

        l_u, l_m = self.__lbdaModel.getLambdas(sample)
        ran = range(len(ma))
        for i in ran:
         ind = [p for p in ran if p not in [i]]
         ma[i] -= lr_ma[i] *( g * self.vecMult(ma, ind) + l_u[i] * ma[i])



    def run(self, data, sink=IgnoringSink()):
        lp          = self.__loss.predict
        updateUandM = self.__model.updateUandM

        def process(sample):
            ma, ten = self.__model.getUandM(sample)
            sample.rawPrediction = self.tenDotSimple(ma, range(len(ma)))
            sample.prediction = lp(sample.rawPrediction)
            if sample.kind == samples.TRAIN and sample.hasY():
                #self.update(sample, ma, ten)
                self.updateSimple(sample, ma)

            sink.append(sample)

        map(process, data)

    def tenDotSimple(self, ma, ind):
        res = ma[ind[0]] * ma[ind[1]]
        for i in ind[2:]:
            res = res*ma[i]
        return numpy.sum(res)


    def vecMult(self, ma, ind):
        res = ma[ind[0]] * ma[ind[1]]
        for i in ind[2:]:
            res = res*ma[i]
        return res



loss    = LeastSquares()
model   = TensorModel(nFactors=[4 ,4 , 4, 4, 4, 4, 4], nDimensions=[84,192, 3, 5, 4, 4, 5])
regularizer = l2
lbdaModel   = dualLambdaModel( [0.0005/x for x in [84,192, 3, 5, 4, 4, 5]], 0.00001) # 0.001
lrStrategy = ConstantLearningRate([0.00001*x for x in [84,192, 3, 5, 4, 4, 5]])
o = OnlineTensorCofi(model, loss, regularizer, lbdaModel, lrStrategy)
testSrc = [x for x in AdomDataSource('/Users/alexis/svn/gpfactorization/data/adom/adom-5folds/fold2-test.csv', kind=samples.TEST)]
trainSrc = [x for x in AdomDataSource('/Users/alexis/svn/gpfactorization/data/adom/adom-5folds/fold2-train.csv', kind=samples.TRAIN)]
sink = MetaSink(trainSink=RMSErrorSink(), testSink=RMSErrorSink(), predictSink=RMSErrorSink() )
results = open('result-adom.csv','w')
results.write('ITER TRAINMAE TESTMAE \n')


for i in xrange(20):
    o.run(trainSrc, sink)
    print i," Train RMSE: ", sink.trainSink.getRMSE(), "\n"
    o.run(testSrc, sink)
    print i," Test RMSE: ", sink.testSink.getRMSE(), "\n"
    results.write('%s %s %s\n'%(i, sink.trainSink.getRMSE(), sink.testSink.getRMSE()))

    sink.testSink.clear()
    sink.trainSink.clear()
