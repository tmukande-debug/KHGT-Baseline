import tensorflow as tf
import tensorflow as tf
from einops import rearrange, reduce
import numpy as np

paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1

def getParamId():
	global paramId
	paramId += 1
	return paramId

def setIta(ITA):
	ita = ITA

def setBiasDefault(val):
	global biasDefault
	biasDefault = val

def getParam(name):
	return params[name]

def addReg(name, param):
	global regParams
	if name not in regParams:
		regParams[name] = param
	# else:
	# 	print('ERROR: Parameter already exists')

def addParam(name, param):
	global params
	if name not in params:
		params[name] = param

def defineRandomNameParam(shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	name = 'defaultParamName%d'%getParamId()
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	global params
	global regParams
	assert name not in params, 'name %s already exists' % name
	if initializer == 'xavier':
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype, shape=shape,
			initializer=tf.keras.initializers.Initializer(),
			trainable=trainable)
	elif initializer == 'trunc_normal':
		ret = tf.get_variable(name=name, initializer=tf.random.truncated_normal(shape=[int(shape[0]), shape[1]], mean=0.0, stddev=0.03, dtype=dtype))
	elif initializer == 'zeros':
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype,
			initializer=tf.zeros(shape=shape, dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'ones':
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype, initializer=tf.ones(shape=shape, dtype=tf.float32), trainable=trainable)
	elif not isinstance(initializer, str):
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype,
			initializer=initializer, trainable=trainable)
	else:
		print('ERROR: Unrecognized initializer')
		exit()
	params[name] = ret
	if reg:
		regParams[name] = ret
	return ret

def getOrDefineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True, reuse=False):
	global params
	global regParams
	if name in params:
		assert reuse, 'Reusing Param %s Not Specified' % name
		if reg and name not in regParams:
			regParams[name] = params[name]
		return params[name]
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def BN(inp, name=None):
	global ita
	dim = inp.get_shape()[1]
	name = 'defaultParamName%d'%getParamId()
	scale = tf.Variable(tf.ones([dim]))
	shift = tf.Variable(tf.zeros([dim]))
	fcMean, fcVar = tf.nn.moments(inp, axes=[0])
	ema = tf.train.ExponentialMovingAverage(decay=0.5)
	emaApplyOp = ema.apply([fcMean, fcVar])
	with tf.control_dependencies([emaApplyOp]):
		mean = tf.identity(fcMean)
		var = tf.identity(fcVar)
	ret = tf.nn.batch_normalization(inp, mean, var, shift,
		scale, 1e-8)
	return ret

def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False):
	global params
	global regParams
	global leaky
	inDim = inp.get_shape()[1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse)
	if dropout != None:
		ret = tf.nn.dropout(inp, rate=dropout) @ W
	else:
		ret = inp @ W
	if useBias:
		ret = Bias(ret, name=name, reuse=reuse)
	if useBN:
		ret = BN(ret)
	if activation != None:
		ret = Activate(ret, activation)
	return ret

def Bias(data, name=None, reg=False, reuse=False):
	inDim = data.get_shape()[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	temBiasName = temName + 'Bias'
	bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer='zeros', reuse=reuse)
	if reg:
		regParams[temBiasName] = bias
	return data + bias

##relu to Gelu
def ActivateHelp(data, method):
	if method == 'relu':
		ret = tf.nn.relu(data)
	elif method == 'sigmoid':
		ret = tf.nn.sigmoid(data)
	elif method == 'gelu':
		ret = tf.keras.activations.gelu (data, approximate=False)
	elif method == 'softmax':
		ret = tf.nn.softmax(data, axis=-1)
	elif method == 'leakyRelu':
		ret = tf.math.maximum(leaky*data, data)
	elif method == 'twoWayLeakyRelu6':
		temMask = tf.compat.v1.to_float(tf.greater(data, 6.0))
		ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * tf.math.maximum(leaky * data, data)
	elif method == '-1relu':
		ret = tf.math.maximum(-1.0, data)
	elif method == 'relu6':
		ret = tf.math.maximum(0.0, tf.minimum(6.0, data))
	elif method == 'relu3':
		ret = tf.math.maximum(0.0, tf.minimum(3.0, data))
	else:
		raise Exception('Error Activation Function')
	return ret

def Activate(data, method, useBN=False):
	global leaky
	if useBN:
		ret = BN(data)
	else:
		ret = data
	ret = ActivateHelp(ret, method)
	return ret

def Regularize(names=None, method='L2'):
	ret = 0
	if method == 'L1':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.abs(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.abs(regParams[name]))
	elif method == 'L2':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.square(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.square(regParams[name]))
	return ret

def Dropout(data, rate):
	if rate == None:
		return data
	else:
		return tf.nn.dropout(data, rate=rate)

def selfAttention(localReps, number, inpDim, numHeads):
	Q = defineRandomNameParam([inpDim, inpDim], reg=True)
	K = defineRandomNameParam([inpDim, inpDim], reg=True)
	V = defineRandomNameParam([inpDim, inpDim], reg=True)
	rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
	q = tf.reshape(rspReps @ Q, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.reshape(rspReps @ K, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.reshape(rspReps @ V, [-1, 1, number, numHeads, inpDim//numHeads])
	att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim/numHeads), axis=2)
	attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	paramId = 'dfltP%d' % getParamId()
	for i in range(number):
		tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
		# tem2 = FC(tem1, inpDim, useBias=True, name=paramId+'_1', reg=True, activation='relu', reuse=True) + localReps[i]
		rets[i] = tem1 + localReps[i]
	return rets

def lightSelfAttention(localReps, number, inpDim, numHeads):
	Q = defineRandomNameParam([inpDim, inpDim], reg=False)
	rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
	tem = rspReps @ Q
	q = tf.reshape(tem, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.reshape(tem, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.reshape(rspReps, [-1, 1, number, numHeads, inpDim//numHeads])
	att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim/numHeads), axis=2)
	#att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / (inpDim/numHeads), axis=2)
	attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	paramId = 'dfltP%d' % getParamId()
	for i in range(number):
		tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
		rets[i] = tem1 + localReps[i]
	return rets

##tf reduce sum to einsum
##Nystromformer
##CosNormer
##Transevolve

def lightSelfAttention1(localReps, number, inpDim, numHeads):
	Q = defineRandomNameParam([inpDim, inpDim], reg=False)
	rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
	tem = rspReps @ Q
	q = tf.reshape(tem, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.reshape(tem, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.reshape(rspReps, [-1, 1, number, numHeads, inpDim//numHeads])
	att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim/numHeads), axis=2)
	attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	paramId = 'dfltP%d' % getParamId()
	for i in range(number):
		tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
		rets[i] = tem1 + localReps[i]
	return rets



class MoorePenrosePseudoinverse(tf.keras.layers.Layer):
    def __init__(self, iteration=6, **kwargs):
        super(MoorePenrosePseudoinverse, self).__init__(**kwargs)

        self.iteration = iteration

    def call(self, inputs, **kwargs):
        abs_inputs = tf.abs(inputs)
        cols = tf.math.reduce_sum(abs_inputs, axis=-1)
        rows = tf.math.reduce_sum(abs_inputs, axis=-2)
        z = rearrange(inputs, "... i j -> ... j i") / (
            tf.math.reduce_max(cols) * tf.math.reduce_max(rows)
        )

        identity = tf.eye(z.shape[-1])
        identity = rearrange(identity, "i j -> () i j")

        for _ in range(self.iteration):
            inputs_bbm_z = inputs @ z
            z = (
                0.25
                * z
                @ (
                    13 * identity
                    - (
                        inputs_bbm_z
                        @ (
                            15 * identity
                            - (inputs_bbm_z @ (7 * identity - inputs_bbm_z))
                        )
                    )
                )
            )

        return z


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, fn, **kwargs):
        super(PreNorm, self).__init__(**kwargs)
        self.fn = fn
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        inputs = self.norm(inputs)
        return self.fn(inputs, **kwargs)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, mult=4, dropout=0.0, **kwargs):
        super(FeedForward, self).__init__(**kwargs)

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim * mult),
                tf.keras.layers.Activation(tf.nn.gelu),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(dim),
            ]
        )

    def call(self, inputs):
        return self.net(inputs)
