from marshmallow import Schema, fields
from marshmallow.validate import Length


class LabelConfiguration(Schema):
    labels = fields.List(fields.String)


class NormalDistributionConfiguration(Schema):
    mean = fields.Float(default=0.0, missing=0.0)
    std = fields.Float(default=0.0, missing=0.001)


class RNNConfiguration(Schema):
    rnn_type = fields.String(default="lstm", load_from="type")
    bidirectional = fields.Boolean(default=True)
    size = fields.Integer(default=512)
    layers = fields.Integer(default=4)
    noise = fields.Nested(NormalDistributionConfiguration, default=None, missing=None)
    batch_norm = fields.Boolean(default=False, missing=False)


class JasperConfiguration(Schema):
    filters = fields.Integer(default=32)
    repeat = fields.Integer(default=1)
    kernel = fields.List(fields.Integer, default=[11], validate=Length(equal=1))
    stride = fields.List(fields.Integer, default=[1], validate=Length(equal=1))
    dilation = fields.List(fields.Integer, default=[1], validate=Length(equal=1))
    dropout = fields.Float(default=0.0, missing=None)
    residual = fields.Boolean(default=False, missing=False)
    residual_dense = fields.Boolean(default=False, missing=False)


class CNNConfiguration(Schema):
    filters = fields.Integer(default=32)
    kernel = fields.List(fields.Integer, default=[21, 11], validate=Length(equal=2))
    stride = fields.List(fields.Integer, default=[2, 1], validate=Length(equal=2))
    padding = fields.List(fields.Integer, allow_none=True, validate=Length(equal=2))
    batch_norm = fields.Boolean(default=False)
    activation = fields.String(default="hardtanh")
    activation_params = fields.List(fields.Field, default=[], missing=[])


class ContextConfiguration(Schema):
    context = fields.Integer(default=20)
    activation = fields.String(default="hardtanh")
    activation_params = fields.List(fields.Field, default=[], missing=[])


class EncoderConfiguration(Schema):
    activation = fields.String(default="hardtanh")
    activation_params = fields.List(fields.Field, default=[], missing=[])


class InputConfiguration(Schema):
    feat_type = fields.String(required=True, default="logspect", missing="logspect", load_from="type")
    normalize = fields.Boolean(default=True, missing=True)
    sample_rate = fields.Int(default=16000, missing=16000)
    window_size = fields.Float(default=0.02, missing=0.02)
    window_stride = fields.Float(default=0.01, missing=0.01)
    window = fields.String(default="hamming", missing="hamming")
    features = fields.Int(default=64, missing=None)
    n_fft = fields.Int(default=512, missing=None)
    int_values = fields.Boolean(default=False, missing=False)


class SpeechModelConfiguration(Schema):
    model = fields.String(required=True)

    input = fields.Nested(InputConfiguration, load_from="input")
    jasper = fields.Nested(JasperConfiguration, load_from="jasper", many=True)
    encoder = fields.Nested(EncoderConfiguration, load_from="encoder")
    cnn = fields.Nested(CNNConfiguration, load_from="cnn", many=True)
    rnn = fields.Nested(RNNConfiguration, load_from="rnn")
    ctx = fields.Nested(ContextConfiguration, load_from="context")
    labels = fields.Nested(LabelConfiguration)
