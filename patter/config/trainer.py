from marshmallow import Schema, fields


class TrainerOutputConfiguration(Schema):
    model_path = fields.String(required=True)
    log_path = fields.String(required=True)
    checkpoint_interval = fields.Integer(default=0, missing=0)


class OptimizerSettings(Schema):
    optimizer = fields.String(required=True)
    lr = fields.Float(required=True)
    momentum = fields.Float(default=0.9)
    nesterov = fields.Boolean(default=True)
    weight_decay = fields.Float(default=0.0, missing=0.0)


class LarcSettings(Schema):
    enabled = fields.Boolean(default=False, missing=False)
    coeff = fields.Float(default=0.02, missing=0.02)
    clip = fields.Boolean(default=True, missing=True)
    eps = fields.Float(defaulut=1e-8, missing=1e-8)


class SchedulerSettings(Schema):
    lr_annealing = fields.Float(default=1.1, load_from="anneal")


class TrainingSettings(Schema):
    epochs = fields.Integer(required=True)
    batch_size = fields.Integer(required=True)
    num_workers = fields.Integer(required=True)
    max_norm = fields.Float(required=False, default=400.0, missing=None)
    sortagrad = fields.Boolean(missing=False, default=False)
    optimizer = fields.Nested(OptimizerSettings)
    scheduler = fields.Nested(SchedulerSettings)
    larc = fields.Nested(LarcSettings, missing=None)


class TrainerConfiguration(Schema):
    expt_id = fields.String(required=True, load_from="id")
    cuda = fields.Boolean(default=True)
    fp16 = fields.Boolean(default=False)

    output = fields.Nested(TrainerOutputConfiguration)
    trainer = fields.Nested(TrainingSettings)
