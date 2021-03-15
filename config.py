class Config:

    def __init__(
            self,
            model,
            dataset,
            device,
            num_cv='',
            sample_duration=32,
            stride=1,
            sample_size=(224, 224),
            ft_begin_idx=3,
            acc_baseline=0.92,
            train_batch=32,
            val_batch=32,
            learning_rate=1e-3,
            momentum=0.5,
            weight_decay=1e-3,
            factor=0.1,
            min_lr=1e-7,
            num_epoch=1000,
            output ='',
            num_prune = 5
    ):

        self.model = model
        self.train_crop = 'random'

        self.dataset = dataset

        self.num_cv = num_cv


        self.device = device


        self.sample_duration = sample_duration
        self.stride = stride
        self.sample_size = sample_size


        self.ft_begin_idx = ft_begin_idx


        self.acc_baseline = acc_baseline


        self.train_batch = train_batch
        self.val_batch = val_batch


        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay


        self.factor = factor
        self.min_lr = min_lr


        self.num_epoch = num_epoch
        self.output = output

        self.num_prune = num_prune