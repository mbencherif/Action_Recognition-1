class Config:

    def __init__(
            self,
            device,
            sample_duration=16,
            stride=1,
            sample_size=(224, 224),
            ft_begin_idx=3,
            test_type = 0
    ):

        self.model = model
        self.device = devide
        self.sample_duration = sample_duration
        self.stride = stride
        self.sample_size = sample_size
        self.ft_begin_idx = ft_begin_idx
        self.test_type = test_type

