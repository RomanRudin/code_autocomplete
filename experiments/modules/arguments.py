class Arguments():
    def __init__(self, data_dir: str = f"{WORKDIR}/Clean_Dataset", ckpt_dir: str = f"{WORKDIR}/checkpoints/{LINE_MODEL_NAME}",
                    plot_dir: str = F"{WORKDIR}/plots/{LINE_MODEL_NAME}", tokenizer: str = "tokenizer.json", 
                    epochs: int = 5, batch: int = 32, lr: float = 5e-4,
                    ctx: int = 128, d_model: int = 256, n_layers: int = 4,
                    n_heads: int = 8, vocab_size: int = 000, max_files: int = 0,
                    val_split: float = 0.1, seed: int = 42, for_usage: bool = False,
                    skip_token: bool = False, skip_line: bool = False, test: bool = False):
        self.data_dir = data_dir
        self.ckpt_dir = ckpt_dir
        self.plot_dir = plot_dir
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.ctx = ctx
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_files = max_files
        self.val_split = val_split
        self.seed = seed
        self.skip_token = skip_token
        self.skip_line = skip_line
        self.test = test
        self.for_usage = for_usage