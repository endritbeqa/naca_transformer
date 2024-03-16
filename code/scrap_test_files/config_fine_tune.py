import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.dataset = '/local/disk1/ebeqa/naca_transformer/Bluff_data/bluff_body'
    config.output_dir = '/local/disk1/ebeqa/naca_transformer/Outputs/parallel_test'
    config.checkpoint_dir = '/local/disk1/ebeqa/naca_transformer/Outputs/parallel_test/nacaVIT'
    config.trainer = 'fine_tune'  # or 'preprocess' or 'fine_tune' or fine_tune_parallel
    config.layers_to_train = ('Layer2', 'ConvTranspose_0' )
    config.num_epochs = 1
    config.batch_size = 20
    config.shuffle_buffer_size = 1024
    config.learning_rate_scheduler = "sgdr"
    config.learning_rate_end_value = 1e-5
    config.sgdr_restarts = 1 #int(config.num_epochs / 50)
    config.warmup_fraction = 0.1
    config.weight_decay = 0.1
    config.output_frequency = 5

    config.vit = ml_collections.ConfigDict()
    config.vit.img_size = (200, 200)
    config.vit.patch_size = (10, 10)  # num_patches = (img_size / patch_size)
    config.vit.hidden_size = 300  # patch_size^2 * num_channels
    config.vit.num_layers = 3
    config.vit.num_heads = 3
    config.vit.dim_mlp = 4 * config.vit.hidden_size
    config.vit.dropout_rate = 0.0
    config.vit.att_dropout_rate = 0.0

    config.preprocess = ml_collections.ConfigDict()
    config.preprocess.readdir = ''
    config.preprocess.writedir = ''
    config.preprocess.incompressible = True
    config.preprocess.train_split = 0.9
    config.preprocess.nsamples = 1024
    config.preprocess.dim = (-0.75, 1.25, -1, 1)
    config.preprocess.num_neighbors = 5
    config.preprocess.gpu_id = 0

    config.pressure_preprocessing = ml_collections.ConfigDict()
    config.pressure_preprocessing.enable = True
    config.pressure_preprocessing.type = 'standardize_all' # standardize , range or coefficient
    config.pressure_preprocessing.new_range = (0, 1)
    
    config.internal_geometry = ml_collections.ConfigDict()
    config.internal_geometry.set_internal_value = False
    config.internal_geometry.value = 0

    return config