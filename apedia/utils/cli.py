import argparse
from apedia.utils.params import train_tumor_patch_detector_params
from apedia.train_tumor_patch_detector import train_tumor_patch_detector

def train_tumor_patch_detector_handler(args):
    # Convert argparse.Namespace to a dictionary
    args_dict = vars(args)
    # Call the training function with the args dictionary
    train_tumor_patch_detector(args_dict)

def setup_tumor_patch_detector_subparser(subparsers):
    # Subcommand for tumor patch detector training
    parser_tpd = subparsers.add_parser('train_tumor_patch_detector', help='Train Tumor Patch Detector')
    
    # Use the default values from train_tumor_patch_detector_params dictionary
    parser_tpd.add_argument('--data_df_path', type=str, required=True, help='Path to the dataframe containing the dataset information')
    parser_tpd.add_argument('--output_dir', type=str, required=True, help='Directory to save training outputs')
    parser_tpd.add_argument('--column_path_data', type=str, default=train_tumor_patch_detector_params['column_path_data'], help='Column name for data patches paths')
    parser_tpd.add_argument('--column_path_mask', type=str, default=train_tumor_patch_detector_params['column_path_mask'], help='Column name for mask patches paths')
    parser_tpd.add_argument('--column_scalar_label', type=str, default=train_tumor_patch_detector_params['column_scalar_label'], help='Column name for scalar labels. If "dummy", use dummy targets')
    parser_tpd.add_argument('--lr', type=float, default=train_tumor_patch_detector_params['lr'], help='Learning rate')
    parser_tpd.add_argument('--epochs', type=int, default=train_tumor_patch_detector_params['epochs'], help='Number of epochs')
    parser_tpd.add_argument('--bs', type=int, default=train_tumor_patch_detector_params['bs'], help='Batch size')
    parser_tpd.add_argument('--aug_mult', type=float, default=train_tumor_patch_detector_params['aug_mult'], help='Augmentation multiplier')
    parser_tpd.add_argument('--label_smoothing', type=float, default=train_tumor_patch_detector_params['label_smoothing'], help='Label smoothing')
    parser_tpd.add_argument('--encoder_name', type=str, default=train_tumor_patch_detector_params['encoder_name'], help='Encoder name for the model')
    parser_tpd.add_argument('--cv_split', type=int, default=train_tumor_patch_detector_params['cv_split'], help='Cross-validation split index')
    parser_tpd.add_argument('--num_workers', type=int, default=train_tumor_patch_detector_params['num_workers'], help='Number of workers for data loading')
    parser_tpd.add_argument('--do_cosine_annealing', type=lambda x: (str(x).lower() == 'true'), default=train_tumor_patch_detector_params['do_cosine_annealing'], help='Whether to use cosine annealing')
    parser_tpd.set_defaults(func=train_tumor_patch_detector_handler)

def main():
    parser = argparse.ArgumentParser(prog="apedia", description='APEDIA Project CLI')
    subparsers = parser.add_subparsers(help='Available commands')

    # Setup for tumor patch detector subcommand
    setup_tumor_patch_detector_subparser(subparsers)

    # TODO Setup for cell type detector subcommand
    # setup_cell_type_detector_subparser(subparsers)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
