preprocess_cell_type_data_replacement_params = {
    'TZ neg.': 'tz_neg',
    'TZ neg': 'tz_neg',
    'TZ pos.': 'tz_pos',
    'TZ pos': 'tz_pos',
    'TZ Pos': 'tz_pos',
    'Neutrophiler Granulozyt': 'other',
    'Keine TZ': 'other',
    'Kein TZ': 'other',
    'Tumorzelle': 'exclude',
    'Eisenpigment': 'exclude',
    #19sep23 - updated annotations
    'KeineTZ': 'other',
    # unknown what's right
    'Keine TZ pos': 'other',
}

preprocess_cell_type_data_params = {
    "no_roi_infos": False,
    "tip_the_balance": 0,
    "replacements": preprocess_cell_type_data_replacement_params,
    "path_deepliif_hema_gan_weights": None,
    "viz": False,
}

train_cell_type_detection_params = {
    "cv_split": 0,
    "mask_col": 'path_seg_one_match',
    "patch_col": 'path_patch_png',
    "instance_seg_col": 'path_exact_one_match',
    "bs": 8,
    "epochs": 25,
    "lr": 0.001,
    "aug_mult": 1.0,
    "encoder_name": 'timm-efficientnet-b5',
    "num_input_channels": 3,
    "num_classes": 4,
    "num_workers": 4,
    "device": 'cuda',
    "do_elastic": False,
    "do_not_analyze": False,
    "label_smoothing": 0.0,
    "weight_decay": 0.01,
    "disable_tqdm": False,
    "class_weights": [0.0001, 1, 1, 1]
}



train_tumor_patch_detector_params = {
    "column_path_data": 'pdl1',
    "column_path_mask": 'mask',
    "column_scalar_label": 'dummy',
    "lr": 0.001,
    "epochs": 50,
    "bs": 8,
    "aug_mult": 1,
    "label_smoothing": 0.0,
    "encoder_name": "timm-efficientnet-b5",
    "cv_split": 0,
    "num_workers": 4,
    "do_cosine_annealing": False,
}
