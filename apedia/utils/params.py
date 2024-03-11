preprocess_cell_type_data_params = {
    
}

train_cell_type_detection_params = {
    
}

train_tumor_patch_detector_params = {
    "column_name_data": 'pdl1',
    "column_name_mask": 'mask',
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
