from datasets import DatasetDict, load_from_disk, load_dataset, concatenate_datasets


def prepare_dataset(data_type, train_dataset_name, seed):
    """dataset load 및 return하는 function

    Args:
        data_type: args에서 받아오는 값
            original: 기존 데이터셋
            korquad: korquad v1 데이터셋
                train: korquad_train
                valid: korquad_valid
            korquad_hard: korquad의 valid까지 train으로 사용
                train: korquad_train + korquad_valid
                valid: original_valid
            mix: original과 korquad를 섞어서 사용
                train: original_train + korquad_train
                valid: original_valid + korquad_valid
            mix_hard: mix 및 korquad의 valid까지 train으로 사용
                train: original_train + korquad_train + korquad_valid
                valid: original_valid
        train_dataset_name: original dataset 경로
        seed: seed

    Returns:
        DatasetDict: dataset
    """
    if data_type == 'original':
        return load_from_disk(train_dataset_name)
    
    elif data_type == 'korquad':
        return load_dataset("squad_kor_v1")
    
    elif data_type == 'korquad_hard':
        # korquad_valid까지 train으로 사용
        original_datasets = load_from_disk(train_dataset_name)
        original_datasets['train'] = original_datasets['train'].remove_columns(['document_id', '__index_level_0__'])
        validation_dataset = original_datasets['validation'].remove_columns(['document_id', '__index_level_0__'])
        
        korquad_datasets = load_dataset("squad_kor_v1", features=original_datasets["train"].features)
        train_dataset = concatenate_datasets([korquad_datasets['train'], korquad_datasets['validation']])
        
        return DatasetDict({'train': train_dataset, 'validation': validation_dataset})
        
    elif 'mix' in data_type:
        # korquad와 병합
        original_datasets = load_from_disk(train_dataset_name)
        original_datasets['train'] = original_datasets['train'].remove_columns(['document_id', '__index_level_0__'])
        original_datasets['validation'] = original_datasets['validation'].remove_columns(['document_id', '__index_level_0__'])
        
        korquad_datasets = load_dataset("squad_kor_v1", features=original_datasets["train"].features)
        
        # train == original_train + korquad_train
        if data_type == 'mix':
            train_dataset = concatenate_datasets([original_datasets['train'], korquad_datasets['train']])
            validation_dataset = concatenate_datasets([original_datasets['validation'], korquad_datasets['validation']])
        
        # train == original_train + korquad_train + korquad_valid
        elif data_type == 'mix_hard':
            korquad_datasets = concatenate_datasets([korquad_datasets['train'], korquad_datasets['validation']])
            train_dataset = concatenate_datasets([original_datasets['train'], korquad_datasets])
            validation_dataset = original_datasets['validation']
        
        else:
            raise ValueError("Invalid data_type")

        train_dataset = train_dataset.shuffle(seed=seed)
        return DatasetDict({'train': train_dataset, 'validation': validation_dataset})
    
    else:
        raise ValueError("Invalid data_type")