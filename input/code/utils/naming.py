def wandb_naming(wandb_name, model_name, batch_size, max_epoch, learning_rate, warmup_steps, weight_decay):
    params = f'{model_name} | batch_size : {batch_size} | max_epoch : {max_epoch} | lr : {learning_rate} | warmup_steps: {warmup_steps} | weight_decay: {weight_decay}'
    if wandb_name != 'test':
        return f'{wandb_name} | {params}'
    else:
        return params