import tensorflow as tf

def train_val_split(data_dir:str, split:float, image_size:tuple, batch_size:int, seed:int):
    return tf.keras.utils.image_dataset_from_directory(data_dir,
                                                        validation_split=split,
                                                        subset='both',
                                                        seed=seed,
                                                        image_size=image_size,
                                                        batch_size = batch_size)
train_val_split('images/train',0.8, image_size=(256,256), batch_size=32, seed=42)