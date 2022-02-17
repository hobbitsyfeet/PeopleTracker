import tensorflow as tf
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import pixellib

from pixellib.custom_train import instance_custom_training

def train(classes, dataset_path, pretrained_model="mask_rcnn_coco.h5", output_path="./Test", batch_size=1, num_epochs=300, network_type="resnet101"):
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    classes = ['BG'] + classes
    num_classes = len(classes) -1
    print(num_classes)

    

    train_maskrcnn = instance_custom_training()
    train_maskrcnn.modelConfig(network_backbone = network_type, num_classes= num_classes, batch_size = batch_size)
    train_maskrcnn.config.class_names=classes
    train_maskrcnn.load_pretrained_model(pretrained_model)
    train_maskrcnn.load_dataset(dataset_path)

    train_maskrcnn.config.NUM_CLASSES = num_classes
    # train_maskrcnn.config.IMAGE_META_SIZE=14

    train_maskrcnn.train_model(num_epochs = num_epochs, augmentation=True, path_trained_models = output_path)
    print("Done training")
