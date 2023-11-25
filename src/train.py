import tensorflow as tf
import os
import glob # Find folders/files
import shutil # Copy files
import uuid
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
import random


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
    train_maskrcnn.modelConfig(network_backbone = network_type, num_classes= num_classes, batch_size = batch_size, gpu_count=len(gpus))
    train_maskrcnn.config.class_names=classes
    train_maskrcnn.load_pretrained_model(pretrained_model)
    train_maskrcnn.load_dataset(dataset_path)

    train_maskrcnn.config.NUM_CLASSES = num_classes
    # train_maskrcnn.config.IMAGE_META_SIZE=14

    train_maskrcnn.train_model(num_epochs = num_epochs, augmentation=True, path_trained_models = output_path)
    print("Done training")

def leave_one_out_split(folder_list, index):
    testing = folder_list[index]
    training = folder_list[:index] + folder_list[index+1:]

    return training, testing

def random_sample_folder(folder, sample_number, clear_empty=True):
    # Each video has a folder holding the data
    internal_folders = glob.glob((folder+"*/"), recursive = True)

    


    # Collect all jpg that correspond to the sampled json files
    sample_jpg_list = []
    sample_json_list = []
    for folder in internal_folders:
        files = glob.glob((folder+"*.json"), recursive = True)
        if not files or files is None:
            continue
        
        

        # search the file list and remove any file that has an empty shape from the random sample list
        if clear_empty:
            for file in files:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if len(data['shapes']) == 0:
                        print("Removing ", file)
                        files.remove(file)
                        

        print("Number of labels:", len(files), " in folder: ", folder)
        if sample_number <= len(files):
            sample_json = random.sample(files, sample_number)
        else:
            print("Number of labels less than sample size, including all labels and not sampling.")
            sample_json = files

        sample_json_list.extend(sample_json)

        for file in sample_json:
            jpg = file[:-4] + "jpg"
            sample_jpg_list.append(jpg)
    
    return sample_json_list, sample_jpg_list

def copy_files_to_folder(file_list, out_folder, folder, reclassify):

    
    if not os.path.exists(out_folder):
        os.mkdir(folder)

    first_folder = out_folder + folder.split('/')[0] + "/"
    if not os.path.exists((first_folder)):
        os.mkdir(first_folder)

    second_folder = first_folder + folder.split('/')[1] + "/"
    if not os.path.exists((second_folder)):
        os.mkdir(second_folder)
   

    for json_file in file_list:

        file_id = uuid.uuid4()
        base_file = os.path.basename(json_file)[:-5] + "_" + str(file_id)
        jpg_file = json_file[:-4] + "jpg"

        

        shutil.copyfile(json_file, (second_folder + base_file + '.json'))
        shutil.copyfile(jpg_file, (second_folder + base_file + '.jpg'))

        # Reclassifies all labels to one set label
        if json_file[-4:] == "json":
            reclassify_labels(reclassify, (second_folder + base_file + '.json'))


def cross_validation_split(folder_list, out_folder, reclassify, skip_generated=True, sample_size=500):
    
    testing_files = []
    training_files = []

    evaluation_folders = []

    for i, folder in enumerate(folder_list): 


        training, testing = leave_one_out_split(folder_list, i)

        print("\n--------------\n", "Training:", training, " \nTesting:", testing)
        # print(testing)

        # Testing only has one folder
        testing_json_sample, testing_jpg_sample = random_sample_folder(testing, sample_size)



        
        testing_files.extend(testing_json_sample)
        testing_files.extend(testing_jpg_sample)
        
        # for file in testing_json_sample:
        #     reclassify_labels(label="Monkey", json_file=file)

        # print("Training Sample:", len())
        # # Testing has multiple folders

        train_json_sample = []
        train_jpg_sample = []

        for folder in training:

            json_sample, jpg_sample = random_sample_folder(folder, sample_size)
            
            train_json_sample.extend(json_sample)
            train_jpg_sample.extend(jpg_sample)

            # for file in train_json_sample:
            #     reclassify_labels(label="Monkey", json_file=file)
            
        print("Training Size:", len(train_json_sample))
        print("Testing Size:", len(testing_json_sample))

        out_training_folder_name = str(i) + "/train/"
        out_testing_folder_name = str(i) + "/test/"

        copy_files_to_folder(train_json_sample, out_folder, out_training_folder_name, reclassify=reclassify)
        # copy_files_to_folder(train_jpg_sample, out_folder, out_training_folder_name, reclassify=reclassify)
        copy_files_to_folder(testing_json_sample, out_folder, out_testing_folder_name, reclassify=reclassify)
        # copy_files_to_folder(testing_jpg_sample, out_folder, out_testing_folder_name, reclassify=reclassify)
        evaluation_folders.append((out_folder + str(i) + "/"))

    return evaluation_folders

def reclassify_labels(label, json_file, update_imagepath=True):
    '''
    Reclassifies all labels into designated label passed into as parameter
    '''

    # read information and update data
    with open(json_file, 'r') as f:
        

        data = json.load(f)
        for shape in data['shapes']:
            shape['label'] = label

        if update_imagepath:
            data['imagePath'] = (json_file[:-4] + "jpg")

    # Delete original file
    os.remove(json_file)

    # Write new, modified file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=3)


    
def prepare_monkey():
    base = "K:/Github/PeopleTracker/Evaluation/Monkeys/"
    folder_list = [
    "K:\Github\PeopleTracker\Evaluation\Monkeys/Displays Aggression/", 
    "K:\Github\PeopleTracker\Evaluation\Monkeys/Grooming/", 
    "K:\Github\PeopleTracker\Evaluation\Monkeys/Play/"
    ]
    
    # base = "K:/Github/PeopleTracker/Evaluation/Monkeys/"
    parent_folder = base + "/TestTrain/"

    evaluation_folders = cross_validation_split(folder_list, out_folder=parent_folder, reclassify="Monkey", sample_size=50)


def prepare_people():
    base = "K:/Github/PeopleTracker/Evaluation/People/"
    folder_list = [
        (base + "John Scott/"),
        (base + "Camera Obscura/"),
        (base + "Chris Cran/"),
        (base + "Contemporary/"),
        (base + "Historical/")
    ]
    parent_folder = base + "/TestTrain/"
    evaluation_folders = cross_validation_split(folder_list, out_folder=parent_folder, reclassify="People", sample_size=100)



def record_validation(output, value):
    pass
if __name__ == "__main__":

    random.seed(42)

    
    # base = "K:/Github/PeopleTracker/Evaluation/Monkeys/"
    # folder_list = [
    # "K:\Github\PeopleTracker\Evaluation\Monkeys/Displays Aggression/", 
    # "K:\Github\PeopleTracker\Evaluation\Monkeys/Grooming/", 
    # "K:\Github\PeopleTracker\Evaluation\Monkeys/Play/"
    # ]
    # base = "K:/Github/PeopleTracker/Evaluation/People/"
    # folder_list = [
    #     (base + "John Scott/"),
    #     (base + "Camera Obscura/"),
    #     (base + "Chris Cran/"),
    #     (base + "Contemporary/"),
    #     (base + "Historical/")
    # ]
    
    # base = "K:/Github/PeopleTracker/Evaluation/Monkeys/"
    # parent_folder = base + "/TestTrain/"
    # folder_list = [
    #     (base + "Displays Aggression/"),
    #     (base + "Grooming/"),
    #     (base + "Play/"),
    # ]

    # evaluation_folders = cross_validation_split(folder_list, out_folder=parent_folder, reclassify="People", sample_size=100)
    # evaluation_folders = cross_validation_split(folder_list, out_folder=parent_folder, reclassify="Monkey", sample_size=50)
    
    # Continue
    # cont = "K:/Github/PeopleTracker/Evaluation/People/TestTrain/0/mask_rcnn_model.049-0.530556.h5"
    # cont = "K:/Github/PeopleTracker/moved/mask_rcnn_coco.h5"
    # for train_test in evaluation_folders:

    test_train_folder = "K:/Github/PeopleTracker/Evaluation/Monkeys/TestTrain/2/"
    # print(test_train_folder[:-1])
    train(classes=["Monkey"], pretrained_model="K:/Github/PeopleTracker/mask_rcnn_coco.h5", dataset_path=test_train_folder[:-1], output_path=test_train_folder, batch_size=1, num_epochs=100)



        
