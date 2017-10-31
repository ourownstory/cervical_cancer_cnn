# Utility functions for CS 231n project
# Based on PyTorch Transfer learning tutorial
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import pandas as pd

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def maybe_makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def split_data(root_dir = '../../data', SPLIT_DATA = False, prop_train = 0.7, prop_val = 0.2, prop_test = 0.1):

## Function to split data into directories
    if SPLIT_DATA:
        import os
        from shutil import copyfile

        # Define parameters of the split

        def get_indices(num_samples, prop_train = 0.7, prop_dev = 0.2, prop_test = 0.1):

            assert (prop_train + prop_dev + prop_test - 1.0 < 1e-5)

            num_train_samples = int(np.floor(num_samples * prop_train))
            num_dev_samples = int(np.floor(num_samples * prop_dev))
            num_test_samples = int(np.ceil(num_samples * prop_test))

            indices = list(range(num_samples))
            np.random.seed(123)
            np.random.shuffle(indices)

            train_indices = indices[0:num_train_samples]
            dev_indices = indices[num_train_samples:num_train_samples + num_dev_samples]
            test_indices = indices[num_train_samples + num_dev_samples:]
            return train_indices, dev_indices, test_indices

        def maybe_makedir(dirname):
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        # Define directories
        root_dir = '../../data/all_data'
        train_path = root_dir
        split_path = root_dir + '/split_new'

        # Set the random seed
        np.random.seed(456)

        # Create split path if it does not exist
        if not os.path.exists(split_path):
            os.makedirs(split_path)

        class_names = ('Type_1', 'Type_2', 'Type_3')
        for class_name in class_names:

            # Define the target location for the split class path
            split_train_class_path = split_path + '/' + 'train' + '/' + class_name
            split_val_class_path = split_path + '/' + 'val' + '/' + class_name
            split_test_class_path = split_path + '/' + 'test' + '/' + class_name

            # Make directories if they dont already exist
            maybe_makedir(split_train_class_path)
            maybe_makedir(split_val_class_path)
            maybe_makedir(split_test_class_path)

            # Directory where original training data are stored
            train_class_path = train_path + '/' + class_name

            # Get list of files in train directory
            train_files = os.listdir(train_class_path)
            train_files_clean = [k for k in train_files if 'additional' not in k]
            train_files_additional = [k for k in train_files if 'additional' in k]

            train_indices, val_indices, test_indices = get_indices(len(train_files_clean), prop_train, prop_val, prop_test)
            print((len(train_indices), len(val_indices), len(test_indices), len(train_files)))

            # Copy the train files to split directory
            for index in train_indices:
                src_path = train_class_path + '/' + train_files_clean[index]
                dest_path = split_train_class_path + '/' + train_files_clean[index]
                copyfile(src_path, dest_path)
            # Copy all additional files to train directory
            for additional_name in train_files_additional:
                src_path = train_class_path + '/' + additional_name
                dest_path = split_train_class_path + '/' + additional_name
                copyfile(src_path, dest_path)
            # Copy the val files to split directory
            for index in val_indices:
                src_path = train_class_path + '/' + train_files_clean[index]
                dest_path = split_val_class_path + '/' + train_files_clean[index]
                copyfile(src_path, dest_path)
            # Copy the test files to split directory
            for index in test_indices:
                src_path = train_class_path + '/' + train_files_clean[index]
                dest_path = split_test_class_path + '/' + train_files_clean[index]
                copyfile(src_path, dest_path)
            
            
            
class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)
            

def data_scaling(load_path = '/data/all_data/split_new/', save_path = '/data/all_data_scaled_alt/'):
## Function to run data scaling. Assumes that the data at load_path have already been split into classes and train/val/test

    
    home = os.path.expanduser('~')
    load_path = home + load_path
    # load_path = home + '/data/train/'
    save_path = home + save_path

    classes = ['Type_1/', 'Type_2/', 'Type_3/']
    splits = ['train/', 'val/', 'test/']
    # splits = ['test/']

    for split in splits:
        # Get data split
        if split == 'train/':
            transform_scale = Scale(256)
        else:
            transform_scale = Scale(224)

        for c in classes:    
            load_path_class = load_path + split + c
            save_path_class = save_path + split + c

            # Make directories
            if not os.path.exists(save_path + split):
                os.makedirs(save_path + split)
            if not os.path.exists(save_path_class):
                os.makedirs(save_path_class)

            directory = os.fsencode(load_path_class)

            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".jpg"): 
    #                 print(load_path_class + filename)
                    im = Image.open(load_path_class + filename)
                    img = transform_scale(im)
                    img.save(save_path_class + filename) 

                
def convert_performance_dict(performance_dict, hyperparam_dict):
    
    df_list = []
    for phase in performance_dict.keys():
        df = pd.DataFrame.from_dict(performance_dict[phase]).stack().reset_index()
        df.columns = ['Metric', 'Class','Performance']
        df = df.set_index(['Metric', 'Class'])['Performance'].apply(pd.Series).stack()
        df = df.reset_index()
        df.columns = ['Metric', 'Class','Epoch','Performance']
        
        df['phase'] = phase
        df_list.append(df)
    df_all = pd.concat(df_list)
    
    for k, v in hyperparam_dict.items():
        print (k, v)
        df_all[k] = v
    return df_all

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def exp_lr_scheduler(optimizer, epoch, init_lr = 0.001, lr_decay_epoch = 16, lr_decay_const = 0.1):
    
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (lr_decay_const ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer 

## train_model function. From tutorial
def train_model(model, criterion, optimizer, lr_scheduler, dset_loaders, dset_sizes, num_epochs=25, init_lr = 1e-3, lr_decay_epochs = 7, lr_decay_const = 0.1, use_gpu = True):
    # Main function for training a model. Based on Pytorch transfer learning tutorial
    since = time.time()

    best_model = model
    best_loss = 1e7
    
    classes = ['Type_1', 'Type_2', 'Type_3']
    
    # A dictionary to log performance
    performance_dict = {
        'train': {
            'Overall': {
                    'loss':[],
                    'accuracy':[]
                    },
            'Type_1': {
                    'specificity':[],
                    'precision':[],
                    'recall':[],
                    'f1':[]
                },
            'Type_2': {
                    'specificity':[],
                    'precision':[],
                    'recall':[],
                    'f1':[]              
                },
            'Type_3': {
                    'specificity':[],
                    'precision':[],
                    'recall':[],
                    'f1':[]           
                }

        },
        'val': {
            'Overall': {
                    'loss':[],
                    'accuracy':[]
                },
            'Type_1': {
                    'specificity':[],
                    'precision':[],
                    'recall':[],
                    'f1':[]
                },
            'Type_2': {
                    'specificity':[],
                    'precision':[],
                    'recall':[],
                    'f1':[]  
                },
            'Type_3': {
                    'specificity':[],
                    'precision':[],
                    'recall':[],
                    'f1':[]
                }
        }
    }
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if lr_scheduler is not None:
                    optimizer = lr_scheduler(optimizer, epoch, init_lr = init_lr, lr_decay_epoch = lr_decay_epochs, lr_decay_const = lr_decay_const)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Initialize a dictionary to maintain running performance values for the epcoh
            epoch_statistics_dict = {
                                    'Type_1': {
                                        'tp':0,
                                        'tn':0,
                                        'fp':0,
                                        'fn':0
                                    },'Type_2': {
                                        'tp':0,
                                        'tn':0,
                                        'fp':0,
                                        'fn':0
                                    },'Type_3': {
                                        'tp':0,
                                        'tn':0,
                                        'fp':0,
                                        'fn':0
                                    }
                                                        }
            
            # Iterate over data.
            batch_id = 0
            for data in dset_loaders[phase]:
                batch_id += 1
                # print(batch_id)
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda(async = True)), \
                        Variable(labels.cuda(async = True))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
#                 print(loss)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                
                # Log the batch statistics for each class
                for i in range(len(classes)):
                    the_class = classes[i]
                    
                    tp = torch.sum((preds == i) & (labels.data == i))
                    tn = torch.sum((preds != i) & (labels.data != i))
                    fp = torch.sum((preds == i) & (labels.data != i))
                    fn = torch.sum((preds != i) & (labels.data == i))
                    
                    epoch_statistics_dict[the_class]['tp'] += tp
                    epoch_statistics_dict[the_class]['tn'] += tn
                    epoch_statistics_dict[the_class]['fp'] += fp
                    epoch_statistics_dict[the_class]['fn'] += fn
                    
#             epoch_loss = running_loss / dset_sizes[phase]
            epoch_loss = running_loss / batch_id
            epoch_acc = running_corrects / dset_sizes[phase]
            
            performance_dict[phase]['Overall']['loss'].append(epoch_loss)
            performance_dict[phase]['Overall']['accuracy'].append(epoch_acc)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Log to the class level performance dict for each epoch
            for i in range(len(classes)):
                the_class = classes[i]
                
                # Compute metrics
                tp = epoch_statistics_dict[the_class]['tp']
                tn = epoch_statistics_dict[the_class]['tn']
                fp = epoch_statistics_dict[the_class]['fp']
                fn = epoch_statistics_dict[the_class]['fn']
                
                recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
                f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
                
                # Append results to performance_dict
                performance_dict[phase][the_class]['recall'].append(recall)
                performance_dict[phase][the_class]['precision'].append(precision)
                performance_dict[phase][the_class]['specificity'].append(specificity)
                performance_dict[phase][the_class]['f1'].append(f1)
                
                print('{} F1: {:.4f} Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}'.format(the_class, f1, precision, recall, specificity))
        
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print('new best model')
                best_loss = epoch_loss
                best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    return best_model, performance_dict


def plot_performance(performance_dict, figure_path_root = '../figures/', figure_name = 'default'):

    ## Performance Plots
    figure_path = figure_path_root

    ## Plot loss
    fig = plt.plot(performance_dict['train']['loss'], label = 'train')
    plt.plot(performance_dict['val']['loss'], label = 'val')
    ax = plt.axes()
    plt.xlabel('Epoch', fontsize = 14)
    plt.ylabel('Loss', fontsize = 14)
    plt.ylim(0, 0.022)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc = 'upper right')
    plt.savefig(figure_path + figure_name + 'loss' + '.png', dpi = 1200)
    plt.show()

    # Plot accuracy
    fig = plt.plot(performance_dict['train']['accuracy'], label = 'train')
    plt.plot(performance_dict['val']['accuracy'], label = 'val')
    ax = plt.axes()
    plt.xlabel('Epoch', fontsize = 14)
    plt.ylabel('Accuracy', fontsize = 14)
    plt.ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc = 'upper right')
    plt.savefig(figure_path + figure_name + 'acc' + '.png', dpi = 1200)
    plt.show()
    

def predict_on_test(model, loader, dtype, dset_size = (512, 3)):
    """
    Check the accuracy of the model.
    """
    # Set the model to eval mode
    model.eval()
    num_correct, num_samples = 0, 0
    softmax = nn.Softmax()
    i = 0
    for x in loader:
        inputs, _ = x
        # Cast the image data to the correct type and wrap it in a Variable. At
        # test-time when we do not need to compute gradients, marking the Variable
        # as volatile can reduce memory usage and slightly improve speed.
        x_var = Variable(inputs.type(dtype), volatile=True)

        # Run the model forward, and compare the argmax score with the ground-truth
        # category.
        scores = model(x_var)
        
        ## which Softmax is correct?
        # Used so far
        probs = softmax(scores)

        if (i == 0):
            probs_out = probs.data.cpu().numpy()
        else:
            probs_out = np.concatenate((probs_out, probs.data.cpu().numpy()), axis = 0)
        i += 1
#     print(labels_out)
    return probs_out  

def predict_on_test2(model, loader, dtype, dset_size = (512, 3)):
    """
    Check the accuracy of the model.
    """
    # Set the model to eval mode
    model.eval()
    num_correct, num_samples = 0, 0
    softmax = nn.Softmax()
    i = 0
    for x in loader:
        inputs, labels = x
        # Cast the image data to the correct type and wrap it in a Variable. At
        # test-time when we do not need to compute gradients, marking the Variable
        # as volatile can reduce memory usage and slightly improve speed.
        x_var = Variable(inputs.type(dtype), volatile=True)

        # Run the model forward, and compare the argmax score with the ground-truth
        # category.
        scores = model(x_var)
        probs = softmax(scores)
        if (i == 0):
            probs_out = probs.data.cpu().numpy()
            labels_out = labels.cpu().numpy()
        else:
            probs_out = np.concatenate((probs_out, probs.data.cpu().numpy()), axis = 0)
            labels_out = np.concatenate((labels_out, labels.cpu().numpy()), axis = 0)
        i += 1
#     print(labels_out)
    return probs_out, labels_out

## stats from scores function

def stats_from_probs(probs, labels, description, phase = 'test'):
    since = time.time()
    
    classes = ['Type_1', 'Type_2', 'Type_3']
    
    # A dictionary to log performance
    performance_dict = {
        'test': {
            'Overall': {
                    'loss':[],
                    'accuracy':[]
                    },
            'Type_1': {
                    'specificity':[],
                    'precision':[],
                    'recall':[],
                    'f1':[]
                },
            'Type_2': {
                    'specificity':[],
                    'precision':[],
                    'recall':[],
                    'f1':[]              
                },
            'Type_3': {
                    'specificity':[],
                    'precision':[],
                    'recall':[],
                    'f1':[]           
                }

        },
    }
    
    loss = 0.0
    corrects = 0
    num_img = len(labels)

    # Initialize a dictionary to maintain running performance values for the epcoh
    statistics_dict = {
                            'Type_1': {
                                'tp':0,
                                'tn':0,
                                'fp':0,
                                'fn':0
                            },'Type_2': {
                                'tp':0,
                                'tn':0,
                                'fp':0,
                                'fn':0
                            },'Type_3': {
                                'tp':0,
                                'tn':0,
                                'fp':0,
                                'fn':0
                            }
    }

    # Iterate over data.

    log_probs  = torch.from_numpy(np.log(probs))
    labels = torch.from_numpy(labels)

    # wrap them in Variable
    log_probs, labels = Variable(log_probs), Variable(labels)

    ## get vote
    _, preds = torch.max(log_probs.data, 1)
    
    #### CHANGE
    loss = nn.functional.nll_loss(log_probs, labels, weight=None, size_average=True)
#     loss = nn.functional.cross_entropy(scores, labels)
#     criterion = nn.CrossEntropyLoss()
#     loss = criterion(scores, labels)
#     print(loss)

    # statistics
    loss = loss.data[0]
    corrects = torch.sum(preds == labels.data)

    # Log the batch statistics for each class
    for i in range(len(classes)):
        the_class = classes[i]

        tp = torch.sum((preds == i) & (labels.data == i))
        tn = torch.sum((preds != i) & (labels.data != i))
        fp = torch.sum((preds == i) & (labels.data != i))
        fn = torch.sum((preds != i) & (labels.data == i))

        statistics_dict[the_class]['tp'] += tp
        statistics_dict[the_class]['tn'] += tn
        statistics_dict[the_class]['fp'] += fp
        statistics_dict[the_class]['fn'] += fn

    acc = corrects / num_img

    performance_dict[phase]['Overall']['loss'].append(loss)
    performance_dict[phase]['Overall']['accuracy'].append(acc)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))

    # Log to the class level performance dict for each epoch
    for i in range(len(classes)):
        the_class = classes[i]

        # Compute metrics
        tp = statistics_dict[the_class]['tp']
        tn = statistics_dict[the_class]['tn']
        fp = statistics_dict[the_class]['fp']
        fn = statistics_dict[the_class]['fn']

        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

        # Append results to performance_dict
        performance_dict[phase][the_class]['recall'].append(recall)
        performance_dict[phase][the_class]['precision'].append(precision)
        performance_dict[phase][the_class]['specificity'].append(specificity)
        performance_dict[phase][the_class]['f1'].append(f1)

        print('{} F1: {:.4f} Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}'.format(the_class, f1, precision, recall, specificity))
    
#     return performance_dict

    ## convert to DF
    df_list = []
    for phase in performance_dict.keys():
        df = pd.DataFrame.from_dict(performance_dict[phase]).stack().reset_index()
        df.columns = ['Metric', 'Class','Performance']
#         df = df.set_index(['Metric', 'Class'])['Performance'].apply(pd.Series).stack()
#         df = df.reset_index()
#         df.columns = ['Metric', 'Class','Epoch','Performance']
        
        df['phase'] = phase
        df_list.append(df)
    df_all = pd.concat(df_list)
    
    df_all['description'] = description
    
    return df_all
        


## With edge indication
class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, edge = "short", interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.edge = edge
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if self.edge == "short":               
                if (w <= h and w == self.size) or (h <= w and h == self.size):
                    return img
                if w < h:
                    ow = self.size
                    oh = int(self.size * h / w)
                    return img.resize((ow, oh), self.interpolation)
                else:
                    oh = self.size
                    ow = int(self.size * w / h)
                    return img.resize((ow, oh), self.interpolation)
            elif self.edge == "long":               
                if (w >= h and w == self.size) or (h >= w and h == self.size):
                    return img
                if w > h:
                    ow = self.size
                    oh = int(self.size * h / w)
                    return img.resize((ow, oh), self.interpolation)
                else:
                    oh = self.size
                    ow = int(self.size * w / h)
                    return img.resize((ow, oh), self.interpolation)
                
        else:
            return img.resize(self.size, self.interpolation)