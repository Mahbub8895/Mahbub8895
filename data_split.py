path = '/home/nibaran/Downloads/DMR/final Project/FINAL/'

#cl = 'healthy'
cl = 'sick'

h_path = path + cl + '/'


import os
import shutil
import sklearn.model_selection as sk


f = os.listdir(h_path)




train, t = sk.train_test_split(f, test_size = 0.5, shuffle = True)
test, val = sk.train_test_split(t, test_size = 0.5, shuffle = True)




for file in train:
    shutil.copyfile(path + cl + '/' + file, path + 'data/' + 'train/' + cl + '/' + file)
    g = file.split('.')
    shutil.copyfile(path + cl + '_canny/' + g[0] + '.png', path + 'data/' + 'train/' + cl + '_canny/' +g[0] + '.png')
    
    
for file in test:
    shutil.copyfile(path + cl + '/' + file, path + 'data/' + 'test/' + cl + '/' + file)
    g = file.split('.')
    shutil.copyfile(path + cl + '_canny/' + g[0] + '.png', path + 'data/' + 'test/' + cl + '_canny/' +g[0] + '.png')    
    
for file in val:
    shutil.copyfile(path + cl + '/' + file, path + 'data/' + 'val/' + cl + '/' + file)
    g = file.split('.')
    shutil.copyfile(path + cl + '_canny/' + g[0] + '.png', path + 'data/' + 'val/' + cl + '_canny/' +g[0] + '.png')    