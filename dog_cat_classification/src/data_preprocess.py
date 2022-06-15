
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
from PIL import Image
import os,shutil
import random
#发现标签，确定使猫or狗
def find_label(str):
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        if str[i ] == '-':
            last = i
        #此处有修改，从.改为-
        if (str[i] == 'c' or str[i] == 'd') and str[i - 1] == '/':
            first = i
            break

    name = str[first:last]
    if name == 'dog':
        return 1
    else:
        return 0
#分离数据，将数据按照8：2分到train和test文件夹
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("src not exist!")
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
def Myloader(path):
    return Image.open(path).convert('RGB')
#数据处理
class MyDataset(Dataset):
    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder
    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
def load_test_data():#学长预测时的函数
    print('data processing...')
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
    ])
    data = []
    f = open(r"F:\PythonSave/dog_cat_classification/input/img_paths_test.txt", 'r')
    ff = f.readlines()
    for line in ff:
        line = line.rstrip("\n")
        name = find_label(line)
        data.append([line,name])
    test = MyDataset(data, transform=transform, loder=Myloader)
    test_data = DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=0)
    return test_data
def load_data():#我训练自己的模型是用到的数据
    print('data processing...')
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
    ])
    num_test = []

    test_rate = 0.2  # 训练集和测试集的比例为8:2。
    img_num = 700
    test_num = int(img_num * test_rate)

    data_test_cat = []
    data_test_dog = []
    data_train_cat = []
    data_train_dog = []
    test_index = random.sample(range(0, img_num), test_num)
    file_path = r"F:/PythonSave/fenpei"
    tr = "train"
    te = "test"
    cat = "Cat"
    dog = "Dog"
    for i in range(len(test_index)):
        num_test.append(test_index[i])
        # 移动猫
        srcfile = os.path.join(file_path, tr, cat, "cat." + str(test_index[i]) + ".jpg")
        dstfile = os.path.join(file_path, te, cat, "cat." + str(test_index[i]) + ".jpg")
        mymovefile(srcfile, dstfile)
        path_test_cat = file_path + "/" + te + "/" + cat + "/" + "cat." + str(test_index[i]) + ".jpg"
        name = find_label(path_test_cat)
        data_test_cat.append([path_test_cat, name])
        # data1 = init_process(file_path,tr,cat,"cat."+str(test_index[i])+".jpg", i)
        # 移动狗
        srcfile = os.path.join(file_path, tr, dog, "dog." + str(test_index[i]) + ".jpg")
        dstfile = os.path.join(file_path, te, dog, "dog." + str(test_index[i]) + ".jpg")
        mymovefile(srcfile, dstfile)
        path_test_dog = file_path + "/" + te + "/" + dog + "/" + "dog." + str(test_index[i]) + ".jpg"
        name = find_label(path_test_dog)
        data_test_dog.append([path_test_dog, name])
        # srcfile=os.path.join(file_path,tr,dog,str(test_index[i])+".jpg")
        # dstfile=os.path.join(file_path,te,dog,str(test_index[i])+".jpg")
        # mymovefile(srcfile,dstfile)
    for j in range(700):
        if j in num_test:
            a = 1
        else:
            path_train_dog = file_path + "/" + tr + "/" + dog + "/" + "dog." + str(j) + ".jpg"
            name = find_label(path_train_dog)
            data_train_dog.append([path_train_dog, name])
            path_train_cat = file_path + "/" + tr + "/" + cat + "/" + "cat." + str(j) + ".jpg"
            name = find_label(path_train_cat)
            data_train_cat.append([path_train_cat, name])

    # train
    train_data = data_train_cat + data_train_dog

    train = MyDataset(train_data, transform=transform, loder=Myloader)
    # test
    test_data = data_test_dog +  data_test_cat
    test = MyDataset(test_data, transform=transform, loder=Myloader)

    train_data = DataLoader(dataset=train, batch_size=10, shuffle=True, num_workers=0)
    test_data = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0)

    return train_data, test_data

#data_train_dog,data_train_cat,data_test_dog,data_test_cat = changetheplace_1()


