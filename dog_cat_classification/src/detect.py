from data_preprocess import load_test_data
import torch
from bulid_network import cnn
import time
import numpy

def test():
    test_loader = load_test_data()#请学长再这个函数里修改需要预测的数据的地址
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.load_state_dict(torch.load("F:/PythonSave/dog_cat_classification/weights/weight_dog_cat.pt"), False)
    model.eval()
    total = 0
    current = 0
    out_test = []
    #print(test_loader)
    for data in test_loader:
        start = time.time()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()
        end = time.time()
        #if predicted == labels:
            #result = 1
        time_do = end-start
        out_test.append([labels,predicted,time_do])#输出三列，第一列为已知答案，第二列为预测答案，第三列为时间
    numpy.savetxt('F:/PythonSave/dog_cat_classification/output/Result.csv', out_test, delimiter=',')
    print('Accuracy:%d%%' % (100 * current / total))
if __name__ == '__main__':
    test()