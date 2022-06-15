from data_preprocess import load_test_data
import torch
from bulid_network import cnn
import time
import numpy
import argparse

def detect(opt):
    test_loader = load_test_data(opt.img_path)#请学长再这个函数里修改需要预测的数据的地址
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.load_state_dict(torch.load(opt.weight), False)
    model.eval()
    total = 0
    current = 0
    out_test = []
    total_time = 0
    #print(test_loader)
    for data in test_loader:
        start = time.time()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum().item()
        end = time.time()
        #if predicted == labels:
            #result = 1
        time_do = end-start
        total_time += time_do
        out_test.append([labels,predicted,time_do])#输出三列，第一列为已知答案，第二列为预测答案，第三列为时间
    numpy.savetxt(opt.out_path, out_test, delimiter=',')
    print('Accuracy: %d %%, Time: %d ms' % (100 * current / total, 1000 * total_time / total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img-path', type=str, default= '../input/img_paths_test.txt', help='input image path file for detect')
    parser.add_argument('-w', '--weight', type=str, default= '../weights/weight_dog_cat.pt', help='weight file')
    parser.add_argument('-o', '--out-path', type=str, default= '../output/out_results.csv', help='output results')
    opt = parser.parse_args()
    detect(opt)