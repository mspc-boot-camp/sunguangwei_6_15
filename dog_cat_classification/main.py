#无作用
from data_preprocess import load_test_data
import torch
from bulid_network import cnn
import numpy

def test():
    test_loader = load_test_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.load_state_dict(torch.load("F:/PythonSave/dog_cat_classification/weights/weight_dog_cat.pt"), False)
    model.eval()
    total = 0
    current = 0
    result_cat_or_dog = []
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()
        result_cat_or_dog.append(labels,"\n")
    numpy.savetxt('F:\PythonData/res.csv', result_cat_or_dog, delimiter=',')
    print('Accuracy:%d%%' % (100 * current / total))
test()