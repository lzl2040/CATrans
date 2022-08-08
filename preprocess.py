import os

def generate_train_and_test():
    files = os.listdir("./datasets/PASCAL-5/VOCdevkit/VOC2012/SegmentationObject")
    # 选取80%作为训练集,20%作为测试集
    train_num = 2330
    train_list = []
    test_list = []
    train_list.append("2007_000738")
    train_list.append("2007_001568")
    # with open("train.txt",'w') as f:
    #     for i in range(train_num):
    #         file_name = files[i]
    #         name = file_name.split(".")
    #         train_list.append(name[0])
    #         f.write(file_name + "\n")
    #     f.close()
    # test_num = 583
    # test_list = []
    # with open("test.txt",'w') as f:
    #     for i in range(train_num,len(files)):
    #         name = file_name.split(".")
    #         test_list.append(name[0])
    #         f.write(file_name + "\n")
    #     f.close()
    return train_list,test_list
