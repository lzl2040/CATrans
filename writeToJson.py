import json

if __name__ == '__main__':
    images_prefix = "JPEGImages/"
    segmentation_prefix = "SegmentationClass/"
    val_txt = "E:/deepLearningTest/CV/CATrans/datasets/PASCAL-5/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
    contents = []
    with open(val_txt,'r') as f:
        for line in f.readlines():
            line = line.split("\n")
            line = line[0]
            image = images_prefix + line + ".jpg"
            segmentation_class = segmentation_prefix + line + ".png"
            content = []
            content.append(image)
            content.append(segmentation_class)
            contents.append(content)
        f.close()
    with open('test2.json','w') as f:
        json.dump(contents, f)
        f.close()