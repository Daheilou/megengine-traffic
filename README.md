方案根据官方提供的baseline改进

数据路径处理，tool/dataset.py修改，训练时需要修改第43行增加ann_file = ann_file.replace('traffic5','')，第106行增加 path = path.replace('traffic5','')
如果需要预测b榜数据,上述tool/dataset.py文件取消训练时增加的43行及106行，并去掉第42行及第105行的注释,需修改test_final.py第96行测试集数量584改为10066
模型为resnext101,需要双卡P100
修改faster_rcnn_resx101_800size_trafficdet_demo参数，主要为增大尺寸
