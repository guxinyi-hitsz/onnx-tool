import onnx_tool
import numpy as np

def profile_yolov3():
    modelpath = 'models/yolov3/yolov3.onnx'
    modelname = 'yolov3'
    m = onnx_tool.Model(modelpath)
    m.graph.shape_infer({'images': np.zeros((1, 3, 640, 640))})  # perform a valid shape_infer() before profile()
    m.graph.profile()  # perform a valid profile() before print_node_map()
    m.graph.print_node_map(f=f'models/yolov3/nodemap_{modelname}.csv', metric='FLOPs',
                           exclude_ops=['Constant'])  # save to csv file
    m.graph.print_op_histogram(f=f'models/yolov3/nodehist_{modelname}.csv')

def profile_swin():
    modelpath = 'models/swin/swin_transformer_image_classification_b1.onnx'
    modelname = 'swin-cls-b1'
    m = onnx_tool.Model(modelpath)
    m.graph.shape_infer({'images': np.zeros((1, 3, 224, 224))})  # perform a valid shape_infer() before profile()
    m.graph.profile()  # perform a valid profile() before print_node_map()
    m.graph.print_node_map(f=f'models/swin/nodemap_{modelname}.csv', metric='FLOPs',
                           exclude_ops=['Constant'])  # save to csv file
    m.graph.print_op_histogram(f=f'models/swin/nodehist_{modelname}.csv')

if __name__ == '__main__':
    profile_swin()
