import onnxruntime
import torch
import cv2
import numpy as np


def run_onnx(img, model):
    img = cv2.resize(raw_image, (320, 184))
    img = img[88:, :, :]
    print(img.shape)
    raw_img = img.copy()
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32)
    img /= 255.0  # Normalize pixel values to [0, 1]

    input_data = np.expand_dims(img, axis=0)  # Add batch dimension
    print("Input shape: ", input_data.shape)

    output = session.run([output_name], {input_name: input_data})
    segmentation_result = output[0]
    print("Output shape: ", segmentation_result.shape)
    x0 = torch.from_numpy(segmentation_result)
    _,da_predict=torch.max(x0, 1)
    DA = da_predict.byte().cpu().data.numpy()[0]*255
    raw_img[DA>100]=[255, 0, 0]

    cv2.imwrite("segment.png", raw_img)

if __name__ == '__main__':

    model_name = 'model_4_DSAU_fulldataset'
    session = onnxruntime.InferenceSession(f"best_model/{model_name}.onnx")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type

    raw_image = cv2.imread('/home/ceec/tri/Lightweight-Attention-Unet/Datasets/Data_Vong2/images/img_0.0.jpg')
    run_onnx(raw_image, session)