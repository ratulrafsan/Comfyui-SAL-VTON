# Comfyui-SAL-VTON
This is my quick implementation of the SAL-VTON node for ComfyUI.

### See the paper[^1] for details 


### Installation
1. Clone this repository into the `custom_nodes` folder.
2. Navigate to the cloned folder and run `pip install -r requirements.txt`. Be sure you're in the ComfyUI venv!
3. Download these [landmark](https://www.modelscope.cn/api/v1/models/iic/cv_SAL-VTON_virtual-try-on/repo?Revision=master&FilePath=landmark.pth), [warp](https://www.modelscope.cn/api/v1/models/iic/cv_SAL-VTON_virtual-try-on/repo?Revision=master&FilePath=warp.pth), and [salvton](https://www.modelscope.cn/api/v1/models/iic/cv_SAL-VTON_virtual-try-on/repo?Revision=master&FilePath=pytorch_model.bin) models.
4. Create a folder, named `salvton`, in the ComfyUI `models` directory and copy all three downloaded models into it.
5. Profit?

### Things to be aware of
1. The garment should be 768x1024. If you don't have an image of the exact size, just resize it in ComfyUI.
2. The results are poor if the background of the `person` image is not white. Consider using rembg or SAM to mask it and replace it with a white background.
3. The garment mask is just the shape of the input garment. You can generate it with SAM or use rembg like I did in the workflow.

![workflow.png](media%2Fworkflow.png)

### Acknowledgement
[^1][Keyu Y. Tingwei G. et al. (2023). Linking Garment with Person via Semantically Associated Landmakrs for Virtual Try-On](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_Linking_Garment_With_Person_via_Semantically_Associated_Landmarks_for_Virtual_CVPR_2023_paper.pdf)

This is a simple wrapper around the inference code available on [ModelScope](https://github.com/modelscope/modelscope). Props to the original authors.
