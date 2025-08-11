import cv2

def retreive_masks(image_path, model,imgsz, input_point, input_label ):
    device = "cpu"
    # Load
    everything_results = model(image_path, device=device, retina_masks=True, imgsz=imgsz, conf=0.0, iou=0.0,)
    prompt_process = FastSAMPrompt(image_path, everything_results, device=device)

    # text prompt
    # masks = prompt_process.text_prompt(text='a shell')

    # point prompt  
    masks = prompt_process.point_prompt(points = input_point, pointlabel = input_label)

    if masks.any() == False:
        masks = None

    return masks

def process_and_save_mask(image_path, model, input_root="data_total", output_root="data_total_masks"):

    masks = None
    try_index = 0

    # Point prompt parameters
    image = cv2.imread(image_path)

    h, w, _ = image.shape

    near = h/1.5 # h/1.6
    far = h/20
    input_point = np.array([[w*0+near,h*0+near],[w*1-near,h*0+near],[w*1-near,h*1-near],[w*0+near,h*1-near],
                            [w*0+far,h*0+far],[w*1-far,h*0+far],[w*1-far,h*1-far],[w*0+far,h*1-far]], dtype=np.int16)
    input_label = np.array([1,1,1,1,
                            0,0,0,0])
    
    img_szs_list = [512,256,128]

    while (masks is None) and (try_index<3):

        imgsz = img_szs_list[try_index]

        masks = retreive_masks(image_path, model, imgsz, input_point, input_label)
        
        try_index += 1

    if masks is None:
        return  # No mask detected
    

    # Combine all masks into one (binary)
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255

    # Derive output path
    rel_path = Path(image_path).relative_to(input_root)
    output_path = Path(output_root) / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save mask image (as PNG)
    cv2.imwrite(str(output_path.with_suffix('.png')), combined_mask)
    #print("output_path :",output_path)