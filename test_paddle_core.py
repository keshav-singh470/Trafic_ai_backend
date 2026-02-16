import paddle.inference
print("Creating Config...")
try:
    config = paddle.inference.Config()
    print("Disabling GPU...")
    config.disable_gpu()
    print("Enabling MKLDNN...")
    config.enable_mkldnn()
    print("Creating Predictor (should fail gracefully if model missing, but NOT CRASH)...")
    # create_predictor usually takes a processed model
    # We can try to create one with empty/dummy paths?
    # Actually create_predictor requires valid model.
    # But it shouldn't SEGFAULT.
    # Let's try to load the model that WAS cached.
    model_dir = r"C:\Users\keshav singh\.paddlex\official_models\PP-LCNet_x1_0_doc_ori"
    model_file = model_dir + r"\inference.pdmodel"
    params_file = model_dir + r"\inference.pdiparams"
    
    import os
    if os.path.exists(model_file) and os.path.exists(params_file):
        print(f"Loading model from {model_dir}")
        config.set_model(model_file, params_file)
        pred = paddle.inference.create_predictor(config)
        print("Predictor Created Successfully!")
    else:
        print("Model files not found, cannot test predictor creation fully.")
except Exception as e:
    import traceback
    traceback.print_exc()
