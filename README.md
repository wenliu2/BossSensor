requirements:
python, opencv-python, keras, tensorflow, h5py.

step1: 
run "python face_capture.py" to capture images of your face. It will capture 100 images of your face and save into ./faces/ directory.

move ./faces/* to ./data/boss/


run it again,  and capture another people's face, and move the files to ./data/others/

step 2:
run "python boss_train.py" to train, this will output a model file to ./store/model.h5

step 3:
run python camera_reader.py to do face recognization. There will be "Yes" or "No" output in the stdout, to indicate if you are recoginized.


