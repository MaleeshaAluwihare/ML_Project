Stage 1: Positive Image Samples Preparation
Convert all positive images to a resolution of 1024x1024.

python download_image_resize.py
This script resizes all images in the positives/ directory and saves them to positives/1024x1024.

Stage 2: Annotate Images
Mark the nose area in all images and create positives_1024x1024.txt with image paths and nose coordinates.

opencv_annotation --annotations=positives_1024x1024.txt --images=positives/1024x1024
Stage 3: Rescale Positive Images
Rescale images and annotations to 512, 256, 128, and 64 sizes.

python script_cascade/positive_images_resize.py
Stage 4: Prepare Negative Samples
Add samples to the negatives/ folder. Ensure there are at least twice as many negatives as positives. Then, resize negative images and create negative.txt.

python script_cascade/negative_images_resize.py
python script_cascade/script_create_negative_txt.py
Stage 5: Train Cascade Classifier
Create a vector file and train the cascade classifier.

opencv_createsamples -info positives_64x64.txt -num 50 -w 24 -h 24 -vec positives_64x64.vec
opencv_traincascade -data script_cascade/output/64 -vec positives_64x64.vec -bg negatives.txt -numPos 50 -numNeg 1600 -numStages 10 -w 24 -h 24 -precalcValBufSize 2048 -precalcIdxBufSize 2048
Stage 6: Test Cascade Classifier
Test the trained cascade classifier.

python script_cascade/cascade_test.py
python script_cascade/cascade_test_loop.py
Stage 7: Debug and Retrain
If the recognizer misclassifies, add those images to the negatives and retrain from Stage 4.

python script_cascade/castcade_debug.py
Stage 8: Extract Noses
Use the trained classifier to extract noses from images.

python script_extract/nose_extract.py
Stage 9: Process Nose-Prints
Grayscale and apply adaptive thresholding to nose-prints.

python script_extract/node_to_print.py
Stage 10: Train Identifier Model
Train an identifier model using the KNN algorithm (or CNN for better results).

python script_identifier/train_identifier.py
Stage 11: Test the Model
Test the trained identifier model.

python script_identifier/model_test.py
Stage 12: Run the App
Run the UI application to test the model. Drag and drop test images to see the labels.

python app/HOMBENAI.py
