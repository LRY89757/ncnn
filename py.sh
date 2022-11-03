# rm -rf build
cmake -H. -Bbuild -DNCNN_PYTHON=ON
# make -j20
cmake --build build --parallel 20
# cd ..

# python
# pip install .
# cd ~/projects/My-CUDA-Examples/ONNX/PNNX/GridSample
# # conda activate pnnx
# python ncnn_test.py

# test_layer
cd /home/lry/projects/ncnn/build/tests
./test_gridsample
