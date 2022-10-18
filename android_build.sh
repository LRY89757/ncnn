##### android aarch64
export ANDROID_NDK=/home/lry/envs/android-ndk-r24/
mkdir -p build-android-aarch64
cd build-android-aarch64
# pushd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 ..
make -j$(nproc)
make install
# popd
