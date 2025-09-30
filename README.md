# rustcorder


Install whiper model files:
```
  mkdir -p models
  curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin -o models/ggml-base.bin
  ```
Install audio deps:
```
sudo apt install pkg-config libasound2-dev
```
Install c++ and build tools:
```
 sudo apt install build-essential cmake
 ```