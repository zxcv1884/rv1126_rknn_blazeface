# RV1126 RKNN Blazeface


## Build

Please refer to [rv1126_rknn](https://github.com/Chinadavid1991/rv1126_rknn) for build.  
Move OpenCV to rv1126.

## Run

1. Move files to rv1126

```shell
adb push blazeface /userdata/
adb push models/anchors.bin /userdata/
adb push models/blazeface.rknn /userdata/   #rknn runtime version <= 1.6 use blazeface-16.rknn
adb push images/hinton.jpg /userdata/
```

2. Run

```shell
adb shell
cd /userdata
./blazeface blazeface.rknn hinton.jpg
```

3. Check output

```shell
exit
adb pull /userdata/output.jpg .
```

![output](https://github.com/zxcv1884/rv1126_rknn_blazeface/blob/master/output.jpg)
