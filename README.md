#   **emotionDetection**
    BRANCH: channelOptimization
    VERSION: 0.7.1
# VERSION HISTORY:


## 0.7.2
    CNN Model with 22 channels
    functions
        model10 -> model32
        model10_all -> model32_all
## 0.8.1
    Output files updated
## 0.8.0
    CNN Model with 8 channels
## 0.7.1
    README
        Version History Updated
## 0.7.0
    CNN Model with 10 channels
    functions
        model32 -> model10
        model32_all -> model10_all
## 0.6.0
    CNN Model with 32 channels
    functions
        + model32()
        + model32_all()
        + cnnmodel_all()
    functions.sampling
        + comments
        
## 0.5.0
    CNN Model with 46 channels
## 0.4.0
    CNN Model with 59 channels
## 0.3.3
    + gc.collect()
## 0.3.2
    functions
        + cnnmodel()
    - functions.cnn
    functions.cnnReduce -> cnnOptimize
        create_cnn_reduce -> create_cnn
        train_cnn_reduce -> train_cnn
            - BatchNormalization()
    functions.sampling
        + comments
## 0.3.1
    functions.sampling
        generate_batched_samples_from_directory
            sample -> finalsample
## 0.3.0
    CNN Model for 60 channels
## 0.2.9
    Renamed Functions
    functions.cnn
        cnn2d() -> create_cnn()
        runcnn2d() -> train_cnn()
    + functions.cnnReduce
        + create_cnn_reduce()
        + train_cnn_reduce()
    functions.model
        modelplot_acc() -> plot_accuracy()
        modelplot_loss() -> plot_loss()
        plotAll() -> plot_all()
        savemodel() -> save_model()
        loadmodel() -> load_model()
    functions.pathlabelchannel
        pathlabelchannel() -> get_path_label_channel()
        channelRemove() -> remove_channels()
        modelTestPLC() -> get_path_label_channel_by_index()
    functions.cnn61 -> reduceComplex
        cnn2d1x -> create_cnn_complex
        runcnn2d61 -> train_cnn_complex
        dirsample61 -> generate_batched_sample_from_directory()



## 0.2.8
    README
        + Version History
    functions.cnn61
        cnn2d1x
            + BatchNormalization()
            + Dropout(0.2)
## 0.2.7
    functions
        - modelTest()
    functions.modelplot -> functions.model
        + plotAll()
        + saveModel()
        + loadModel()
## 0.2.6
    functions.cnn
        runcnn2d61() -> runcnn2d()
        - modelplot_acc()
        - modelplot_loss()
    + functions.cnn61
        + cnn2d1x()
        + runcnn2d61()
    + functions.modelplot
        + modelplot_acc()
        + modelplot_loss()
## 0.2.5
    CNN model.fit() modified with callbacks
## 0.2.4
    functions
        + modelTest()
    functions.pathlabelchannel
        + channelRemove()
        + modelTestPLC()
## 0.2.3
    Added .gitignore
## 0.2.2
    - __pycache__
## 0.2.1
    Sampling Function Updated
    functions.sampling
        sampling() -> sampling61()
        - dirsample()

## 0.2.0
    CNN Model for 61 channel
    functions.cnn
        runcnn2d() -> runcnn2d61()
    functions.sampling
        dirsample() -> dirsample61()
        - sampleshuffle()
## 0.1.5
    Branch Name Changed: channelOptimization
## 0.1.4
    + functions.cnn
        + cnn2d()
        + runcnn2d()
        + modelplot_acc()
        + modelplot_loss()
    functions.sampling
        + sample_shuffle()
        + dirsample()
    functions.pathlabelchannel
        - show()
## 0.1.3
    README.md changed
## 0.1.2
    + functions.pathlabelchannel
        + pathlabelchannel()
        + show()
    + functions.sampling
        + sampling()
        + sampling61()
## 0.1.1
    Code and Functions Declared
## 0.1.0
    Branch Created: emotionDetection