# MCX simulation
**Environment use**
| OS | compiler | GPU |
| -------- | -------- | -------- |
| linux ubuntu 18.04       | spyder     | RTX2080 ( based on NVIDIA's Turing architecture)       |  
  
  
The core executive file of simulation was compiled from **C language(modified by Eric Su) click** [here](https://github.com/fangq/mcx/blob/master/src/Makefile)  
You can directly use the exe file in this repo(compiled from our lab modified C code.)   
**If you download from the original repo(Prof. Fang) it can still run simulation, but the result may have some difference.**  

About how to do mcx simulation I split it into six steps.
## Generate simulation array (S1)
From the literature, we get the range of tissue optical parameter. 
```py
skin_mus = np.array(np.linspace(skin_mus_bound[0],skin_mus_bound[1],7))
subcuit_mus = np.array(np.linspace(subcuit_mus_bound[0],subcuit_mus_bound[1],7))
muscle_mus = np.array(np.linspace(muscle_mus_bound[0],muscle_mus_bound[1],5))
vessel_mus = np.array(np.linspace(vessel_mus_bound[0],vessel_mus_bound[1],5))

skin_mua = np.array(np.linspace(skin_mua_bound[0],skin_mua_bound[1],3))
subcuit_mua = np.array(np.linspace(subcuit_mua_bound[0],subcuit_mua_bound[1],3))
muscle_mua = np.array(np.linspace(muscle_mua_bound[0],muscle_mua_bound[1],5))
ijv_mua = np.array(np.linspace(ijv_mua_bound[0],ijv_mua_bound[1],7))
cca_mua = np.array(np.linspace(cca_mua_bound[0],cca_mua_bound[1],7))
```
Here, I divid each layer each parameter 7*7*5*5*3*3*5*7*7 = (1225)*(2205) = 2.7 million combinations

## Simulation setting (S2)
You can setting the photon number you want to simulate, and decide the subject name, folder name.

```py
sessionID = "ctchen_1e9_ijv_large_to_small"
PhotonNum = 1e9
#%% run
subject = "ctchen"
mus_set = np.load("mus_set.npy")
# copy config.json ijv_dense_symmetric_detectors_backgroundfiber_pmc.json model_parameters.json mua_test.json to each sim
copylist = ["config.json",
            "ijv_dense_symmetric_detectors_backgroundfiber_pmc.json",
            "model_parameters.json",
            "mua_test.json"]
```

## Run simulation (S3 and mcx_ultrasound_opsbased.py)
This is the main file to start to run simulations.

Here you can decide if you want to consider NA ( NA_enable )
if you consider NA then, this program will automatic delete photons not in this NA (save memory space)
```py
ID = "ctchen_1e9_ijv_large_to_small" #ID = "ctchen_ijv_small_to_large"
mus_start = 508
mus_end = 625


NA_enable = 1
#%% run
NA = 0.22
mus_set = np.load("mus_set.npy")
muaPath = "mua_test.json"
```

Also you can decide the CV threshold you want to constraint.
If you set runningNum=0 which mean we would not consider the CV.
Because we have lots of SDS(21), the longer SDS the CV will be very high.
Therefore, you also can choose what SDS you want to make it as baseline to lower the CV.
```py
#  Setting
    foldername = "LUT"
    session = f"run_{run_idx}"
    sessionID = os.path.join(ID,foldername,session)
    runningNum = 0 # (Integer or False)self.session
    cvThreshold = 3
    cvBaselineSDS = 'sds_15'
    repeatTimes = 10

```

## Do WMC (S4)
After finising MC, we do WMC.

```py
ID = "ctchen_1e9_ijv_large_to_small"
datasetpath = "ctchen_dataset_large"
mus_start = 1
mus_end = 1225
```

## Check CV (S6)
![image](https://github.com/dicky1031/IJV_simulation/blob/main/pic/CV_test1.png)
![image](https://github.com/dicky1031/IJV_simulation/blob/main/pic/CV_test2.png)
