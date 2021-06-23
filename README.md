# Requirements
1. Installed MongoDB and created collection names `image_features` inside database `image_retrieval`.
2. Installed all of Python modules in file `requirements.txt`. You may use below command to install these modules:
```
    pip install -r requirements.txt
```
# Dataset
- All of images must be stored in folder `./data/{dataset_name}/{sub_dataset_name}/{class_name}/`.
- Folder `./data/{dataset_name}/` must contain dataset description files that corresponding to sub dataset (i.e. trainset, testset,...).
# Scripts
## Extract features
- Running `python extract_features.py` to extract images features and save to database.
- You can configure before running this script by creating or modifying configuration files and pass their paths as parameters of function `extract_and_save`. Below are the configuration files that affect this function:
    + Database (ex: `./config/database.json`).
    + Features Extractors (ex: `./config/feature_extractor.json`).
    + Dataset (ex: `./data/cifar-10/train.json`).
- You can also update images features collect (adding, updating, removing,...) by modifying configuration files and running this script again.
## Run tests
- Run `python retrieve.py` to test image retrieval.
- Like `extract_features` script, you can also modify some configuration files to change behavior of this script.
- You can modify matching stragies and rules by:
    + Extending class `Matcher`.
    + Modifying metric class (ex: `class EuclideanDistance`).
    + Modifying list of features and list of extractors that you want the system to be insterested in.

## Start Web UI
- Run `python3 app.py [args]` to start the server.
- Arguments:
    * ```-i, --ip```: Defaults to ```localhost```
    * ```-o, --port```: Defaults to ```5000```
