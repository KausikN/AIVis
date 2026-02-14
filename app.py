"""
Stream lit GUI for hosting AIVis
"""

# Imports
import os
import json
import streamlit as st

from AIVis import *

# Main Vars
config = json.load(open("./StreamLitGUI/UIConfig.json", "r"))

# Main Functions
def main():
    # Create Sidebar
    selected_box = st.sidebar.selectbox(
    "Choose one of the following",
        tuple(
            [config["PROJECT_NAME"]] + 
            config["PROJECT_MODES"]
        )
    )

    # Load Cache
    LoadCache()
    
    if selected_box == config["PROJECT_NAME"]:
        HomePage()
    else:
        correspondingFuncName = selected_box.replace(" ", "_").lower()
        if correspondingFuncName in globals().keys():
            globals()[correspondingFuncName]()
 

def HomePage():
    st.title(config["PROJECT_NAME"])
    st.markdown("Github Repo: " + "[" + config["PROJECT_LINK"] + "](" + config["PROJECT_LINK"] + ")")
    st.markdown(config["PROJECT_DESC"])

    # st.write(open(config["PROJECT_README"], "r").read())

#############################################################################################################################
# Repo Based Vars
PATHS = {
    "cache": "StreamLitGUI/CacheData/Cache.json",
    "default": {
        "datasets_dir": "StreamLitGUI/DefaultData/DefaultDatasets/",
    },
    "save": {
        "csv": "StreamLitGUI/DefaultData/SavedDataset.csv"
    }
}

# Util Vars
CACHE = {}

# Util Functions
def LoadCache():
    '''
    Load Cache
    '''
    global CACHE
    CACHE = json.load(open(PATHS["cache"], "r"))

def SaveCache():
    '''
    Save Cache
    '''
    global CACHE
    json.dump(CACHE, open(PATHS["cache"], "w"), indent=4)

def LoadDefaultDatasets():
    '''
    Load Default Datasets
    '''
    global CACHE
    CACHE["default_datasets"] = []
    for f in os.listdir(PATHS["default"]["datasets_dir"]):
        if f.endswith(".csv"):
            CACHE["default_datasets"].append(PATHS["default"]["datasets_dir"] + f)
    SaveCache()

    return CACHE["default_datasets"]

def GetFileNames(file_paths):
    '''
    Get File Names from File Paths
    '''
    return [os.path.basename(file_path) for file_path in file_paths]

# Main Functions
def GenerateDatasetBasicInfo(USERINPUT_DatasetData):
    '''
    Generate Basic Dataset Info
    '''
    Columns = USERINPUT_DatasetData.columns
    ColumnsType = autodetect_column_types(USERINPUT_DatasetData)
    ColumnsData = []

    for i in range(len(Columns)):
        c = Columns[i]
        ColumnsData.append({
            "name": Columns[i],
            "type": ColumnsType[c],
            "dtype": str(USERINPUT_DatasetData.dtypes[i]),
            "categorizable": is_categorizable(USERINPUT_DatasetData, c, 25, 0.025),
            "unique_values_count": len(USERINPUT_DatasetData[c].unique())
        })

    DatasetBasicInfo = {
        "columns_count": len(Columns),
        "rows_count": USERINPUT_DatasetData.shape[0],
        "columns_data": ColumnsData
    }

    return DatasetBasicInfo

# UI Functions
def UI_LoadDataset():
    '''
    UI - Load Dataset
    '''
    DefaultDatasetPaths = LoadDefaultDatasets()
    DefaultDatasetNames = GetFileNames(DefaultDatasetPaths)
    DatasetNames = list(DefaultDatasetNames)
    if os.path.exists(PATHS["save"]["csv"]): DatasetNames = ["Uploaded Dataset"] + DatasetNames

    USERINPUT_DatasetChoice = st.sidebar.selectbox("Choose a Dataset", DatasetNames)
    if USERINPUT_DatasetChoice == "Uploaded Dataset":
        USERINPUT_DatasetData = load_csv(PATHS["save"]["csv"])
    else:
        USERINPUT_DatasetData = load_csv(DefaultDatasetPaths[DefaultDatasetNames.index(USERINPUT_DatasetChoice)])

    return USERINPUT_DatasetData, USERINPUT_DatasetChoice

def UI_DatasetDetails(DatasetBasicInfo):
    '''
    UI - Dataset Details
    '''
    st.markdown("## Column Details")
    colSize = (1, 3)

    col1, col2 = st.columns(colSize)
    col1.markdown("Column Count")
    col2.markdown(str(DatasetBasicInfo["columns_count"]))

    col1, col2 = st.columns(colSize)
    col1.markdown("Row Count")
    col2.markdown(str(DatasetBasicInfo["rows_count"]))

def UI_DisplayColumnDetails(DatasetBasicInfo):
    '''
    UI - Display Column Details
    '''
    ColumnsData = DatasetBasicInfo["columns_data"]
    ColumnNames = [c["name"] for c in ColumnsData]
    USERINPUT_ColumnChoice = st.selectbox("Choose a Data Column", ColumnNames)
    ColumnData = ColumnsData[ColumnNames.index(USERINPUT_ColumnChoice)]

    st.markdown("## Column Details")
    colSize = (1, 3)

    col1, col2 = st.columns(colSize)
    col1.markdown("Column Name")
    col2.markdown(ColumnData["name"])

    col1, col2 = st.columns(colSize)
    col1.markdown("Column DataType")
    col2.markdown(ColumnData["dtype"])

    col1, col2 = st.columns(colSize)
    col1.markdown("Column Type")
    col2.markdown(ColumnData["type"])

    col1, col2 = st.columns(colSize)
    col1.markdown("Column Categorizable")
    col2.markdown(":heavy_check_mark:" if ColumnData["categorizable"] else ":x:")

    col1, col2 = st.columns(colSize)
    col1.markdown("Unique Values Count")
    uniqueCount = ColumnData["unique_values_count"]
    totalCount = DatasetBasicInfo["rows_count"]
    percent = round(((uniqueCount*100) / totalCount), 2)
    latexCode = "$\\frac{}{} = {}$".format(
        "{" + str(uniqueCount) + "}",
        "{" + str(totalCount) + "}",
        str(percent) + "\%"
    )
    col2.markdown(latexCode)


# Repo Based Functions
def upload_dataset():
    # Title
    st.header("Upload Dataset")

    # Load Inputs
    USERINPUT_DatasetData = st.file_uploader("Upload Dataset", ["csv"])

    # Process Inputs
    if USERINPUT_DatasetData is not None:
        open(PATHS["save"]["csv"], "wb").write(USERINPUT_DatasetData.read())
    elif not os.path.exists(PATHS["save"]["csv"]):
        st.markdown("Upload a dataset :sweat_smile:")
        return

    USERINPUT_DatasetData = load_csv(PATHS["save"]["csv"])

    # Display Outputs
    st.markdown("## Uploaded Dataset")
    if USERINPUT_DatasetData is not None: st.table(USERINPUT_DatasetData.head())

def view_dataset():
    # Title
    st.header("View Dataset")

    # Load Inputs
    USERINPUT_DatasetData, DatasetName = UI_LoadDataset()

    # Display Outputs
    st.markdown("## " + DatasetName)
    col1, col2 = st.columns(2)
    USERINPUT_DatasetShowRangeStart = col1.number_input("Display Row Start", 0, USERINPUT_DatasetData.shape[0]-1, 0, 1)
    USERINPUT_DatasetShowRangeCount = col2.number_input("Display Row Count", 1, USERINPUT_DatasetData.shape[0]-USERINPUT_DatasetShowRangeStart, 1, 1)
    st.table(USERINPUT_DatasetData.iloc[USERINPUT_DatasetShowRangeStart:USERINPUT_DatasetShowRangeStart+USERINPUT_DatasetShowRangeCount, :])

def dataset_basic_info():
    # Title
    st.header("Dataset Basic Info")

    # Load Inputs
    USERINPUT_DatasetData, DatasetName = UI_LoadDataset()

    # Process Inputs
    DatasetBasicInfo = {}
    if DatasetName in CACHE["dataset_basic_info"]:
        if st.button("Regenerate Dataset Info"):
            GeneratedText = st.empty()
            GeneratedText.markdown("Generating Dataset Info...")
            DatasetBasicInfo = GenerateDatasetBasicInfo(USERINPUT_DatasetData)
            CACHE["dataset_basic_info"][DatasetName] = DatasetBasicInfo
            SaveCache()
            GeneratedText.markdown("Regenerated Dataset Info!")
        else:
            DatasetBasicInfo = CACHE["dataset_basic_info"][DatasetName]
    else:
        GeneratedText = st.empty()
        GeneratedText.markdown("Generating Dataset Info...")
        DatasetBasicInfo = GenerateDatasetBasicInfo(USERINPUT_DatasetData)
        CACHE["dataset_basic_info"][DatasetName] = DatasetBasicInfo
        SaveCache()
        GeneratedText.markdown("Regenerated Dataset Info!")
        
    # Display Outputs
    st.markdown("## Dataset Overview")
    UI_DatasetDetails(DatasetBasicInfo)

    st.markdown("## Columns Overview")
    UI_DisplayColumnDetails(DatasetBasicInfo)

    
#############################################################################################################################
# Run Code
if __name__ == "__main__":
    main()