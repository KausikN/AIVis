"""
Stream lit GUI for hosting AIVis
"""

# Imports
import os
import streamlit as st
import json

import AIVis

# Main Vars
config = json.load(open('./StreamLitGUI/UIConfig.json', 'r'))

# Main Functions
def main():
    # Create Sidebar
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
        tuple(
            [config['PROJECT_NAME']] + 
            config['PROJECT_MODES']
        )
    )

    # Load Cache
    LoadCache()
    
    if selected_box == config['PROJECT_NAME']:
        HomePage()
    else:
        correspondingFuncName = selected_box.replace(' ', '_').lower()
        if correspondingFuncName in globals().keys():
            globals()[correspondingFuncName]()
 

def HomePage():
    st.title(config['PROJECT_NAME'])
    st.markdown('Github Repo: ' + "[" + config['PROJECT_LINK'] + "](" + config['PROJECT_LINK'] + ")")
    st.markdown(config['PROJECT_DESC'])

    # st.write(open(config['PROJECT_README'], 'r').read())

#############################################################################################################################
# Repo Based Vars
CACHE_PATH = "StreamLitGUI/CacheData/Cache.json"
DEFAULT_DATASETS_DIR = "StreamLitGUI/DefaultData/DefaultDatasets/"
SAVE_DATASET_PATH = "StreamLitGUI/DefaultData/SavedDataset.csv"

# Util Vars
CACHE = {}

# Util Functions
def LoadCache():
    global CACHE
    CACHE = json.load(open(CACHE_PATH, 'r'))

def SaveCache():
    global CACHE
    json.dump(CACHE, open(CACHE_PATH, 'w'), indent=4)

def LoadDefaultDatasets():
    global CACHE
    CACHE['default_datasets'] = []
    for f in os.listdir(DEFAULT_DATASETS_DIR):
        if f.endswith(".csv"):
            CACHE['default_datasets'].append(DEFAULT_DATASETS_DIR + f)
    SaveCache()
    return CACHE['default_datasets']

def GetFileNames(file_paths):
    return [os.path.basename(file_path) for file_path in file_paths]

# Main Functions
def GenerateDatasetBasicInfo(USERINPUT_DatasetData):
    Columns = USERINPUT_DatasetData.columns
    ColumnsType = AIVis.GetDatasetTypes(USERINPUT_DatasetData)
    ColumnsDataType = [str(dt) for dt in USERINPUT_DatasetData.dtypes]
    catThresh = [25, 0.025]
    ColumnsCategorizable = [AIVis.IsCategorizable(USERINPUT_DatasetData, c, catThresh) for c in USERINPUT_DatasetData.columns]
    ColumnsUniqueValuesCount = [len(AIVis.GetUniqueValues(USERINPUT_DatasetData, c)) for c in USERINPUT_DatasetData.columns]

    ColumnsData = []
    for i in range(len(Columns)):
        colData = {
            'name': Columns[i],
            'type': ColumnsType[i],
            'dtype': ColumnsDataType[i],
            'categorizable': ColumnsCategorizable[i],
            'unique_values_count': ColumnsUniqueValuesCount[i]
        }
        ColumnsData.append(colData)

    DatasetBasicInfo = {
        'columns_count': len(Columns),
        'rows_count': USERINPUT_DatasetData.shape[0],
        'columns_data': ColumnsData
    }
    return DatasetBasicInfo

# UI Functions
def UI_LoadDataset():
    DefaultDatasetPaths = LoadDefaultDatasets()
    DefaultDatasetNames = GetFileNames(DefaultDatasetPaths)
    DatasetNames = list(DefaultDatasetNames)
    if os.path.exists(SAVE_DATASET_PATH):
        DatasetNames = ["Uploaded Dataset"] + DatasetNames
    USERINPUT_DatasetChoice = st.sidebar.selectbox("Choose a Dataset", DatasetNames)
    if USERINPUT_DatasetChoice == "Uploaded Dataset":
        USERINPUT_DatasetData = AIVis.LoadDataset(SAVE_DATASET_PATH)
    else:
        USERINPUT_DatasetData = AIVis.LoadDataset(DefaultDatasetPaths[DefaultDatasetNames.index(USERINPUT_DatasetChoice)])
    return USERINPUT_DatasetData, USERINPUT_DatasetChoice

def UI_DatasetDetails(DatasetBasicInfo):
    st.markdown("## Column Details")
    colSize = (1, 3)

    col1, col2 = st.columns(colSize)
    col1.markdown("Column Count")
    col2.markdown(str(DatasetBasicInfo['columns_count']))

    col1, col2 = st.columns(colSize)
    col1.markdown("Row Count")
    col2.markdown(str(DatasetBasicInfo['rows_count']))

def UI_DisplayColumnDetails(DatasetBasicInfo):
    ColumnsData = DatasetBasicInfo['columns_data']
    ColumnNames = [c['name'] for c in ColumnsData]
    USERINPUT_ColumnChoice = st.selectbox("Choose a Data Column", ColumnNames)
    ColumnData = ColumnsData[ColumnNames.index(USERINPUT_ColumnChoice)]

    st.markdown("## Column Details")
    colSize = (1, 3)

    col1, col2 = st.columns(colSize)
    col1.markdown("Column Name")
    col2.markdown(ColumnData['name'])

    col1, col2 = st.columns(colSize)
    col1.markdown("Column DataType")
    col2.markdown(ColumnData['dtype'])

    col1, col2 = st.columns(colSize)
    col1.markdown("Column Type")
    col2.markdown(ColumnData['type'])

    col1, col2 = st.columns(colSize)
    col1.markdown("Column Categorizable")
    col2.markdown(":heavy_check_mark:" if ColumnData['categorizable'] else ":x:")

    col1, col2 = st.columns(colSize)
    col1.markdown("Unique Values Count")
    uniqueCount = ColumnData['unique_values_count']
    totalCount = DatasetBasicInfo['rows_count']
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
    USERINPUT_DatasetData = st.file_uploader("Upload Dataset", ['csv'])

    # Process Inputs
    if USERINPUT_DatasetData is not None:
        open(SAVE_DATASET_PATH, 'wb').write(USERINPUT_DatasetData.read())
    elif not os.path.exists(SAVE_DATASET_PATH):
        st.markdown("Upload a dataset :sweat_smile:")
        return

    USERINPUT_DatasetData = AIVis.LoadDataset(SAVE_DATASET_PATH)

    # Display Outputs
    st.markdown("## Uploaded Dataset")
    if USERINPUT_DatasetData is not None:
        st.table(USERINPUT_DatasetData.head())

def view_dataset():
    # Title
    st.header("View Dataset")

    # Load Inputs
    USERINPUT_DatasetData, DatasetName = UI_LoadDataset()

    # Display Dataset Rows
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
    if DatasetName in CACHE['dataset_basic_info']:
        if st.button("Regenerate Dataset Info"):
            GeneratedText = st.empty()
            GeneratedText.markdown("Generating Dataset Info...")
            DatasetBasicInfo = GenerateDatasetBasicInfo(USERINPUT_DatasetData)
            CACHE['dataset_basic_info'][DatasetName] = DatasetBasicInfo
            SaveCache()
            GeneratedText.markdown("Regenerated Dataset Info!")
        else:
            DatasetBasicInfo = CACHE['dataset_basic_info'][DatasetName]
    else:
        GeneratedText = st.empty()
        GeneratedText.markdown("Generating Dataset Info...")
        DatasetBasicInfo = GenerateDatasetBasicInfo(USERINPUT_DatasetData)
        CACHE['dataset_basic_info'][DatasetName] = DatasetBasicInfo
        SaveCache()
        GeneratedText.markdown("Regenerated Dataset Info!")
        
    # Display Outputs
    st.markdown("## Dataset Overview")
    UI_DatasetDetails(DatasetBasicInfo)

    st.markdown("## Columns Overview")
    UI_DisplayColumnDetails(DatasetBasicInfo)

    
#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()