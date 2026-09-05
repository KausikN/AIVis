"""
Stream lit GUI for hosting AIVis
"""

# Imports
from streamlit_common_utils.streamlit_common_ui_setup import *

from AIVis import *

# Main Vars
UI_CONFIG = UIConfig("./StreamLitGUI/UIConfig.json")
UI_DATA = UIData("./StreamLitGUI/UIData.json")
UI_CACHE = UICache(UI_DATA.get("paths.cache"))

# Main Functions
def main():
    build_sidebar_app(UI_CONFIG, globals())


#############################################################################################################################
# Util Functions
def LoadDefaultDatasets():
    """
    Load Default Datasets
    """
    datasets_dir = UI_DATA.get("paths.default.datasets_dir", "Data/DefaultDatasets/")
    dataset_paths = []
    if os.path.isdir(datasets_dir):
        for file_name in os.listdir(datasets_dir):
            if file_name.endswith(".csv"):
                dataset_paths.append(os.path.join(datasets_dir, file_name))
    UI_CACHE["default_datasets"] = dataset_paths
    UI_CACHE.save()

    return dataset_paths


def GetFileNames(file_paths):
    """
    Get File Names from File Paths
    """
    return [os.path.basename(file_path) for file_path in file_paths]


# Main Functions
def GenerateDatasetBasicInfo(USERINPUT_DatasetData):
    """
    Generate Basic Dataset Info
    """
    columns = USERINPUT_DatasetData.columns
    columns_type = autodetect_column_types(USERINPUT_DatasetData)
    columns_data = []

    for i in range(len(columns)):
        column_name = columns[i]
        columns_data.append({
            "name": column_name,
            "type": columns_type[column_name],
            "dtype": str(USERINPUT_DatasetData.dtypes[column_name]),
            "categorizable": is_categorizable(USERINPUT_DatasetData, column_name, 25, 0.025),
            "unique_values_count": len(USERINPUT_DatasetData[column_name].unique())
        })

    dataset_basic_info = {
        "columns_count": len(columns),
        "rows_count": USERINPUT_DatasetData.shape[0],
        "columns_data": columns_data
    }

    return dataset_basic_info


# UI Functions
def UI_LoadDataset():
    """
    UI - Load Dataset
    """
    default_dataset_paths = LoadDefaultDatasets()
    default_dataset_names = GetFileNames(default_dataset_paths)
    dataset_names = list(default_dataset_names)
    saved_csv_path = UI_DATA.get("paths.save.csv", "Data/SavedDataset.csv")
    if os.path.exists(saved_csv_path):
        dataset_names = ["Uploaded Dataset"] + dataset_names

    USERINPUT_DatasetChoice = st.sidebar.selectbox("Choose a Dataset", dataset_names)
    if USERINPUT_DatasetChoice == "Uploaded Dataset":
        USERINPUT_DatasetData = load_csv(saved_csv_path)
    else:
        USERINPUT_DatasetData = load_csv(default_dataset_paths[default_dataset_names.index(USERINPUT_DatasetChoice)])

    return USERINPUT_DatasetData, USERINPUT_DatasetChoice


def UI_DatasetDetails(dataset_basic_info):
    """
    UI - Dataset Details
    """
    st.markdown("## Column Details")
    col_size = (1, 3)

    col1, col2 = st.columns(col_size)
    col1.markdown("Column Count")
    col2.markdown(str(dataset_basic_info["columns_count"]))

    col1, col2 = st.columns(col_size)
    col1.markdown("Row Count")
    col2.markdown(str(dataset_basic_info["rows_count"]))


def UI_DisplayColumnDetails(dataset_basic_info):
    """
    UI - Display Column Details
    """
    columns_data = dataset_basic_info["columns_data"]
    column_names = [column["name"] for column in columns_data]
    USERINPUT_ColumnChoice = st.selectbox("Choose a Data Column", column_names)
    column_data = columns_data[column_names.index(USERINPUT_ColumnChoice)]

    st.markdown("## Column Details")
    col_size = (1, 3)

    col1, col2 = st.columns(col_size)
    col1.markdown("Column Name")
    col2.markdown(column_data["name"])

    col1, col2 = st.columns(col_size)
    col1.markdown("Column DataType")
    col2.markdown(column_data["dtype"])

    col1, col2 = st.columns(col_size)
    col1.markdown("Column Type")
    col2.markdown(column_data["type"])

    col1, col2 = st.columns(col_size)
    col1.markdown("Column Categorizable")
    col2.markdown(":heavy_check_mark:" if column_data["categorizable"] else ":x:")

    col1, col2 = st.columns(col_size)
    col1.markdown("Unique Values Count")
    unique_count = column_data["unique_values_count"]
    total_count = dataset_basic_info["rows_count"]
    percent = round(((unique_count * 100) / total_count), 2)
    latex_code = "$\\frac{}{} = {}$".format(
        "{" + str(unique_count) + "}",
        "{" + str(total_count) + "}",
        str(percent) + "\%"
    )
    col2.markdown(latex_code)


# Repo Based Functions
def upload_dataset():
    # Title
    st.header("Upload Dataset")

    # Load Inputs
    USERINPUT_DatasetData = st.file_uploader("Upload Dataset", ["csv"])

    # Process Inputs
    saved_csv_path = UI_DATA.get("paths.save.csv", "Data/SavedDataset.csv")
    if USERINPUT_DatasetData is not None:
        with open(saved_csv_path, "wb") as file:
            file.write(USERINPUT_DatasetData.read())
    elif not os.path.exists(saved_csv_path):
        st.info("Upload a dataset :sweat_smile:")
        return

    USERINPUT_DatasetData = load_csv(saved_csv_path)

    # Display Outputs
    st.markdown("## Uploaded Dataset")
    if USERINPUT_DatasetData is not None:
        st.table(USERINPUT_DatasetData.head())


def view_dataset():
    # Title
    st.header("View Dataset")

    # Load Inputs
    USERINPUT_DatasetData, DatasetName = UI_LoadDataset()

    # Display Outputs
    st.markdown("## " + DatasetName)
    col1, col2 = st.columns(2)
    USERINPUT_DatasetShowRangeStart = col1.number_input("Display Row Start", 0, USERINPUT_DatasetData.shape[0] - 1, 0, 1)
    USERINPUT_DatasetShowRangeCount = col2.number_input("Display Row Count", 1, USERINPUT_DatasetData.shape[0] - USERINPUT_DatasetShowRangeStart, 1, 1)
    st.table(USERINPUT_DatasetData.iloc[USERINPUT_DatasetShowRangeStart:USERINPUT_DatasetShowRangeStart + USERINPUT_DatasetShowRangeCount, :])


def dataset_basic_info():
    # Title
    st.header("Dataset Basic Info")

    # Load Inputs
    USERINPUT_DatasetData, DatasetName = UI_LoadDataset()

    # Process Inputs
    dataset_basic_info = {}
    dataset_info_cache = UI_CACHE.get("dataset_basic_info", {})
    
    if DatasetName not in dataset_info_cache or st.button("Regenerate Dataset Info"):
        GeneratedText = st.empty()
        GeneratedText.info("Generating Dataset Info...")
        dataset_basic_info = GenerateDatasetBasicInfo(USERINPUT_DatasetData)
        dataset_info_cache[DatasetName] = dataset_basic_info
        UI_CACHE["dataset_basic_info"] = dataset_info_cache
        UI_CACHE.save()
        GeneratedText.success("Regenerated Dataset Info!")
    else:
        dataset_basic_info = dataset_info_cache[DatasetName]

    # Display Outputs
    st.markdown("## Dataset Overview")
    UI_DatasetDetails(dataset_basic_info)

    st.markdown("## Columns Overview")
    UI_DisplayColumnDetails(dataset_basic_info)


#############################################################################################################################
# Run Code
if __name__ == "__main__":
    main()