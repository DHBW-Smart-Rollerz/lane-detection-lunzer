import pandas as pd
import xml.etree.ElementTree as ET
import os
import re

#Paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
data_path = os.path.join(base_path,"data")

#Class
class DataParser():
    cleaned_dataframe = pd.DataFrame()

    def __init__(self, task_names_to_load_list):
        raw_data_list = []
        for task in task_names_to_load_list:
            parsed_task_data = self._parse_task_data(task)
            raw_data_list.extend(parsed_task_data)
        raw_df = pd.DataFrame.from_dict(raw_data_list)
        self.cleaned_dataframe = self._clean_unlabeled_data(raw_df)
        return

    def _parse_task_data(self, task_name_to_load):
        parsed_task_data = []
        root = self._read_xml(task_name_to_load)
        for child in root:
            if child.tag == "image":
                dataset = {}
                dataset["task_name"] = self._get_task_name(child.attrib["name"])
                dataset["image_id"] = child.attrib["id"]
                dataset["image_path"] = self._img_path_from_name(child.attrib["name"], task_name_to_load)
                dataset["image_width"] = child.attrib["width"]
                dataset["image_height"] = child.attrib["height"]
                for subchild in child:
                    if subchild.tag == "polyline":
                        dataset[subchild.attrib["label"]] = subchild.attrib["points"]
                parsed_task_data.append(dataset)
        return parsed_task_data

    def _read_xml(self, task_name_to_load):
        label_folder_path = os.path.join(data_path,task_name_to_load,"label")
        label_location=os.path.join(label_folder_path, "annotations.xml")
        tree = ET.parse(label_location)
        root = tree.getroot() 
        return root

    def _get_task_name(self, attribute_name_value):
        return(attribute_name_value[8:15])

    def _img_path_from_name(self, attribute_name_value, task):
        img_folder_path = os.path.join(data_path,task,"img")
        path_ending = attribute_name_value[8::]
        return(os.path.join(img_folder_path, path_ending))

    def _clean_unlabeled_data(self, dataframe_to_clean:pd.DataFrame):
        lane_cols = ["left lane", "right lane", "center lane"]
        _point_re = re.compile(r"\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?")

        def _haspoints(value) -> bool:
            if pd.isna(value):
                return False
            s = str(value).strip()
            if not s or s.lower() in {"[]", "{}", "none", "nan"}:
                return False
            return bool(_point_re.search(s))
    
        has_any = dataframe_to_clean[lane_cols].map(_haspoints)
        nonempty_count = has_any.sum(axis = 1)
        df_clean = dataframe_to_clean[nonempty_count >= 2].reset_index(drop=True)
        return df_clean

    def get_data(self):
        return self.cleaned_dataframe

if __name__ == "__main__":
    task_names_to_load = ["2023-05", "2023-06", "2023-10", "2023-12", "2025-05"]
    data = DataParser(task_names_to_load).get_data()
    df = pd.DataFrame.from_dict(data)
    df.to_excel("excel.xlsx")
    print(df)
    