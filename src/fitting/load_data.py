import pandas as pd
import xml.etree.ElementTree as ET
import os
import re

#Paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
data_path = os.path.join(base_path,"data")

#Class
class DataParser():
    #cleaned_dataframe = pd.DataFrame()

    def __init__(self, task_names_to_load_list, jobs_to_ommit: list = []):
        raw_data_list = []
        for task in task_names_to_load_list:
            parsed_task_data = self._parse_task_data(task, jobs_to_ommit)
            raw_data_list.extend(parsed_task_data)
        raw_df = pd.DataFrame.from_dict(raw_data_list)
        self.cleaned_dataframe = self._clean_unlabeled_data(raw_df)
        return

    def _parse_task_data(self, task_name_to_load :list, jobs_to_ommit: list) -> list:
        parsed_task_data = []
        root = self._read_xml(task_name_to_load)
        excluded_image_ids = self._get_image_ids_to_ommit(root, jobs_to_ommit)
        for child in root:
            if child.tag == "image" and int(child.attrib["id"]) not in excluded_image_ids :
                dataset = {}
                dataset["task_name"] = self._get_task_name_from_root(root)
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
    
    def _get_image_ids_to_ommit(self, xml_root, jobs_to_ommit: list) -> set[int]:
        """
        Liefert die Menge aller image_ids (Frames), die ausgelassen werden sollen,
        basierend auf den Job-/Segment-IDs in 'jobs_to_ommit'.
        Akzeptiert Job-IDs als int, "123" oder URLs wie ".../api/jobs/123".
        """
        root = xml_root

        # --- Job-ID-Input normalisieren -> Set[int]
        jobs_set: set[int] = set()
        for j in (jobs_to_ommit or []):
            if isinstance(j, str):
                m = re.search(r'(\d+)$', j.strip())  # unterstützt auch URLs
                if m:
                    j = int(m.group(1))
                else:
                    try:
                        j = int(j)
                    except Exception:
                        continue
            try:
                jobs_set.add(int(j))
            except Exception:
                pass

        if not jobs_set:
            return set()

        # --- Segmente lesen und image_ids (start..stop inkl.) sammeln
        excluded_image_ids: set[int] = set()
        try:
            meta = root.find("meta")
            task = meta.find("task") if meta is not None else None
            segments = task.find("segments") if task is not None else None

            if segments is not None:
                for seg in segments.findall("segment"):
                    seg_id_txt = seg.findtext("id")
                    try:
                        seg_id = int(seg_id_txt)
                    except Exception:
                        continue

                    if seg_id in jobs_set:
                        try:
                            start = int(seg.findtext("start"))
                            stop = int(seg.findtext("stop"))
                        except Exception:
                            continue
                        excluded_image_ids.update(range(start, stop + 1))
        except Exception:
            # Wenn Meta/Task/Segments fehlen, gibt es nichts zu omittten
            return set()

        return excluded_image_ids

    def _get_task_name_from_root(self, root) -> str:
        """
        Liest <meta>/<task>/<name> aus dem CVAT-XML.
        Gibt '' zurück, wenn nicht vorhanden.
        """
        name = root.findtext('./meta/task/name')
        return name.strip() if name else ''

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
    
def get_current_data():
    """Lädt mit den vorhandenen Task Names und Jobs to ommit die Daten

    Returns:
        PandasDataframe: Daten
    """
    task_names_to_load = ["2023-05", "2023-06", "2023-10", "2023-12", "2025-05"]
    jobs_to_ommit = [119, 120, 121, 111, 112, 113, 114, 115, 116, 117]
    data = DataParser(task_names_to_load, jobs_to_ommit).cleaned_dataframe
    
    return data

if __name__ == "__main__":
    data = get_current_data()
    data.to_excel("excel.xlsx")
    print(data)
    