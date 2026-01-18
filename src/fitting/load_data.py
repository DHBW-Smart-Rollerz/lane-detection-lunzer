import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

# Paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.path.join(base_path, "data")


# Class


class DataParser:
    """
    Parses CVAT XML annotation tasks into a cleaned pandas DataFrame.

    The DataParser loads multiple annotation tasks, extracts image-level
    polyline labels, optionally excludes specific jobs/segments, and removes
    samples with insufficient lane annotations.
    """

    # cleaned_dataframe = pd.DataFrame()

    def __init__(self, task_names_to_load_list, jobs_to_ommit: list = []):
        """
        Initialize the DataParser and immediately load and clean the data.

        All provided tasks are parsed, combined into a single DataFrame,
        and filtered to remove images with insufficient annotation coverage.

        Args:
            task_names_to_load_list (list[str]):
                List of task directory names to load from the dataset root.
            jobs_to_ommit (list, optional):
                List of job or segment IDs to exclude. Entries may be integers,
                numeric strings, or job URLs. Defaults to an empty list.
        """
        raw_data_list = []
        for task in task_names_to_load_list:
            parsed_task_data = self._parse_task_data(task, jobs_to_ommit)
            raw_data_list.extend(parsed_task_data)
        raw_df = pd.DataFrame.from_dict(raw_data_list)
        self.cleaned_dataframe = self._clean_unlabeled_data(raw_df)
        return

    def _parse_task_data(self, task_name_to_load: list, jobs_to_ommit: list) -> list:
        """
        Parse a single CVAT task and extract image-level annotation data.

        Images belonging to excluded jobs or segments are skipped. For each
        remaining image, metadata and polyline labels are collected into a
        dictionary.

        Args:
            task_name_to_load (str):
                Name of the task directory to parse.
            jobs_to_ommit (list):
                List of job or segment IDs to exclude.

        Returns:
            list[dict]:
                A list of dictionaries, one per image, containing metadata
                and extracted polyline annotations.
        """
        parsed_task_data = []
        root = self._read_xml(task_name_to_load)
        excluded_image_ids = self._get_image_ids_to_ommit(root, jobs_to_ommit)
        for child in root:
            if (
                child.tag == "image"
                and int(child.attrib["id"]) not in excluded_image_ids
            ):
                dataset = {}
                dataset["task_name"] = self._get_task_name_from_root(root)
                dataset["image_id"] = child.attrib["id"]
                dataset["image_path"] = self._img_path_from_name(
                    child.attrib["name"], task_name_to_load
                )
                dataset["image_width"] = child.attrib["width"]
                dataset["image_height"] = child.attrib["height"]
                for subchild in child:
                    if subchild.tag == "polyline":
                        dataset[subchild.attrib["label"]] = subchild.attrib["points"]
                parsed_task_data.append(dataset)
        return parsed_task_data

    def _read_xml(self, task_name_to_load):
        """
        Load and parse the CVAT XML annotation file for a given task.

        Args:
            task_name_to_load (str):
                Name of the task directory.

        Returns:
            xml.etree.ElementTree.Element:
                Root element of the parsed XML tree.
        """
        label_folder_path = os.path.join(data_path, task_name_to_load, "label")
        label_location = os.path.join(label_folder_path, "annotations.xml")
        tree = ET.parse(label_location)
        root = tree.getroot()
        return root

    def _get_image_ids_to_ommit(self, xml_root, jobs_to_ommit: list) -> set[int]:
        """
        Determine image IDs that should be excluded based on job or segment IDs.

        Job identifiers may be provided as integers, numeric strings, or URLs.
        For each matching segment, all image IDs in the inclusive range
        [start, stop] are excluded.

        Args:
            xml_root (xml.etree.ElementTree.Element):
                Root element of the parsed CVAT XML.
            jobs_to_ommit (list):
                List of job or segment identifiers to exclude.

        Returns:
            set[int]:
                Set of image IDs that should be omitted.
        """
        root = xml_root

        # --- Job-ID-Input normalisieren -> Set[int]
        jobs_set: set[int] = set()
        for j in jobs_to_ommit or []:
            if isinstance(j, str):
                m = re.search(r"(\d+)$", j.strip())  # unterstützt auch URLs
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
        Extract the task name from the CVAT XML metadata.

        Args:
            root (xml.etree.ElementTree.Element):
                Root element of the parsed XML.

        Returns:
            str:
                Task name, or an empty string if not present.
        """
        name = root.findtext("./meta/task/name")
        return name.strip() if name else ""

    def _img_path_from_name(self, attribute_name_value, task):
        """
        Construct the absolute image path from a CVAT image name attribute.

        Args:
            attribute_name_value (str):
                Value of the image 'name' attribute from the XML.
            task (str):
                Task directory name.

        Returns:
            str:
                Absolute path to the corresponding image file.
        """
        img_folder_path = os.path.join(data_path, task, "img")
        path_ending = attribute_name_value[8::]
        return os.path.join(img_folder_path, path_ending)

    def _clean_unlabeled_data(self, dataframe_to_clean: pd.DataFrame):
        """
        Remove samples with insufficient lane annotations.

        Rows are retained only if at least two of the expected lane columns
        contain valid polyline point data.

        Args:
            dataframe_to_clean (pandas.DataFrame):
                Raw DataFrame containing extracted annotations.

        Returns:
            pandas.DataFrame:
                Cleaned DataFrame with poorly labeled samples removed.
        """
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
        nonempty_count = has_any.sum(axis=1)
        df_clean = dataframe_to_clean[nonempty_count >= 2].reset_index(drop=True)
        return df_clean


def get_current_data():
    """
    Load the current default dataset configuration.

    This function uses predefined task names and job exclusions to construct
    a DataParser instance and return the cleaned annotation DataFrame.

    Returns:
        pandas.DataFrame:
            Cleaned dataset containing the combined annotations.
    """
    task_names_to_load = ["2023-05", "2023-06", "2023-10", "2023-12", "2025-05"]
    jobs_to_ommit = [119, 120, 121, 111, 112, 113, 114, 115, 116, 117]
    data = DataParser(task_names_to_load, jobs_to_ommit).cleaned_dataframe

    return data
