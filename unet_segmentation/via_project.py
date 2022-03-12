import json
import os
from enum import Enum
from typing import Any, List, Dict
import numpy as np

class RegionType(Enum):
    POINT = 1
    RECTANGLE = 2
    CIRCLE = 3
    ELLIPSE = 4
    LINE = 5
    POLYLINE = 6
    POLYGON = 7
    EXTREME_RECTANGLE = 8
    EXTREME_CIRCLE = 9


class ViaView:
    view_id: str
    files_associated: list
    region_type: RegionType
    x_list: list
    y_list: list
    all_points: list

    def __init__(self, view_id: str):
        self.view_id = view_id
        self.files_associated = []
        self.x_list = []
        self.y_list = []
        self.all_points = []

    def has_same_id(self, other_view_id: str):
        return self.view_id.__eq__(other_view_id)

    def set_region_type(self, rt: int):
        if rt == 1:
            self.region_type = RegionType.POINT
        elif rt == 2:
            self.region_type = RegionType.RECTANGLE
        elif rt == 3:
            self.region_type = RegionType.CIRCLE
        elif rt == 4:
            self.region_type = RegionType.ELLIPSE
        elif rt == 5:
            self.region_type = RegionType.LINE
        elif rt == 6:
            self.region_type = RegionType.POLYLINE
        elif rt == 7:
            self.region_type = RegionType.POLYGON
        elif rt == 8:
            self.region_type = RegionType.EXTREME_RECTANGLE
        elif rt == 9:
            self.region_type = RegionType.EXTREME_CIRCLE
        else:
            print("project json malformed: regiontype {} incorrect", rt)
            exit()

    def set_points_coordinates(self, xy: list):
        """
        Set coordinates for this view. It is assumed the first digit in the list has been removed (the regionType)
        :param xy: coordinate list
        """
        pair_count = 0
        same_pair = False
        for coord in xy:
            coord = int(np.round(coord))
            if same_pair:
                # This is about Y
                self.y_list.insert(pair_count, coord)
                same_pair = False
                pair_count += 1
            else:
                # This is about X
                self.x_list.insert(pair_count, coord)
                same_pair = True
        for i, x in enumerate(self.x_list):
            self.all_points.append([x, self.y_list[i]])

    def get_coordinates_point(self, point_number: int):
        """
        Return a given point for the view
        :param point_number: the nth point coordinates to be retrieved
        :return:
        """
        return self.x_list[point_number], self.y_list[point_number]

    def __str__(self):
        return "ViaView {} | Associated file: {} | RegionType: {} | Points: {}" \
            .format(self.view_id, self.files_associated, self.region_type.name, self.all_points[:3])


class ViaAttribute:
    attr_number: int
    attr_name: str
    anchor_id: str
    type: int
    desc: str
    options: list
    default_option_id: str

    def __init__(self, attr_number, data: dict):
        self.attr_number = attr_number
        self.attr_name = data["aname"]
        self.anchor_id = data["anchor_id"]
        self.type = data["type"]
        self.desc = data["desc"]
        self.options = data["options"]
        self.default_option_id = data["default_option_id"]

    def __str__(self):
        return "ViaAttribute {} | {}, type {} and description {}" \
            .format(self.attr_number, self.attr_name, self.type, self.desc)


class ViaFile:
    file_id: str
    file_name: str
    type: int
    loc: int
    src: str

    def __init__(self, data: dict):
        self.file_id = data["fid"]
        self.file_name = data["fname"]
        self.type = data["type"]
        self.loc = data["loc"]
        self.src = data["src"]

    def __str__(self):
        return "ViaFile {} | {} of type {}".format(self.file_id, self.file_name, self.type)




class ViaProject:
    # Project section
    project_name: str
    views_list: List[ViaView]
    views_count: int
    # Attributes section
    attribute_list: List[ViaAttribute]
    attribute_count: int
    # Files
    files_list: List[ViaFile]
    files_count: int

    raw_data: dict

    def __init__(self, json_path):
        with open(json_path) as f:
            self.raw_data = json.load(f)

        # Project section
        self.project_name = self.raw_data["project"]["pname"]
        view_simple_list: dict = self.raw_data["project"]["vid_list"]
        self.views_count = len(view_simple_list)
        self.views_list = []
        for key in view_simple_list:
            self.views_list.append(ViaView(key))
        # Attributes section
        attributes_dict: dict = self.raw_data["attribute"]
        self.attribute_count = len(attributes_dict)
        self.attribute_list = []
        for key in attributes_dict.keys():
            self.attribute_list.append(
                ViaAttribute(key, attributes_dict[key])
            )
        # Files section
        files_dict: dict = self.raw_data["file"]
        self.files_count = len(files_dict)
        self.files_list = []
        for key in files_dict.keys():
            self.files_list.append(
                ViaFile(files_dict[key])
            )
        # Views section
        # 1. Metadata
        metadata: dict = self.raw_data["metadata"]
        for key in metadata.keys():
            vid: str = metadata[key]["vid"]
            for view in self.views_list:
                if view.has_same_id(vid):
                    if not metadata[key]:
                        print("There is no metadata for key {}".format(key))
                        continue
                    view.set_region_type(metadata[key]["xy"][0])
                    view.set_points_coordinates(metadata[key]["xy"][1:])
        # 2. Associated files
        pair_view_files: dict = self.raw_data["view"]
        for view_id in pair_view_files.keys():
            for view in self.views_list:
                if view.has_same_id(view_id):
                    view.files_associated = (pair_view_files[view_id]["fid_list"])

    def find_file_by_id(self, fid: int):
        for file in self.files_list:
            if file.file_id == fid:
                return file
        return False

    def __str__(self):
        return "ViaProject {} | With {} views, {} files, {} attributes"\
            .format(self.project_name, self.views_count, self.files_count, self.attribute_count)


def fusion_two_via_projects(project_1: ViaProject, project_2: ViaProject):
    """
    Fusion 2 projects into 1, project 1 remains the principal project with added data from project_2
    :param project_1:
    :param project_2:
    :return:
    """
    maxViewID = project_1.views_count - 1
    counter = maxViewID
    # fusion of views
    for view_l in project_2.views_list:
        view: ViaView = view_l
        view.view_id = counter
        project_1.views_list.append(view)
        counter += 1
    maxFileID = project_1.files_count - 1
    counter = maxFileID
    for file_assoc in project_2.files_list:
        file: ViaFile = file_assoc
        file.file_id = counter
        project_1.files_list.append(file)
    if not hasattr(project_1, "nb_fusion"):
        project_1
    return project_1


if __name__ == '__main__':
    json_path = os.path.join(os.getcwd(), "via_project_salamandres.json")
    json_path2 = os.path.join(os.getcwd(), "projets_rhetos", "Alexander", "via_project_23Feb2022_16h12m00s.json")
    json_path3 = os.path.join(os.getcwd(), "projets_rhetos", "Clément", "via_project_23Feb2022_15h58m21s.json")
    json_path4 = os.path.join(os.getcwd(), "projets_rhetos", "david", "via_project_23Feb2022_16h09m17s.json")
    json_path5 = os.path.join(os.getcwd(), "projets_rhetos", "Justin", "via_project_23Feb2022_15h56m09s.json")
    json_path6 = os.path.join(os.getcwd(), "projets_rhetos", "Romane", "via_project_23Feb2022_16h00m26s.json")
    project = ViaProject(json_path)
    project2 = ViaProject(json_path2)
    project3 = ViaProject(json_path3)
    project4 = ViaProject(json_path4)
    project5 = ViaProject(json_path5)
    project6 = ViaProject(json_path6)
