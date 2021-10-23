# This is a sample Python script.
import os.path

import exiftool
from typing import Tuple, List

import folium

from Photo import Photo

folder_s3 = "prospection 3"
folder_s4 = "prospection 4"
folder_s5 = "prospection 5"

files = [
    # prospection 3 - IPHONE
    os.path.join("..", folder_s3, "s3_salam_1_iphone_1.HEIC"),
    os.path.join("..", folder_s3, "s3_salam_2_iphone_1.HEIC"),
    os.path.join("..", folder_s3, "s3_salam_3_iphone_1.HEIC"),
    os.path.join("..", folder_s3, "s3_salam_4_iphone_1.HEIC"),
    os.path.join("..", folder_s3, "s3_salam_5_iphone_1.HEIC"),
    os.path.join("..", folder_s3, "s3_salam_6_iphone_1.HEIC"),
    # prospection 3 - Xiaomi

    # prospection 4
    os.path.join("..", folder_s4, "s4_salam_1_xiaomi_4.jpg"),
    os.path.join("..", folder_s4, "s4_salam_2_xiaomi_4.jpg"),
    # prospection 5 - Xiaomi
    os.path.join("..", folder_s5, "s5_salam_1_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_2_xiaomi_5.jpg"),
    os.path.join("..", folder_s5, "s5_salam_3_xiaomi_5.jpg"),
    os.path.join("..", folder_s5, "s5_salam_4_xiaomi_4.jpg"),
    os.path.join("..", folder_s5, "s5_salam_5_xiaomi_5.jpg"),
    os.path.join("..", folder_s5, "s5_salam_6_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_7_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_8_xiaomi_1.jpg"),
    os.path.join("..", folder_s5, "s5_salam_9_xiaomi_4.jpg"),
    os.path.join("..", folder_s5, "s5_salam_10_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_11_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_12_xiaomi_4.jpg"),
    os.path.join("..", folder_s5, "s5_salam_13_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_14_xiaomi_3.jpg"),
    os.path.join("..", folder_s5, "s5_salam_15_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_16_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_17_xiaomi_1.jpg"),
    os.path.join("..", folder_s5, "s5_salam_18_xiaomi_1.jpg"),
    os.path.join("..", folder_s5, "s5_salam_19_xiaomi_3.jpg"),
    os.path.join("..", folder_s5, "s5_salam_20_xiaomi_1.jpg"),
    os.path.join("..", folder_s5, "s5_salam_21_xiaomi_1.jpg"),
    os.path.join("..", folder_s5, "s5_salam_22_xiaomi_1.jpg"),
    os.path.join("..", folder_s5, "s5_salam_23_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_24_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_25_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_26_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_27_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_28_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_29_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_30_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_31_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_32_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_33_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_34_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_35_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_36_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_37_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_38_xiaomi_2.jpg"),
    os.path.join("..", folder_s5, "s5_salam_39_xiaomi_2.jpg"),
    # prospection 5 - Oneplus

    # prospection 6 - Xiaomi

    # prospection 6 - Oneplus

    # prospection 6 - Samsung

]


def analyse_exif_on_photos(filenames: list):
    # coordonates: List[Tuple[float, float]] = []
    photos: List[Photo] = []
    with exiftool.ExifTool() as et:
        metadata = et.get_metadata_batch(filenames)
        for d in metadata:
            # print("{} {} || ({}, {})".format(
            #     d["SourceFile"],
            #     d["EXIF:DateTimeOriginal"],
            #     #d["Composite:GPSPosition"],
            #     d["EXIF:GPSLatitude"],
            #     d["EXIF:GPSLongitude"]
            # ))
            # print("{}".format(d.keys()))
            # gps_coord = (d["EXIF:GPSLatitude"], d["EXIF:GPSLongitude"])
            # coordonates.append(gps_coord)

            stripped_filepath: str = d["SourceFile"].split("/")
            print(d["SourceFile"])
            prospection_name_stripped = stripped_filepath[1].split(" ")
            prospection_name: str
            prospection_num: int
            if prospection_name_stripped.__len__() == 1:   # pas une prospection numerotée
                prospection_name = prospection_name_stripped[0]
                prospection_num = 1
            else:
                prospection_name = prospection_name_stripped[0]
                prospection_num = int(prospection_name_stripped[1])
            stripped_filename: list = stripped_filepath[2].split(".")
            stripped_filename = stripped_filename[0].split("_")
            salamandre_num = stripped_filename[2]
            appareil = stripped_filename[3]
            cliche_num = stripped_filename[4]
            date = d["EXIF:ModifyDate"]
            gps_lat = d["EXIF:GPSLatitude"]
            gps_long = d["EXIF:GPSLongitude"]
            photos.append(Photo(d["SourceFile"], date, prospection_name, prospection_num, salamandre_num, cliche_num, appareil, gps_lat, gps_long))
    return photos


def create_map(photos: List[Photo]):
    my_map = folium.Map(location=(photos[0].gps_lat, photos[0].gps_long), zoom_start=16)
    for ph in photos:
        if ph.prospection_name.__eq__("prospection"):
            if ph.prospection_num == 3:
                folium.Marker(location=(ph.gps_lat, ph.gps_long),
                              popup="prosp:{} salam:{} date:{}".format(ph.prospection_name, ph.salamandre_num, ph.date),
                              icon=folium.Icon(icon="calendar")
                              ).add_to(my_map)
            elif ph.prospection_num == 4:
                folium.Marker(location=(ph.gps_lat, ph.gps_long),
                              popup="prosp:{} salam:{} date:{}".format(ph.prospection_name, ph.salamandre_num, ph.date),
                              icon=folium.Icon(icon="adjust", color="red")
                              ).add_to(my_map)
            elif ph.prospection_num == 5:
                folium.Marker(location=(ph.gps_lat, ph.gps_long),
                              popup="prosp:{} salam:{} date:{}".format(ph.prospection_name, ph.salamandre_num, ph.date),
                              icon=folium.Icon(icon="file", color="green")
                              ).add_to(my_map)
    my_map.save("salamandres_bois_lauzelle_carte.html")
    return my_map


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    coordonates = analyse_exif_on_photos(files)
    print(coordonates)
    create_map(coordonates)
