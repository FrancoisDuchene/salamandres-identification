# représente une photo
class Photo:
    filepath: str
    date: str
    prospection_name: str
    prospection_num: int
    salamandre_num: int
    cliche_num: int
    appareil: str
    gps_lat: float
    gps_long: float

    def __init__(self, filepath, date, prospection_name, prospection_num, salamandre_num, cliche_num, appareil, gps_lat, gps_long):
        self.filepath = filepath
        self.date = date
        self.prospection_name = prospection_name
        self.prospection_num = prospection_num
        self.salamandre_num = salamandre_num
        self.cliche_num = cliche_num
        self.appareil = appareil
        self.gps_lat = gps_lat
        self.gps_long = gps_long


    def __len__(self):
        return 1


    def __str__(self):
        return "{} ({}), {} n{}, salamandre {} (cliche {}), pris par {} à ({},{})"\
            .format(self.filepath, self.date, self.prospection_name, self.prospection_num,
                    self.salamandre_num, self.cliche_num, self.appareil, self.gps_lat, self.gps_long)


    def __repr__(self):
        return "<Photo {} ({})>".format(self.filepath, self.date)