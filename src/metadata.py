import exiftool


with exiftool.ExifTool() as et:
    metadata = et.get_metadata("C:/Users/legom/Desktop/Lots_of_people/GP074188.MP4")

print(metadata)
# for d in metadata:
#     print("{:20.20} {:20.20}".format(d["SourceFile"],
#                                      d["EXIF:DateTimeOriginal"]))