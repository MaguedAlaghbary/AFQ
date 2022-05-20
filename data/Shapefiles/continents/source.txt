lpk file :
https://www.arcgis.com/home/item.html?id=a3cb207855b348a297ab85261743351d

convert lpk to shp formatusing OSGeow4 shell

first unzip with 7-zip

then access v107

ogr2ogr -f "ESRI Shapefile" continent continent.gdb
