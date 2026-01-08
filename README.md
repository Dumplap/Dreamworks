## What you'll need to get started

* [Install OSGeo4W](https://trac.osgeo.org/osgeo4w/)
  * this will add some python libraries and allow us to use combine multiple files covering a larger area into a single mosaic and let the program see it as one file
* Download digital elevation models | [USGS](https://apps.nationalmap.gov/downloader/)
  * I used data specially from "Elevation Products.../1-meter DEM" use the "extent" filter feature to quickly search all the LiDAR scans of the selected area (you can also use the show feature to see if these models cover the area you are interested in) 


# DEM Analysis

### I used the following file structure
    using this stucture will ensure compatibility with the commands given and script loops.
```Main Project
├── Aspen/                  # Name of location, matches the list in the master.py
│   ├── DEM Models/        
│   │   └── .tif            # All tif files of the area
│   └── moasic.vrt          # these will be generated with a later command
├── RMNP/                   # same as the other locations, ex. Aspen
│   ├── DEM Models/
│   │   └── .tif
│   └── moasic.vrt
├── Results/                # output folder
└── Master.py
```

```gdalbuildvrt RMNP/mosaic.vrt "RMNP/DEM Models/*.tif"```
