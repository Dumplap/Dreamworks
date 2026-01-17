This was originally built as a tool to map out an impossible shot of someone skiing in front of a lunar eclipse, however could easily be repurposed for aligning full moons with mountains or even the 2045 solar eclipse!

## What You'll Need to Get Started

* [Install OSGeo4W](https://trac.osgeo.org/osgeo4w/)
  * this will add some python libraries and allow us to use combine multiple files covering a larger area into a single mosaic and let the program see it as one file
* Download digital elevation models | [USGS](https://apps.nationalmap.gov/downloader/)
  * I used data specially from "Elevation Products.../1-meter DEM" use the "extent" filter feature to quickly search all the LiDAR scans of the selected area (you can also use the show feature to see if these models cover the area you are interested in) 


## DEM Analysis

* Before we run the script we need python to be able to see all the files as one collage (this will also test if you installed the required additional libraries correctly!) using terminal in the "Main Project" directory run the following line for you locations

    ```gdalbuildvrt RMNP/mosaic.vrt "RMNP/DEM Models/*.tif"```

Now you can then run the program with adjustments to the regression's and tweak the other variables (min dist, max dist, step size, tolerance, and even time range) Note that time is handeled differently because of the 2 modes, so its further down in the file and penumbral csv ranges will need manual adjustments.
  * Note that the first time you run the analysis it will take significantly longer (3-5min) as it will be finding ridgelines and generating a shaded relief map to overlay the alignments onto, these will then be cached speeding it up from here on out.
<br><br>

The Program will create a /RESULTS/{site} folder where it will make the following files for each site:

* PENUMBRAL alignments.png
* PENUMBRAL alignments.tif
* PENUMBRAL---gps 3min.csv
* PENUMBRAL---gps 10min.csv
* Ridges.png
* TOTALITY alignments.png
* TOTALITY alignments.tif
* TOTALITY---gps 3min.csv
* TOTALITY---gps 10min.csv
  
The .png and .tif files will have the same visual data, however the tif files will also have geo location data so they can be imported into mapping software such as CalTopo as map sheets. Ridges.png is a debugging file, if you are not getting the desired output changing the -2 on the laplacian will adjust the sensitivity. .csv files are reference files for the .cpp script to generate mapping cords.


## Further Visualizing The Data

After running completing an analysis there's 2 other scripts that allow the data to be visualized in other ways. The ```CalTopo Cords.cpp``` will take in a lon/lat of the camera position in DD format and output a  .json  that can be imported directly into CalTopo allowing the points to be visually linked. Those same lat/lon can then also be fed into ```Moon Grapher.py``` which will create a graph of the ridgeline and the moon's position at 5min before intersect, intersect, and disappear. Note that the slope of the ridgeline in the graph will not visually match what the actual slope of the ridge.


## Additional Information

### File Structure

using this stucture will ensure compatibility with the commands given and script loops.

```
Main Project
├── Aspen/                  # Name of location, matches the list in the master.py
│   ├── DEM Models/        
│   │   └── .tif            # All tif files of a location
│   └── moasic.vrt          # these will be generated with a later command
├── RMNP/                   # same as the other locations, ex. Aspen
│   ├── DEM Models/
│   │   └── .tif
│   └── moasic.vrt
├── Results/                # output folder
├── Lunar Plots/            # output folder
├── Moon Grapher.py
└── Master.py
ALT Folder
└── CalTopo Cords.cpp
```

### A Note on The Regression Equations

I used Stellarium to gather azimuth and altitude angles for the beginning end and peak of the march 3rd lunar eclipse totality period. I then fit these to a second order regression equation, this allows the program to iterate through each minute of the eclipse and test even more alignments. Further improvements could be made to this by using more data gathed to have multi-variable regressions to allow for longitude, latitude, and elevation to be set which could then also be cross referenced with the cordinates of the ridgeline being tested!
