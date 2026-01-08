## What you'll need to get started

* [Install OSGeo4W](https://trac.osgeo.org/osgeo4w/)
  * this will add some python libraries and allow us to use combine multiple files covering a larger area into a single mosaic and let the program see it as one file
* Download digital elevation models | [USGS](https://apps.nationalmap.gov/downloader/)
  * I used data specially from "Elevation Products.../1-meter DEM" use the "extent" filter feature to quickly search all the LiDAR scans of the selected area (you can also use the show feature to see if these models cover the area you are interested in) 


# DEM Analysis

### I used the following file structure
Main Project
├── docs/                 # Core content files
│   ├── index.md          # Homepage / Introduction
│   ├── getting-started/
│   │   ├── install.md
│   │   └── config.md
│   ├── guides/
│   │   ├── advanced-usage.md
│   │   └── troubleshooting.md
│   └── api/
│       └── reference.md
├── assets/               # Non-markdown files
│   ├── images/           # Screenshots and diagrams
│   │   └── architecture.png
│   ├── pdfs/             # Downloadable resources
│   └── templates/        # Reusable snippets or boilerplate
├── .github/              # Automation (if using GitHub)
│   └── workflows/
│       └── deploy.yml
├── .gitignore            # Files to exclude from version control
├── mkdocs.yml            # Config (if using MkDocs) or docusaurus.config.js
├── README.md             # Project overview for developers
└── SUMMARY.md            # Table of contents (used by GitBook/mdBook)

```gdalbuildvrt RMNP/mosaic.vrt "RMNP/DEM Models/*.tif"```
